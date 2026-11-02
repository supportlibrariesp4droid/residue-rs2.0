//! Matrix product kernels written with explicit x86 vector instructions.
//!
//! The portable kernels rely on LLVM's auto-vectorizer, which handles one
//! loop over contiguous slices well but not a register-blocked tile: in the
//! Phase 0 experiment it vectorized tiles along the wrong dimension, with
//! gathers and scatters. Here the tile is spelled out with AVX2 or AVX-512
//! intrinsics, through `pulp`'s feature tokens: an intrinsic can only be
//! called with a token that proves the CPU supports it, so the code stays
//! free of `unsafe`.
//!
//! The arithmetic is the library's delayed reduction. A block of `RB × CB`
//! output sums stays in registers for the whole loop over `k`. Residues are
//! below `2^31`, so a product is below `2^62` and four of them fit a `u64`:
//! products are summed four at a time, then split into the 32-bit halves that
//! [`Channel::redc_halves`] reduces once at the end.

use core::arch::x86_64::{__m256i, __m512i};

use pulp::bytemuck::cast;
use pulp::x86::{V3, V4};

use crate::word::Channel;

/// `out = a·b` in one channel (`a` is `m×k`, `b` is `k×p`, row-major,
/// Montgomery residues). Returns `false`, leaving `out` untouched, when the CPU
/// has neither AVX2 nor AVX-512; the caller then uses the portable kernel.
///
/// Requires `k < p_mod` (the modulus), the condition of the division-free
/// reduction.
///
/// `chs` holds either one channel, used for every output row, or one channel
/// per output row (`m` of them): the accumulation does not depend on the
/// modulus, only the final reduction does, so the rows of one product can
/// belong to different primes, as in a base extension.
pub(crate) fn gemm(
    chs: &[Channel],
    m: usize,
    k: usize,
    p: usize,
    a: &[u32],
    b: &[u32],
    out: &mut [u32],
) -> bool {
    debug_assert!(chs.len() == 1 || chs.len() == m);
    let g = Gemm {
        chs,
        m,
        k,
        p,
        a,
        b,
        out,
    };
    let level = LEVEL.load(core::sync::atomic::Ordering::Relaxed);
    if level >= 2 {
        return false;
    }
    if let Some(v4) = V4::try_new().filter(|_| level == 0) {
        v4.vectorize(GemmV4 { v: v4, g });
        true
    } else if let Some(v3) = V3::try_new() {
        v3.vectorize(GemmV3 { v: v3, g });
        true
    } else {
        false
    }
}

/// 0: best available; 1: AVX2 even if AVX-512 is present; 2: portable kernel.
static LEVEL: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(0);

/// Test hook: cap the explicit SIMD kernels (0 best, 1 AVX2, 2 none), so that
/// every path can be tested on one machine. Not part of the API.
#[doc(hidden)]
pub fn __set_simd_level(level: u8) {
    LEVEL.store(level, core::sync::atomic::Ordering::Relaxed);
}

struct Gemm<'a> {
    chs: &'a [Channel],
    m: usize,
    k: usize,
    p: usize,
    a: &'a [u32],
    b: &'a [u32],
    out: &'a mut [u32],
}

impl Gemm<'_> {
    /// The channel that reduces output row `r`.
    #[inline(always)]
    fn ch(&self, r: usize) -> Channel {
        if self.chs.len() == 1 {
            self.chs[0]
        } else {
            self.chs[r]
        }
    }
}

/// Columns past the last full tile, one sum at a time.
#[inline(always)]
fn tail_columns(g: &mut Gemm<'_>, rows: core::ops::Range<usize>, j0: usize) {
    let (k, p) = (g.k, g.p);
    for r in rows {
        for j in j0..p {
            let (mut lo, mut hi) = (0u64, 0u64);
            for t in 0..k {
                let v = g.a[r * k + t] as u64 * g.b[t * p + j] as u64;
                lo += v & 0xFFFF_FFFF;
                hi += v >> 32;
            }
            g.out[r * p + j] = g.ch(r).redc_halves(lo, hi);
        }
    }
}

struct GemmV4<'a> {
    v: V4,
    g: Gemm<'a>,
}

impl pulp::NullaryFnOnce for GemmV4<'_> {
    type Output = ();
    #[inline(always)]
    fn call(self) {
        let GemmV4 { v, mut g } = self;
        const CB: usize = 16;
        let full = g.p - g.p % CB;
        let mut r0 = 0;
        while r0 + 4 <= g.m {
            for j0 in (0..full).step_by(CB) {
                tile_v4::<4>(v, &mut g, r0, j0);
            }
            tail_columns(&mut g, r0..r0 + 4, full);
            r0 += 4;
        }
        while r0 < g.m {
            for j0 in (0..full).step_by(CB) {
                tile_v4::<1>(v, &mut g, r0, j0);
            }
            tail_columns(&mut g, r0..r0 + 1, full);
            r0 += 1;
        }
    }
}

#[allow(clippy::needless_range_loop)]
#[inline(always)]
fn tile_v4<const RB: usize>(v: V4, g: &mut Gemm<'_>, r0: usize, j0: usize) {
    let f = v.avx512f;
    let (k, p) = (g.k, g.p);
    let zero = f._mm512_setzero_si512();
    let mask = f._mm512_set1_epi64(0xFFFF_FFFF);
    let mut lo: [[__m512i; 2]; RB] = [[zero; 2]; RB];
    let mut hi: [[__m512i; 2]; RB] = [[zero; 2]; RB];
    let a_rows: [&[u32]; RB] = core::array::from_fn(|r| &g.a[(r0 + r) * k..(r0 + r + 1) * k]);
    let mut b_rows = g.b.get(j0..).unwrap_or(&[]).chunks(p).take(k);
    let mut kk = 0;
    while kk < k {
        let mut t: [[__m512i; 2]; RB] = [[zero; 2]; RB];
        if kk + 4 <= k {
            let xs: [[u32; 4]; RB] =
                core::array::from_fn(|r| a_rows[r][kk..kk + 4].try_into().unwrap());
            for s in 0..4 {
                let b = b_rows.next().unwrap_or(&[]);
                step_v4(v, &mut t, core::array::from_fn(|r| xs[r][s]), b);
            }
            kk += 4;
        } else {
            for q in kk..k {
                let b = b_rows.next().unwrap_or(&[]);
                step_v4(v, &mut t, core::array::from_fn(|r| a_rows[r][q]), b);
            }
            kk = k;
        }
        for r in 0..RB {
            for h in 0..2 {
                lo[r][h] = f._mm512_add_epi64(lo[r][h], f._mm512_and_si512(t[r][h], mask));
                hi[r][h] = f._mm512_add_epi64(hi[r][h], f._mm512_srli_epi64::<32>(t[r][h]));
            }
        }
    }
    for r in 0..RB {
        let k4 = Redc4::new(v, g.ch(r0 + r));
        for h in 0..2 {
            let out: [u32; 8] = cast(f._mm512_cvtepi64_epi32(k4.redc_halves(lo[r][h], hi[r][h])));
            let o = (r0 + r) * p + j0 + 8 * h;
            g.out[o..o + 8].copy_from_slice(&out);
        }
    }
}

/// One step of the AVX-512 tile: `t[r] += x[r]·b[0..16]`.
#[inline(always)]
fn step_v4<const RB: usize>(v: V4, t: &mut [[__m512i; 2]; RB], x: [u32; RB], b: &[u32]) {
    let f = v.avx512f;
    let b: &[u32; 16] = b[..16].try_into().unwrap();
    let (b0, b1) = split16(b);
    let b0 = f._mm512_cvtepu32_epi64(cast::<[u32; 8], __m256i>(b0));
    let b1 = f._mm512_cvtepu32_epi64(cast::<[u32; 8], __m256i>(b1));
    for (tr, &x) in t.iter_mut().zip(&x) {
        let x = f._mm512_set1_epi64(x as i64);
        tr[0] = f._mm512_add_epi64(tr[0], f._mm512_mul_epu32(x, b0));
        tr[1] = f._mm512_add_epi64(tr[1], f._mm512_mul_epu32(x, b1));
    }
}

/// The two halves of sixteen residues.
#[inline(always)]
fn split16(b: &[u32; 16]) -> ([u32; 8], [u32; 8]) {
    let mut x = [0; 8];
    let mut y = [0; 8];
    x.copy_from_slice(&b[..8]);
    y.copy_from_slice(&b[8..]);
    (x, y)
}

struct GemmV3<'a> {
    v: V3,
    g: Gemm<'a>,
}

impl pulp::NullaryFnOnce for GemmV3<'_> {
    type Output = ();
    #[inline(always)]
    fn call(self) {
        let GemmV3 { v, mut g } = self;
        const CB: usize = 8;
        let full = g.p - g.p % CB;
        let mut r0 = 0;
        while r0 + 2 <= g.m {
            for j0 in (0..full).step_by(CB) {
                tile_v3::<2>(v, &mut g, r0, j0);
            }
            tail_columns(&mut g, r0..r0 + 2, full);
            r0 += 2;
        }
        while r0 < g.m {
            for j0 in (0..full).step_by(CB) {
                tile_v3::<1>(v, &mut g, r0, j0);
            }
            tail_columns(&mut g, r0..r0 + 1, full);
            r0 += 1;
        }
    }
}

#[allow(clippy::needless_range_loop)]
#[inline(always)]
fn tile_v3<const RB: usize>(v: V3, g: &mut Gemm<'_>, r0: usize, j0: usize) {
    let (x2, x1) = (v.avx2, v.avx);
    let (k, p) = (g.k, g.p);
    let zero = x1._mm256_setzero_si256();
    let mask = x1._mm256_set1_epi64x(0xFFFF_FFFF);
    let mut lo: [[__m256i; 2]; RB] = [[zero; 2]; RB];
    let mut hi: [[__m256i; 2]; RB] = [[zero; 2]; RB];
    let a_rows: [&[u32]; RB] = core::array::from_fn(|r| &g.a[(r0 + r) * k..(r0 + r + 1) * k]);
    let mut b_rows = g.b.get(j0..).unwrap_or(&[]).chunks(p).take(k);
    let mut kk = 0;
    while kk < k {
        let mut t: [[__m256i; 2]; RB] = [[zero; 2]; RB];
        if kk + 4 <= k {
            let xs: [[u32; 4]; RB] =
                core::array::from_fn(|r| a_rows[r][kk..kk + 4].try_into().unwrap());
            for s in 0..4 {
                let b = b_rows.next().unwrap_or(&[]);
                step_v3(v, &mut t, core::array::from_fn(|r| xs[r][s]), b);
            }
            kk += 4;
        } else {
            for q in kk..k {
                let b = b_rows.next().unwrap_or(&[]);
                step_v3(v, &mut t, core::array::from_fn(|r| a_rows[r][q]), b);
            }
            kk = k;
        }
        for r in 0..RB {
            for h in 0..2 {
                lo[r][h] = x2._mm256_add_epi64(lo[r][h], x2._mm256_and_si256(t[r][h], mask));
                hi[r][h] = x2._mm256_add_epi64(hi[r][h], x2._mm256_srli_epi64::<32>(t[r][h]));
            }
        }
    }
    for r in 0..RB {
        let k3 = Redc3::new(v, g.ch(r0 + r));
        for h in 0..2 {
            let w: [u64; 4] = cast(k3.redc_halves(lo[r][h], hi[r][h]));
            let o = (r0 + r) * p + j0 + 4 * h;
            for (dst, &x) in g.out[o..o + 4].iter_mut().zip(&w) {
                *dst = x as u32;
            }
        }
    }
}

/// One step of the AVX2 tile: `t[r] += x[r]·b[0..8]`.
#[inline(always)]
fn step_v3<const RB: usize>(v: V3, t: &mut [[__m256i; 2]; RB], x: [u32; RB], b: &[u32]) {
    let (x2, x1) = (v.avx2, v.avx);
    let b: &[u32; 8] = b[..8].try_into().unwrap();
    let b0: [u32; 4] = [b[0], b[1], b[2], b[3]];
    let b1: [u32; 4] = [b[4], b[5], b[6], b[7]];
    let b0 = x2._mm256_cvtepu32_epi64(cast(b0));
    let b1 = x2._mm256_cvtepu32_epi64(cast(b1));
    for (tr, &x) in t.iter_mut().zip(&x) {
        let x = x1._mm256_set1_epi64x(x as i64);
        tr[0] = x2._mm256_add_epi64(tr[0], x2._mm256_mul_epu32(x, b0));
        tr[1] = x2._mm256_add_epi64(tr[1], x2._mm256_mul_epu32(x, b1));
    }
}

/// Product scanning for a conversion to binary, AVX-512, `D ≤ 12` digits.
///
/// `rows` holds `R` rows of `stride` 32-bit multipliers each (the `ξ_i` of
/// every channel for a block of elements, then `α'`), zero beyond `b`
/// elements; `digits[r]` holds the `D` digits that row `r` multiplies. For
/// each group of eight elements the `2D` digit sums stay in registers while
/// the rows stream past, two rows per step (two products of a 31-bit and a
/// 32-bit number fit a `u64`). The carries are then propagated lane by lane and
/// the `D/2` limbs of each element written to `out`. Returns `false` when
/// AVX-512 is not available (or disabled for testing); the caller then uses
/// the portable kernel.
pub(crate) fn twos_digits<const D: usize>(
    rows: &[u32],
    stride: usize,
    digits: &[[u32; D]],
    b: usize,
    out: &mut [u64],
) -> bool {
    if LEVEL.load(core::sync::atomic::Ordering::Relaxed) != 0 || D > 12 {
        return false;
    }
    let Some(v4) = V4::try_new() else {
        return false;
    };
    v4.vectorize(TwosV4 {
        v: v4,
        rows,
        stride,
        digits,
        b,
        out,
    });
    true
}

struct TwosV4<'a, const D: usize> {
    v: V4,
    rows: &'a [u32],
    stride: usize,
    digits: &'a [[u32; D]],
    b: usize,
    out: &'a mut [u64],
}

impl<const D: usize> pulp::NullaryFnOnce for TwosV4<'_, D> {
    type Output = ();
    #[inline(always)]
    fn call(self) {
        let f = self.v.avx512f;
        let zero = f._mm512_setzero_si512();
        let mask = f._mm512_set1_epi64(0xFFFF_FFFF);
        let n_rows = self.digits.len();
        let w_out = D / 2;
        let load = |r: usize, k0: usize| {
            let x: [u32; 8] = self.rows[r * self.stride + k0..r * self.stride + k0 + 8]
                .try_into()
                .unwrap();
            f._mm512_cvtepu32_epi64(cast::<[u32; 8], __m256i>(x))
        };
        let mut k0 = 0;
        while k0 < self.b {
            let mut lo = [zero; D];
            let mut hi = [zero; D];
            let mut r = 0;
            while r < n_rows {
                let x = load(r, k0);
                let pair = r + 1 < n_rows;
                let z = if pair { load(r + 1, k0) } else { zero };
                for w in 0..D {
                    let dx = f._mm512_set1_epi64(i64::from(self.digits[r][w]));
                    let mut t = f._mm512_mul_epu32(x, dx);
                    if pair {
                        let dz = f._mm512_set1_epi64(i64::from(self.digits[r + 1][w]));
                        t = f._mm512_add_epi64(t, f._mm512_mul_epu32(z, dz));
                    }
                    lo[w] = f._mm512_add_epi64(lo[w], f._mm512_and_si512(t, mask));
                    hi[w] = f._mm512_add_epi64(hi[w], f._mm512_srli_epi64::<32>(t));
                }
                r += 2;
            }
            let lo: [[u64; 8]; D] = lo.map(cast);
            let hi: [[u64; 8]; D] = hi.map(cast);
            for e in 0..8.min(self.b - k0) {
                let o = &mut self.out[(k0 + e) * w_out..(k0 + e + 1) * w_out];
                let (mut c, mut p) = (0u64, 0u64);
                for (j, limb) in o.iter_mut().enumerate() {
                    let s0 = lo[2 * j][e] + p + c;
                    c = s0 >> 32;
                    p = hi[2 * j][e];
                    let s1 = lo[2 * j + 1][e] + p + c;
                    c = s1 >> 32;
                    p = hi[2 * j + 1][e];
                    *limb = (s0 & 0xFFFF_FFFF) | (s1 << 32);
                }
            }
            k0 += 8;
        }
    }
}

/// The constants of one channel, broadcast to every 64-bit lane (AVX-512).
struct Redc4 {
    v: V4,
    p: __m512i,
    ninv: __m512i,
    r2: __m512i,
    mask: __m512i,
}

impl Redc4 {
    #[inline(always)]
    fn new(v: V4, ch: Channel) -> Self {
        let f = v.avx512f;
        Redc4 {
            v,
            p: f._mm512_set1_epi64(i64::from(ch.modulus())),
            ninv: f._mm512_set1_epi64(i64::from(ch.neg_pinv())),
            r2: f._mm512_set1_epi64(i64::from(ch.r2())),
            mask: f._mm512_set1_epi64(0xFFFF_FFFF),
        }
    }

    /// `REDC(t)` lane by lane, for `t < p·2^32`: `m = lo32(t)·(−p⁻¹) mod 2^32`,
    /// `u = (t + m·p) / 2^32 < 2p`, then `min(u, u − p)` (wrapping).
    #[inline(always)]
    fn redc(&self, t: __m512i) -> __m512i {
        let f = self.v.avx512f;
        let m = f._mm512_and_si512(f._mm512_mul_epu32(t, self.ninv), self.mask);
        let u = f._mm512_srli_epi64::<32>(f._mm512_add_epi64(t, f._mm512_mul_epu32(m, self.p)));
        f._mm512_min_epu64(u, f._mm512_sub_epi64(u, self.p))
    }

    /// [`Channel::redc_halves`] lane by lane: `(hi mod p) + REDC(lo)`, with
    /// `hi mod p = REDC(REDC(hi)·R²)`.
    #[inline(always)]
    fn redc_halves(&self, lo: __m512i, hi: __m512i) -> __m512i {
        let f = self.v.avx512f;
        let h = self.redc(f._mm512_mul_epu32(self.redc(hi), self.r2));
        let s = f._mm512_add_epi64(self.redc(lo), h);
        f._mm512_min_epu64(s, f._mm512_sub_epi64(s, self.p))
    }
}

/// The same with AVX2, which has no unsigned 64-bit minimum: values here are
/// below `2^33`, so a signed comparison selects `u − p` when `u ≥ p`.
struct Redc3 {
    v: V3,
    p: __m256i,
    ninv: __m256i,
    r2: __m256i,
    mask: __m256i,
}

impl Redc3 {
    #[inline(always)]
    fn new(v: V3, ch: Channel) -> Self {
        let x1 = v.avx;
        Redc3 {
            v,
            p: x1._mm256_set1_epi64x(i64::from(ch.modulus())),
            ninv: x1._mm256_set1_epi64x(i64::from(ch.neg_pinv())),
            r2: x1._mm256_set1_epi64x(i64::from(ch.r2())),
            mask: x1._mm256_set1_epi64x(0xFFFF_FFFF),
        }
    }

    #[inline(always)]
    fn reduce_once(&self, u: __m256i) -> __m256i {
        let x2 = self.v.avx2;
        let keep = x2._mm256_cmpgt_epi64(self.p, u);
        x2._mm256_blendv_epi8(x2._mm256_sub_epi64(u, self.p), u, keep)
    }

    #[inline(always)]
    fn redc(&self, t: __m256i) -> __m256i {
        let x2 = self.v.avx2;
        let m = x2._mm256_and_si256(x2._mm256_mul_epu32(t, self.ninv), self.mask);
        let u = x2._mm256_srli_epi64::<32>(x2._mm256_add_epi64(t, x2._mm256_mul_epu32(m, self.p)));
        self.reduce_once(u)
    }

    #[inline(always)]
    fn redc_halves(&self, lo: __m256i, hi: __m256i) -> __m256i {
        let x2 = self.v.avx2;
        let h = self.redc(x2._mm256_mul_epu32(self.redc(hi), self.r2));
        self.reduce_once(x2._mm256_add_epi64(self.redc(lo), h))
    }
}
