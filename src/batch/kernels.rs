//! Per-channel slice kernels for the batch type.
//!
//! The kernels are plain loops over contiguous slices with a *single* modulus
//! per call, written so that LLVM auto-vectorizes them (no branches in the loop
//! body, no cross-iteration dependency except plain sums). With the `std`
//! feature, [`dispatch`] runs them inside a context compiled for AVX2 or AVX-512,
//! chosen at runtime by `pulp`; without it they are compiled for the build
//! target only. Either way this crate contains no `unsafe` code.

use crate::word::Channel;

/// A batch operation whose body should run with the best available SIMD ISA.
///
/// Implementations must mark `run` `#[inline(always)]` and call only
/// `#[inline(always)]` kernels, with no closures: `pulp` calls `run` from inside
/// a `#[target_feature(enable = "avx2"…)]` function, and only code that is
/// actually inlined there is compiled for that ISA. A closure is not inlined
/// across that boundary: with closures, the AVX path ran SSE2 code, which
/// showed up both in the timings and in the generated assembly.
pub(crate) trait Op {
    type Out;
    fn run(self) -> Self::Out;
}

/// Run `op` with the best SIMD instruction set available on this CPU.
#[inline(always)]
pub(crate) fn dispatch<T: Op>(op: T) -> T::Out {
    #[cfg(feature = "std")]
    {
        struct Run<T>(T);
        impl<T: Op> pulp::WithSimd for Run<T> {
            type Output = T::Out;
            #[inline(always)]
            fn with_simd<S: pulp::Simd>(self, _simd: S) -> T::Out {
                self.0.run()
            }
        }
        pulp::Arch::new().dispatch(Run(op))
    }
    #[cfg(not(feature = "std"))]
    {
        op.run()
    }
}

#[inline(always)]
pub(crate) fn add_assign(ch: Channel, a: &mut [u32], b: &[u32]) {
    for (x, &y) in a.iter_mut().zip(b) {
        *x = ch.add(*x, y);
    }
}

#[inline(always)]
pub(crate) fn sub_assign(ch: Channel, a: &mut [u32], b: &[u32]) {
    for (x, &y) in a.iter_mut().zip(b) {
        *x = ch.sub(*x, y);
    }
}

/// Montgomery product, element by element: 32×32→64 multiply, REDC, one
/// conditional subtraction. Maps to `vpmuludq` (AVX2) / `vpmuludq zmm` (AVX-512).
#[inline(always)]
pub(crate) fn mul_assign(ch: Channel, a: &mut [u32], b: &[u32]) {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    return super::wasm::mul_assign(ch, a, b);
    #[allow(unreachable_code)]
    for (x, &y) in a.iter_mut().zip(b) {
        *x = ch.mul_mont(*x, y);
    }
}

#[inline(always)]
pub(crate) fn scale_assign(ch: Channel, a: &mut [u32], k: u32) {
    for x in a.iter_mut() {
        *x = ch.mul_mont(*x, k);
    }
}

#[inline(always)]
pub(crate) fn neg_assign(ch: Channel, a: &mut [u32]) {
    for x in a.iter_mut() {
        *x = ch.neg(*x);
    }
}

/// `Σ a_k·b_k`, unreduced: see [`crate::word::dot_halves`]. The caller
/// reduces once at the end.
#[inline(always)]
pub(crate) fn dot(a: &[u32], b: &[u32]) -> (u64, u64) {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    return super::wasm::dot(a, b);
    #[allow(unreachable_code)]
    crate::word::dot_halves(a, b)
}

/// `Σ a_k`, unreduced (each term `< 2^31`, safe for `< 2^33` terms).
#[inline(always)]
pub(crate) fn sum(a: &[u32]) -> u64 {
    a.iter().map(|&x| x as u64).sum()
}

/// Montgomery residues of signed 128-bit inputs given as magnitude limbs
/// `(lo, hi)` and a negation mask (`neg[k] = 1` for negative values), using the
/// division-free power table `c = [fwd(i,0), fwd(i,1)]`.
///
/// The four 32-bit halves are multiplied by their table constants and the
/// products accumulated in two halves, then reduced once with
/// [`Channel::redc_halves`]: three reductions per element instead of four, and
/// no modular additions. Requires `p > 4` (four products per sum); use
/// [`from_two_limbs_small`] otherwise.
#[inline(always)]
pub(crate) fn from_two_limbs(
    ch: Channel,
    c: [[u32; 2]; 2],
    lo: &[u64],
    hi: &[u64],
    neg: &[u32],
    out: &mut [u32],
) {
    const MASK: u64 = 0xFFFF_FFFF;
    for (((o, &l), &h), &n) in out.iter_mut().zip(lo).zip(hi).zip(neg) {
        let t0 = (l & MASK) * c[0][0] as u64;
        let t1 = (l >> 32) * c[0][1] as u64;
        let t2 = (h & MASK) * c[1][0] as u64;
        let t3 = (h >> 32) * c[1][1] as u64;
        let sl = (t0 & MASK) + (t1 & MASK) + (t2 & MASK) + (t3 & MASK);
        let sh = (t0 >> 32) + (t1 >> 32) + (t2 >> 32) + (t3 >> 32);
        let v = ch.redc_halves(sl, sh);
        let nv = ch.neg(v);
        *o = if n != 0 { nv } else { v };
    }
}

/// [`from_two_limbs`] for moduli `p <= 4` (that is, 3), one reduction per product.
#[inline(always)]
pub(crate) fn from_two_limbs_small(
    ch: Channel,
    c: [[u32; 2]; 2],
    lo: &[u64],
    hi: &[u64],
    neg: &[u32],
    out: &mut [u32],
) {
    for (((o, &l), &h), &n) in out.iter_mut().zip(lo).zip(hi).zip(neg) {
        let a = ch.redc((l & 0xFFFF_FFFF) * c[0][0] as u64);
        let b = ch.redc((l >> 32) * c[0][1] as u64);
        let d = ch.redc((h & 0xFFFF_FFFF) * c[1][0] as u64);
        let e = ch.redc((h >> 32) * c[1][1] as u64);
        let v = ch.add(ch.add(a, b), ch.add(d, e));
        let nv = ch.neg(v);
        *o = if n != 0 { nv } else { v };
    }
}

/// One output row of a matrix product, unreduced: `lo[j], hi[j]` are the sums
/// of the low and high 32-bit halves of `a_row[k]·b[k][j]` over `k` (`b` is
/// row-major with `p` columns). The inner loop runs over `j`: a broadcast of
/// `a_row[k]` times a contiguous row of `b`, so it vectorizes across columns.
#[inline(always)]
pub(crate) fn gemm_row_acc(a_row: &[u32], b: &[u32], p: usize, lo: &mut [u64], hi: &mut [u64]) {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    return super::wasm::gemm_row_acc(a_row, b, p, lo, hi);
    #[allow(unreachable_code)]
    lo.fill(0);
    hi.fill(0);
    for (kk, &aik) in a_row.iter().enumerate() {
        let brow = &b[kk * p..(kk + 1) * p];
        let a = aik as u64;
        for ((l, h), &bkj) in lo.iter_mut().zip(hi.iter_mut()).zip(brow) {
            let t = a * bkj as u64;
            *l += t & 0xFFFF_FFFF;
            *h += t >> 32;
        }
    }
}

/// Reduce accumulated halves to Montgomery residues, without division.
///
/// The products are of Montgomery forms, `t = ã·b̃ = ab·R²`, and we want
/// `(Σab)·R = Σt·R⁻¹`. With `Σt = h·2^32 + l` and `R = 2^32`:
/// `Σt·R⁻¹ = h + l·R⁻¹ = (h mod p) + REDC(l)`, and `h mod p = to_mont(REDC(h))`.
/// Requires `l, h < p·2^32`, i.e. an inner dimension below `p`.
#[inline(always)]
pub(crate) fn gemm_row_reduce(ch: Channel, lo: &[u64], hi: &[u64], out: &mut [u32]) {
    for ((o, &l), &h) in out.iter_mut().zip(lo).zip(hi) {
        *o = ch.redc_halves(l, h);
    }
}

/// Same as [`gemm_row_reduce`] for any inner dimension (uses division).
#[inline(always)]
pub(crate) fn gemm_row_reduce_slow(ch: Channel, lo: &[u64], hi: &[u64], out: &mut [u32]) {
    let p = ch.modulus() as u64;
    let two32 = (1u64 << 32) % p;
    for ((o, &l), &h) in out.iter_mut().zip(lo).zip(hi) {
        let v = ((h % p) * two32 % p + l % p) % p;
        *o = ch.redc(v);
    }
}

/// Redundant channel of one output row: `out[j] = Σ_k a_row[k]·b[k][j] mod 2^64`.
#[inline(always)]
pub(crate) fn gemm_row_red(a_row: &[u64], b: &[u64], p: usize, out: &mut [u64]) {
    out.fill(0);
    for (kk, &aik) in a_row.iter().enumerate() {
        let brow = &b[kk * p..(kk + 1) * p];
        for (o, &bkj) in out.iter_mut().zip(brow) {
            *o = o.wrapping_add(aik.wrapping_mul(bkj));
        }
    }
}

/// `row[j] -= f · pivot_row[j]` in Montgomery form (Gaussian elimination step).
#[inline(always)]
pub(crate) fn axpy_neg(ch: Channel, f: u32, pivot_row: &[u32], row: &mut [u32]) {
    for (x, &y) in row.iter_mut().zip(pivot_row) {
        *x = ch.sub(*x, ch.mul_mont(f, y));
    }
}

/// Determinant of an `n×n` row-major matrix of Montgomery residues over the
/// field `ℤ/pℤ`, by Gaussian elimination with row swaps. Destroys `a`.
/// Each channel picks its own pivots: a zero mod one prime is not a zero mod
/// another, so channels never have to agree.
#[inline(always)]
pub(crate) fn det_channel(ch: Channel, n: usize, a: &mut [u32]) -> u32 {
    let mut det = ch.to_mont(1);
    for c in 0..n {
        let mut piv = n;
        for r in c..n {
            if a[r * n + c] != 0 {
                piv = r;
                break;
            }
        }
        if piv == n {
            return 0;
        }
        if piv != c {
            let (top, bottom) = a.split_at_mut(piv * n);
            top[c * n..(c + 1) * n].swap_with_slice(&mut bottom[..n]);
            det = ch.neg(det);
        }
        let pv = a[c * n + c];
        det = ch.mul_mont(det, pv);
        let inv = ch.inv_mont(pv);
        for r in c + 1..n {
            let f = ch.mul_mont(a[r * n + c], inv);
            if f != 0 {
                let (top, bottom) = a.split_at_mut(r * n);
                axpy_neg(
                    ch,
                    f,
                    &top[c * n + c + 1..(c + 1) * n],
                    &mut bottom[c + 1..n],
                );
            }
        }
    }
    det
}

#[inline(always)]
pub(crate) fn red_add_assign(a: &mut [u64], b: &[u64]) {
    for (x, &y) in a.iter_mut().zip(b) {
        *x = x.wrapping_add(y);
    }
}

#[inline(always)]
pub(crate) fn red_sub_assign(a: &mut [u64], b: &[u64]) {
    for (x, &y) in a.iter_mut().zip(b) {
        *x = x.wrapping_sub(y);
    }
}

#[inline(always)]
pub(crate) fn red_mul_assign(a: &mut [u64], b: &[u64]) {
    for (x, &y) in a.iter_mut().zip(b) {
        *x = x.wrapping_mul(y);
    }
}

#[inline(always)]
pub(crate) fn red_scale_assign(a: &mut [u64], k: u64) {
    for x in a.iter_mut() {
        *x = x.wrapping_mul(k);
    }
}

#[inline(always)]
pub(crate) fn red_neg_assign(a: &mut [u64]) {
    for x in a.iter_mut() {
        *x = x.wrapping_neg();
    }
}

#[inline(always)]
pub(crate) fn red_dot(a: &[u64], b: &[u64]) -> u64 {
    a.iter()
        .zip(b)
        .fold(0u64, |acc, (&x, &y)| acc.wrapping_add(x.wrapping_mul(y)))
}

#[inline(always)]
pub(crate) fn red_sum(a: &[u64]) -> u64 {
    a.iter().fold(0u64, |acc, &x| acc.wrapping_add(x))
}

/// `out[k] = REDC(a[k]·c)`. With Montgomery residues `a` and an ordinary
/// constant `c`, the result is `a·c` in ordinary form: this is how the CRT
/// coefficients `ξ = r·y mod p` are formed without leaving the kernel.
#[inline(always)]
pub(crate) fn mul_const_into(ch: Channel, a: &[u32], c: u32, out: &mut [u32]) {
    for (o, &x) in out.iter_mut().zip(a) {
        *o = ch.mul_mont(x, c);
    }
}

/// `acc[k] += x[k]·c` modulo `2^64`.
#[inline(always)]
pub(crate) fn red_mac(acc: &mut [u64], x: &[u32], c: u64) {
    for (a, &v) in acc.iter_mut().zip(x) {
        *a = a.wrapping_add((v as u64).wrapping_mul(c));
    }
}

/// One digit position of a product-scanning sum, for many elements at once:
/// `lo[k] += lo32(x[k]·d)` and `hi[k] += hi32(x[k]·d)`. Each piece is below
/// `2^32`, so the sums cannot overflow for fewer than `2^32` rows.
#[inline(always)]
pub(crate) fn digit_acc(lo: &mut [u64], hi: &mut [u64], x: &[u32], d: u32) {
    let d = d as u64;
    for ((l, h), &v) in lo.iter_mut().zip(hi.iter_mut()).zip(x) {
        let t = v as u64 * d;
        *l += t & 0xFFFF_FFFF;
        *h += t >> 32;
    }
}

/// Carry propagation through one digit position, for many elements at once.
/// The digit receives its own low halves, the high halves of the previous
/// position and the carry: `s = lo + prev + carry`. On return `lo` holds the
/// 32-bit digit, `carry` the new carry and `prev` this position's high halves.
#[inline(always)]
pub(crate) fn carry_digit(lo: &mut [u64], hi: &[u64], prev: &mut [u64], carry: &mut [u64]) {
    for (((l, &h), p), c) in lo
        .iter_mut()
        .zip(hi)
        .zip(prev.iter_mut())
        .zip(carry.iter_mut())
    {
        let s = *l + *p + *c;
        *l = s & 0xFFFF_FFFF;
        *c = s >> 32;
        *p = h;
    }
}

/// One limb of a conversion from binary, for many elements at once: the two
/// 32-bit halves of `mag[k]` times the channel's power constants `c`, summed
/// in two halves (see [`Channel::redc_halves`]).
#[inline(always)]
pub(crate) fn limb_acc(lo: &mut [u64], hi: &mut [u64], mag: &[u64], c: [u32; 2]) {
    const MASK: u64 = 0xFFFF_FFFF;
    let (c0, c1) = (c[0] as u64, c[1] as u64);
    for ((l, h), &m) in lo.iter_mut().zip(hi.iter_mut()).zip(mag) {
        let a = (m & MASK) * c0;
        let b = (m >> 32) * c1;
        *l += (a & MASK) + (b & MASK);
        *h += (a >> 32) + (b >> 32);
    }
}

/// Final step of a conversion from binary: reduce the two halves to a
/// Montgomery residue and negate where `neg[k] != 0`. Requires fewer than `p`
/// products in each sum (see [`Channel::redc_halves`]).
#[inline(always)]
pub(crate) fn limb_finish(ch: Channel, lo: &[u64], hi: &[u64], neg: &[u32], out: &mut [u32]) {
    for (((o, &l), &h), &n) in out.iter_mut().zip(lo).zip(hi).zip(neg) {
        let v = ch.redc_halves(l, h);
        let nv = ch.neg(v);
        *o = if n != 0 { nv } else { v };
    }
}

/// [`limb_finish`] for any number of products (bases with very small primes).
#[inline(always)]
pub(crate) fn limb_finish_any(ch: Channel, lo: &[u64], hi: &[u64], neg: &[u32], out: &mut [u32]) {
    for (((o, &l), &h), &n) in out.iter_mut().zip(lo).zip(hi).zip(neg) {
        let v = ch.reduce_halves(l, h);
        *o = if n != 0 { ch.neg(v) } else { v };
    }
}
