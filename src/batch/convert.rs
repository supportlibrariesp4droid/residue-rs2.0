//! Conversions of whole [`RnsVec`]s between RNS and binary, vectorized across
//! elements.
//!
//! A scalar [`Rns`] converts one number at a time, and its loops run over the
//! channels of that number. Here every loop runs over many *elements* with the
//! same constants, which is the shape SIMD needs, and the same mathematics is
//! applied block by block.
//!
//! **To binary.** For each element, `x = S − α'·M` with `S = Σ ξ_i·M_i`,
//! `ξ_i = r_i·y_i mod m_i`, and `α'` exact thanks to the redundant channel (see
//! DESIGN.md). Modulo `2^(64w)`, subtracting `α'·M` is the same as adding
//! `α'·(2^(64w) − M)`, so `α'` simply becomes one more row of the sum. The sum
//! is built one 32-bit digit position at a time: for each row and each digit
//! `d`, every element accumulates `x·d` in two 64-bit halves (delayed
//! reduction, as in the dot product), and a final pass propagates the carries.
//! Only the `w` limbs the caller needs are computed: a value that fits `i128`
//! needs two limbs whatever the size of the base.
//!
//! For outputs of up to 8 limbs, one pass over a block handles every digit of
//! a pair of rows (two products of a 31-bit `ξ` and a 32-bit digit still fit a
//! `u64`), and `α'`, which is at most `N`, is computed modulo `2^32` so that no
//! 64-bit multiplication is needed. DESIGN.md records the versions that were
//! measured and discarded.
//!
//! **From binary.** Each element's magnitude limbs are multiplied by the
//! division-free power table of the base, summed in two halves and reduced
//! once per channel, then negated where the element is negative.

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use super::{kernels, RnsVec};
use crate::base::{Base, RnsBase};
use crate::exact::{OutOfRange, Rns};

/// Elements processed together. The digit accumulators of a block take
/// `2 · 2w · BLOCK` words, which stays in the L1 cache for the widths that
/// matter most.
const BLOCK: usize = 128;

/// Limbs that hold every value of bound `bits` in two's complement.
#[inline]
fn width_for_bits(bits: u32) -> usize {
    ((bits as usize) + 1).div_ceil(64).max(1)
}

/// Two's complement of every element modulo `2^(64·width)`, element `k` at
/// `out[k·width .. (k+1)·width]`.
struct ToTwosOp<'a, const N: usize> {
    base: &'a Base<N>,
    /// The residues of the elements to convert, one slice per channel.
    res: [&'a [u32]; N],
    red: &'a [u64],
    width: usize,
    out: &'a mut [u64],
}

#[allow(clippy::needless_range_loop)]
impl<const N: usize> kernels::Op for ToTwosOp<'_, N> {
    type Out = ();

    #[inline(always)]
    fn run(self) {
        let (base, width) = (self.base, self.width);
        let ch = base.channels();
        let (y, mi64, minv) = (base.crt_y(), base.crt_mi_mod64(), base.m_inv64());
        let digits = 2 * width;

        let cols = base.crt_cols();
        let digit = |limb: u64, w: usize| (limb >> (32 * (w % 2))) as u32;
        let mut dig = vec![0u32; digits * N];
        for w in 0..digits {
            if w / 2 < N {
                for i in 0..N {
                    dig[w * N + i] = digit(cols[w / 2][i], w);
                }
            }
        }
        let m = base.modulus_limbs();
        let mut neg_m = vec![0u64; width];
        let mut carry = 1u64;
        for (j, l) in neg_m.iter_mut().enumerate() {
            let (v, c) = (!m.get(j).copied().unwrap_or(0)).overflowing_add(carry);
            *l = v;
            carry = c as u64;
        }

        let n = self.red.len();
        let fused = FusedArgs {
            ch,
            y,
            mi64,
            minv,
            dig: &dig,
            neg_m: &neg_m,
            res: &self.res,
            red: self.red,
        };
        match digits {
            2 => return to_twos_fused::<2, N>(&fused, self.out),
            4 => return to_twos_fused::<4, N>(&fused, self.out),
            6 => return to_twos_fused::<6, N>(&fused, self.out),
            8 => return to_twos_fused::<8, N>(&fused, self.out),
            10 => return to_twos_fused::<10, N>(&fused, self.out),
            12 => return to_twos_fused::<12, N>(&fused, self.out),
            14 => return to_twos_fused::<14, N>(&fused, self.out),
            16 => return to_twos_fused::<16, N>(&fused, self.out),
            _ => {}
        }
        let mut xi = vec![0u32; BLOCK];
        let mut s64 = vec![0u64; BLOCK];
        let mut lo = vec![0u64; digits * BLOCK];
        let mut hi = vec![0u64; digits * BLOCK];
        let mut prev = vec![0u64; BLOCK];
        let mut carry = vec![0u64; BLOCK];

        let mut start = 0;
        while start < n {
            let b = BLOCK.min(n - start);
            let span = start..start + b;
            s64[..b].fill(0);
            lo.fill(0);
            hi.fill(0);

            for i in 0..N {
                kernels::mul_const_into(ch[i], &self.res[i][span.clone()], y[i], &mut xi[..b]);
                kernels::red_mac(&mut s64[..b], &xi[..b], mi64[i]);
                for w in 0..digits {
                    let d = dig[w * N + i];
                    if d != 0 {
                        let (l, h) = (
                            &mut lo[w * BLOCK..w * BLOCK + b],
                            &mut hi[w * BLOCK..w * BLOCK + b],
                        );
                        kernels::digit_acc(l, h, &xi[..b], d);
                    }
                }
            }

            for ((a, &s), &r) in xi[..b].iter_mut().zip(&s64[..b]).zip(&self.red[span]) {
                let alpha = s.wrapping_sub(r).wrapping_mul(minv);
                debug_assert!(alpha <= N as u64, "invariant broken: alpha' out of range");
                *a = alpha as u32;
            }
            for w in 0..digits {
                let d = digit(neg_m[w / 2], w);
                if d != 0 {
                    let (l, h) = (
                        &mut lo[w * BLOCK..w * BLOCK + b],
                        &mut hi[w * BLOCK..w * BLOCK + b],
                    );
                    kernels::digit_acc(l, h, &xi[..b], d);
                }
            }

            prev[..b].fill(0);
            carry[..b].fill(0);
            for w in 0..digits {
                let (l, h) = (
                    &mut lo[w * BLOCK..w * BLOCK + b],
                    &hi[w * BLOCK..w * BLOCK + b],
                );
                kernels::carry_digit(l, h, &mut prev[..b], &mut carry[..b]);
            }
            let out = &mut self.out[start * width..(start + b) * width];
            for (k, limbs) in out.chunks_exact_mut(width).enumerate() {
                for (j, limb) in limbs.iter_mut().enumerate() {
                    *limb = lo[2 * j * BLOCK + k] | (lo[(2 * j + 1) * BLOCK + k] << 32);
                }
            }
            start += b;
        }
    }
}

/// Inputs of [`to_twos_fused`].
struct FusedArgs<'a, const N: usize> {
    ch: &'a [crate::word::Channel; N],
    y: &'a [u32; N],
    mi64: &'a [u64; N],
    minv: u64,
    /// Digit `w` of `M_i` at `dig[w·N + i]`.
    dig: &'a [u32],
    neg_m: &'a [u64],
    res: &'a [&'a [u32]; N],
    red: &'a [u64],
}

/// The conversion to `D/2` limbs, with one pass over a block per row.
///
/// All `D` digit sums of a block live in one buffer, digit-major
/// (`acc[w·BLOCK + k]` low halves, `acc[(D + w)·BLOCK + k]` high halves). For
/// one row, a single loop over the elements updates every digit: the loop
/// over `w` has a constant trip count and is unrolled, and because all the
/// sums share one base pointer at constant offsets, the compiler knows they do
/// not overlap and vectorizes across elements without runtime alias checks.
///
/// `α'` is computed modulo `2^32` only. It is at most `N`, so its low 32 bits
/// are the whole value, and every multiplication of the redundant part becomes
/// a 32-bit one instead of a 64-bit one, which vector units lack or run slowly.
#[inline(always)]
#[allow(clippy::needless_range_loop)]
fn to_twos_fused<const D: usize, const N: usize>(a: &FusedArgs<'_, N>, out: &mut [u64]) {
    const MASK: u64 = 0xFFFF_FFFF;
    let w_out = D / 2;
    let n = a.red.len();
    let rows: Vec<[u32; D]> = (0..N)
        .map(|i| core::array::from_fn(|w| a.dig[w * N + i]))
        .collect();
    let alpha_row: [u32; D] = core::array::from_fn(|w| (a.neg_m[w / 2] >> (32 * (w % 2))) as u32);
    let mi32: [u32; N] = core::array::from_fn(|i| a.mi64[i] as u32);
    let minv32 = a.minv as u32;

    let mut xa = vec![0u32; BLOCK];
    let mut xb = vec![0u32; BLOCK];
    let mut s32 = vec![0u32; BLOCK];
    let mut acc = vec![0u64; 2 * D * BLOCK];
    let mut prev = vec![0u64; BLOCK];
    let mut carry = vec![0u64; BLOCK];

    let xi_into = |i: usize, start: usize, b: usize, x: &mut [u32], s32: &mut [u32]| {
        kernels::mul_const_into(a.ch[i], &a.res[i][start..start + b], a.y[i], &mut x[..b]);
        for (s, &v) in s32[..b].iter_mut().zip(&x[..b]) {
            *s = s.wrapping_add(v.wrapping_mul(mi32[i]));
        }
    };
    let alpha_into = |start: usize, b: usize, x: &mut [u32], s32: &[u32]| {
        for ((v, &s), &r) in x[..b]
            .iter_mut()
            .zip(&s32[..b])
            .zip(&a.red[start..start + b])
        {
            *v = s.wrapping_sub(r as u32).wrapping_mul(minv32);
            debug_assert!(*v as usize <= N, "invariant broken: alpha' out of range");
        }
    };

    #[cfg(all(feature = "std", target_arch = "x86_64"))]
    if D <= 12 {
        let mut all = vec![0u32; (N + 1) * BLOCK];
        let mut digits = rows.clone();
        digits.push(alpha_row);
        let mut start = 0;
        let mut explicit = true;
        while explicit && start < n {
            let b = BLOCK.min(n - start);
            s32[..b].fill(0);
            let (chans, alpha) = all.split_at_mut(N * BLOCK);
            for (i, row) in chans.chunks_exact_mut(BLOCK).enumerate() {
                xi_into(i, start, b, row, &mut s32);
            }
            alpha_into(start, b, alpha, &s32);
            let dst = &mut out[start * w_out..(start + b) * w_out];
            explicit = super::x86::twos_digits::<D>(&all, BLOCK, &digits, b, dst);
            start += b;
        }
        if explicit {
            return;
        }
    }

    let mut start = 0;
    while start < n {
        let b = BLOCK.min(n - start);
        acc.fill(0);
        s32[..b].fill(0);
        let mut i = 0;
        while i + 1 < N {
            xi_into(i, start, b, &mut xa, &mut s32);
            xi_into(i + 1, start, b, &mut xb, &mut s32);
            row_digits2::<D>(&mut acc, &xa[..b], &rows[i], &xb[..b], &rows[i + 1]);
            i += 2;
        }
        if i < N {
            xi_into(i, start, b, &mut xa, &mut s32);
            alpha_into(start, b, &mut xb, &s32);
            row_digits2::<D>(&mut acc, &xa[..b], &rows[i], &xb[..b], &alpha_row);
        } else {
            alpha_into(start, b, &mut xa, &s32);
            row_digits::<D>(&mut acc, &xa[..b], &alpha_row);
        }

        prev[..b].fill(0);
        carry[..b].fill(0);
        for w in 0..D {
            let (l, h) = acc.split_at_mut((D + w) * BLOCK);
            let lo = &mut l[w * BLOCK..w * BLOCK + b];
            kernels::carry_digit(lo, &h[..b], &mut prev[..b], &mut carry[..b]);
        }
        let dst = &mut out[start * w_out..(start + b) * w_out];
        for (k, limbs) in dst.chunks_exact_mut(w_out).enumerate() {
            for (j, limb) in limbs.iter_mut().enumerate() {
                *limb = acc[2 * j * BLOCK + k] | ((acc[(2 * j + 1) * BLOCK + k] & MASK) << 32);
            }
        }
        start += b;
    }
}

/// Two rows at once: `x[k]·d[w] + z[k]·e[w]` is below `2^64` (each product is
/// below `2^63`), so the pair is summed in full and split into halves once,
/// which halves the updates of the accumulators.
#[inline(always)]
fn row_digits2<const D: usize>(acc: &mut [u64], x: &[u32], d: &[u32; D], z: &[u32], e: &[u32; D]) {
    const MASK: u64 = 0xFFFF_FFFF;
    let acc = &mut acc[..2 * D * BLOCK];
    let len = x.len().min(z.len()).min(BLOCK);
    let (x, z) = (&x[..len], &z[..len]);
    for (k, (&u, &v)) in x.iter().zip(z).enumerate() {
        let (u, v) = (u64::from(u), u64::from(v));
        for w in 0..D {
            let t = u * u64::from(d[w]) + v * u64::from(e[w]);
            acc[w * BLOCK + k] += t & MASK;
            acc[(D + w) * BLOCK + k] += t >> 32;
        }
    }
}

/// One row of the product scanning for a block: for every element `k`,
/// every digit `w`, `x[k]·d[w]` split into the two halves of digit `w`.
#[inline(always)]
fn row_digits<const D: usize>(acc: &mut [u64], x: &[u32], d: &[u32; D]) {
    const MASK: u64 = 0xFFFF_FFFF;
    let acc = &mut acc[..2 * D * BLOCK];
    let x = &x[..x.len().min(BLOCK)];
    for (k, &v) in x.iter().enumerate() {
        let v = u64::from(v);
        for w in 0..D {
            let t = v * u64::from(d[w]);
            acc[w * BLOCK + k] += t & MASK;
            acc[(D + w) * BLOCK + k] += t >> 32;
        }
    }
}

/// Montgomery residues, in one channel, of signed integers given by magnitude
/// limbs. One operation per channel, so that channels can run on different
/// threads.
struct FromMagChannelOp<'a> {
    ch: crate::word::Channel,
    /// The channel's power constants, one pair per limb.
    fwd: &'a [[u32; 2]],
    /// `mag[j·n + k]`: limb `j` of the magnitude of element `k`.
    mag: &'a [u64],
    neg: &'a [u32],
    out: &'a mut [u32],
}

impl kernels::Op for FromMagChannelOp<'_> {
    type Out = ();

    #[inline(always)]
    fn run(self) {
        let n = self.neg.len();
        let limbs = self.fwd.len();
        let fast = 2 * limbs < self.ch.modulus() as usize;
        let mut lo = vec![0u64; BLOCK];
        let mut hi = vec![0u64; BLOCK];
        let mut start = 0;
        while start < n {
            let b = BLOCK.min(n - start);
            lo[..b].fill(0);
            hi[..b].fill(0);
            for (j, &c) in self.fwd.iter().enumerate() {
                let mag = &self.mag[j * n + start..j * n + start + b];
                kernels::limb_acc(&mut lo[..b], &mut hi[..b], mag, c);
            }
            let (neg, out) = (&self.neg[start..start + b], &mut self.out[start..start + b]);
            if fast {
                kernels::limb_finish(self.ch, &lo[..b], &hi[..b], neg, out);
            } else {
                kernels::limb_finish_any(self.ch, &lo[..b], &hi[..b], neg, out);
            }
            start += b;
        }
    }
}

impl<B: RnsBase<N>, const N: usize> RnsVec<B, N> {
    /// The number of 64-bit limbs that holds every element in two's
    /// complement, given the vector's certified bound: `⌈(bits + 1) / 64⌉`,
    /// and at least 1.
    #[must_use]
    pub fn twos_complement_width(&self) -> usize {
        width_for_bits(self.bits)
    }

    /// Every element as `width` little-endian 64-bit limbs in two's
    /// complement, element `k` at `out[k·width .. (k+1)·width]`.
    ///
    /// Vectorized across elements; only the limbs that are needed are
    /// computed. When the bound is at most 63 bits the redundant channel already
    /// holds every value and nothing is computed at all.
    ///
    /// # Errors
    /// [`OutOfRange`] if some element does not fit `width` limbs. This is only
    /// possible when `width` is below [`RnsVec::twos_complement_width`]; `out`
    /// is then left in an unspecified state.
    ///
    /// # Panics
    /// If `width == 0` or `out.len() != self.len() · width`.
    pub fn to_twos_complement(&self, width: usize, out: &mut [u64]) -> Result<(), OutOfRange> {
        assert!(
            width > 0,
            "residue: two's complement width must be at least 1"
        );
        assert_eq!(
            out.len(),
            self.len() * width,
            "residue: output length must be len · width"
        );
        if self.bits <= 63 {
            for (limbs, &r) in out.chunks_exact_mut(width).zip(&self.red) {
                limbs[0] = r;
                limbs[1..].fill(if (r as i64) < 0 { u64::MAX } else { 0 });
            }
            return Ok(());
        }
        let need = self.twos_complement_width();
        if width >= need {
            if width == need {
                self.twos_into(need, out);
            } else {
                let mut tmp = vec![0u64; self.len() * need];
                self.twos_into(need, &mut tmp);
                for (dst, src) in out.chunks_exact_mut(width).zip(tmp.chunks_exact(need)) {
                    dst[..need].copy_from_slice(src);
                    let ext = if (src[need - 1] as i64) < 0 {
                        u64::MAX
                    } else {
                        0
                    };
                    dst[need..].fill(ext);
                }
            }
            return Ok(());
        }
        let mut tmp = vec![0u64; self.len() * need];
        self.twos_into(need, &mut tmp);
        for (dst, src) in out.chunks_exact_mut(width).zip(tmp.chunks_exact(need)) {
            let ext = if (src[width - 1] as i64) < 0 {
                u64::MAX
            } else {
                0
            };
            if src[width..].iter().any(|&l| l != ext) {
                return Err(OutOfRange);
            }
            dst.copy_from_slice(&src[..width]);
        }
        Ok(())
    }

    fn twos_into(&self, width: usize, out: &mut [u64]) {
        let n = self.len();
        let par = super::par::enabled(n * (N + 1) * 2 * width);
        let chunk = if par {
            n.div_ceil(2 * super::par::threads())
                .next_multiple_of(BLOCK)
        } else {
            n.max(1)
        };
        let mut tasks: Vec<(usize, &mut [u64])> = out
            .chunks_mut(chunk * width)
            .enumerate()
            .map(|(t, o)| (t * chunk, o))
            .collect();
        let (base, res, red) = (Self::base(), &self.res, &self.red);
        super::par::for_each_mut(&mut tasks, par, |_, (start, o)| {
            let range = *start..*start + o.len() / width;
            kernels::dispatch(ToTwosOp {
                base,
                res: core::array::from_fn(|i| &res[i][range.clone()]),
                red: &red[range],
                width,
                out: o,
            });
        });
    }

    /// Every element as an `i128`.
    ///
    /// # Errors
    /// [`OutOfRange`] if some element does not fit.
    pub fn to_i128s(&self) -> Result<Vec<i128>, OutOfRange> {
        let mut limbs = vec![0u64; 2 * self.len()];
        self.to_twos_complement(2, &mut limbs)?;
        Ok(limbs
            .chunks_exact(2)
            .map(|l| (l[0] as u128 | (l[1] as u128) << 64) as i128)
            .collect())
    }

    /// Every element as an `i64`. Free when the bound is at most 63 bits.
    ///
    /// # Errors
    /// [`OutOfRange`] if some element does not fit.
    pub fn to_i64s(&self) -> Result<Vec<i64>, OutOfRange> {
        let mut limbs = vec![0u64; self.len()];
        self.to_twos_complement(1, &mut limbs)?;
        Ok(limbs.into_iter().map(|l| l as i64).collect())
    }

    /// Build from integers given as `width` little-endian 64-bit limbs in
    /// two's complement each, element `k` at `input[k·width .. (k+1)·width]`.
    ///
    /// # Errors
    /// [`OutOfRange`] if some value does not fit the base.
    ///
    /// # Panics
    /// If `width == 0` or `input.len()` is not a multiple of `width`.
    pub fn from_twos_complement(width: usize, input: &[u64]) -> Result<Self, OutOfRange> {
        assert!(
            width > 0,
            "residue: two's complement width must be at least 1"
        );
        assert_eq!(
            input.len() % width,
            0,
            "residue: input length must be a multiple of width"
        );
        let n = input.len() / width;
        let limbs = width.min(Self::base().limbs_used());
        let mut mag = vec![0u64; limbs * n];
        let mut neg = Vec::with_capacity(n);
        let mut red = Vec::with_capacity(n);
        let mut bits = 0u32;
        let max_bits = u64::from(Rns::<B, N>::max_bits());
        for (k, x) in input.chunks_exact(width).enumerate() {
            let negative = (x[width - 1] as i64) < 0;
            let flip = if negative { u64::MAX } else { 0 };
            let mut carry = u64::from(negative);
            let mut top = 0u64;
            for (j, &l) in x.iter().enumerate() {
                let (v, c) = (l ^ flip).overflowing_add(carry);
                carry = u64::from(c);
                if v != 0 {
                    top = 64 * j as u64 + 64 - u64::from(v.leading_zeros());
                }
                if j < limbs {
                    mag[j * n + k] = v;
                }
            }
            if top > max_bits {
                return Err(OutOfRange);
            }
            bits = bits.max(top as u32);
            neg.push(u32::from(negative));
            red.push(x[0]);
        }
        Ok(Self::from_magnitudes(&mag, limbs, &neg, red, bits))
    }

    /// Assemble from magnitudes stored limb-major (`mag[j·n + k]`), negation
    /// flags, redundant residues and the bound, all already validated.
    pub(super) fn from_magnitudes(
        mag: &[u64],
        limbs: usize,
        neg: &[u32],
        red: Vec<u64>,
        bits: u32,
    ) -> Self {
        let n = neg.len();
        let mut res: [Vec<u32>; N] = core::array::from_fn(|_| vec![0u32; n]);
        if limbs > 0 {
            let base = Self::base();
            let ch = base.channels();
            super::par::for_each_mut(
                &mut res,
                super::par::enabled(2 * N * n * limbs),
                |i, out| {
                    let fwd: Vec<[u32; 2]> = (0..limbs).map(|j| base.fwd(i, j)).collect();
                    kernels::dispatch(FromMagChannelOp {
                        ch: ch[i],
                        fwd: &fwd,
                        mag,
                        neg,
                        out,
                    });
                },
            );
        }
        RnsVec {
            res,
            red,
            bits,
            _base: PhantomData,
        }
    }
}

#[cfg(feature = "num-bigint")]
mod bigint {
    use alloc::vec;
    use alloc::vec::Vec;
    use num_bigint::{BigInt, Sign};

    use super::RnsVec;
    use crate::base::RnsBase;
    use crate::exact::{OutOfRange, Rns};

    impl<B: RnsBase<N>, const N: usize> RnsVec<B, N> {
        /// Every element as a `BigInt`, through one vectorized conversion.
        #[must_use]
        pub fn to_bigints(&self) -> Vec<BigInt> {
            let w = self.twos_complement_width();
            let mut limbs = vec![0u64; self.len() * w];
            self.to_twos_complement(w, &mut limbs)
                .expect("the bound's own width always fits");
            limbs
                .chunks_exact(w)
                .map(|l| {
                    let bytes: Vec<u8> = l.iter().flat_map(|x| x.to_le_bytes()).collect();
                    BigInt::from_signed_bytes_le(&bytes)
                })
                .collect()
        }

        /// Build from `BigInt`s through one vectorized conversion.
        ///
        /// # Errors
        /// [`OutOfRange`] if some value does not fit the base.
        pub fn from_bigints(xs: &[BigInt]) -> Result<Self, OutOfRange> {
            let n = xs.len();
            let limbs = Self::base().limbs_used();
            let mut mag = vec![0u64; limbs * n];
            let mut neg = Vec::with_capacity(n);
            let mut red = Vec::with_capacity(n);
            let mut bits = 0u32;
            for (k, x) in xs.iter().enumerate() {
                let len = x.bits();
                if len > u64::from(Rns::<B, N>::max_bits()) {
                    return Err(OutOfRange);
                }
                bits = bits.max(len as u32);
                let digits = x.magnitude().to_u64_digits();
                for (j, &d) in digits.iter().enumerate() {
                    mag[j * n + k] = d;
                }
                let negative = x.sign() == Sign::Minus;
                let low = digits.first().copied().unwrap_or(0);
                neg.push(u32::from(negative));
                red.push(if negative { low.wrapping_neg() } else { low });
            }
            Ok(Self::from_magnitudes(&mag, limbs, &neg, red, bits))
        }
    }
}
