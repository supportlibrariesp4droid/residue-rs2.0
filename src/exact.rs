//! [`Rns`]: exact signed integers in residue representation.
//!
//! Unlike [`RnsMod`](crate::RnsMod), an `Rns` stands for a true integer `x`,
//! not a class modulo `M`. Three mechanisms make that possible.
//!
//! The first is a redundant channel holding `x mod 2^64`, maintained with
//! native wrapping `u64` arithmetic at almost no cost. It makes the CRT overflow
//! coefficient `α'` exactly computable (Shenoy and Kumaresan), which gives exact
//! signed reconstruction and base extension without the sequential Garner chain.
//!
//! The second is a certified bit bound. Every value carries `bits` with
//! `|x| < 2^bits`, updated on each operation, and operations whose bound would
//! exceed [`Base::max_bits`](crate::Base::max_bits) are rejected. The redundant
//! channel and the sign test therefore cannot be silently wrong.
//!
//! The third is a sign test in three tiers: `O(1)` when the bound is below 64
//! bits, `O(N)` for large values through a 64-bit fixed-point estimate of
//! `x/M`, and an `O(N²)` exact reconstruction for the values in between that the
//! estimate cannot decide. The answer is exact in every tier.
//!
//! ## The mathematics
//! With `M_i = M/m_i`, `y_i = M_i^{-1} mod m_i` and `ξ_i = r_i·y_i mod m_i`:
//! `S = Σ ξ_i·M_i ≡ x (mod M)` and `0 ≤ S < N·M`, so `S = x + α'·M` with an
//! integer `α' ∈ [0, N]`. Reducing mod `2^64` (where `M` is invertible):
//! `α' = (S mod 2^64 − x mod 2^64)·M^{-1} mod 2^64`, and since `α' ≤ N` this is
//! the exact value. Then `x = S − α'·M`, sign included.

use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::str::FromStr;

use crate::base::{Base, RnsBase};
use crate::limbs;
use crate::modular::RnsMod;

/// An exact signed integer `x` with `|x| < 2^B::max_bits`, represented by its
/// residues, a redundant residue mod `2^64`, and a certified bit bound.
///
/// Arithmetic operators panic on overflow (unlike primitive integers in
/// release mode, they never wrap: a wrapped value would break the invariants).
/// Use the `checked_*` methods to handle overflow explicitly.
pub struct Rns<B: RnsBase<N>, const N: usize> {
    /// Residues in Montgomery form.
    r: [u32; N],
    /// `x mod 2^64`.
    red: u64,
    /// Certified bound: `|x| < 2^bits`.
    bits: u32,
    _base: PhantomData<B>,
}

#[cold]
#[inline(never)]
fn overflow() -> ! {
    panic!("residue: integer overflow (result may exceed the base's range ±2^max_bits)")
}

#[inline]
fn add_bits(a: u32, b: u32) -> u32 {
    if a == 0 {
        b
    } else if b == 0 {
        a
    } else {
        a.max(b) + 1
    }
}

#[inline]
fn mul_bits(a: u32, b: u32) -> u32 {
    if a == 0 || b == 0 {
        0
    } else {
        a + b
    }
}

impl<B: RnsBase<N>, const N: usize> Rns<B, N> {
    #[inline]
    fn base() -> &'static Base<N> {
        B::base()
    }

    #[inline]
    fn raw(r: [u32; N], red: u64, bits: u32) -> Self {
        Rns {
            r,
            red,
            bits,
            _base: PhantomData,
        }
    }

    /// Internal: assemble from Montgomery residues, redundant residue and bound.
    /// The caller guarantees all three describe the same in-range integer.
    #[inline]
    pub(crate) fn from_parts_unchecked(r: [u32; N], red: u64, bits: u32) -> Self {
        Self::raw(r, red, bits)
    }

    /// Internal: `(Montgomery residues, redundant residue, bound)`.
    #[cfg(feature = "alloc")]
    #[inline]
    pub(crate) fn into_parts(self) -> ([u32; N], u64, u32) {
        (self.r, self.red, self.bits)
    }

    /// Internal: the redundant residue `x mod 2^64`.
    #[inline]
    pub(crate) fn redundant(&self) -> u64 {
        self.red
    }

    /// Values satisfy `|x| < 2^max_bits()`. See [`Base::max_bits`].
    #[inline]
    #[must_use]
    pub fn max_bits() -> u32 {
        Self::base().max_bits()
    }

    /// Zero.
    #[inline]
    #[must_use]
    pub fn zero() -> Self {
        Self::raw([0; N], 0, 0)
    }

    /// One.
    #[must_use]
    pub fn one() -> Self {
        let ch = Self::base().channels();
        Self::raw(core::array::from_fn(|i| ch[i].to_mont(1)), 1, 1)
    }

    /// Build from a sign and a magnitude given as little-endian `u64` limbs of
    /// any length. `None` if `|x| >= 2^max_bits`.
    #[must_use]
    pub fn from_sign_magnitude(negative: bool, magnitude: &[u64]) -> Option<Self> {
        let bits = limbs::bit_len_slice(magnitude);
        if bits > Self::max_bits() as u64 {
            return None;
        }
        let sig = &magnitude[..(bits as usize).div_ceil(64)];
        let base = Self::base();
        let r: [u32; N] = core::array::from_fn(|i| base.mont_from_limbs(i, sig));
        let red = magnitude.first().copied().unwrap_or(0);
        let x = Self::raw(r, red, bits as u32);
        Some(if negative { -x } else { x })
    }

    /// `None` if out of range.
    #[must_use]
    pub fn from_i128(x: i128) -> Option<Self> {
        let m = x.unsigned_abs();
        Self::from_sign_magnitude(x < 0, &[m as u64, (m >> 64) as u64])
    }

    /// `None` if out of range (only possible on bases with fewer than ~65 bits).
    #[must_use]
    pub fn from_i64(x: i64) -> Option<Self> {
        Self::from_i128(x as i128)
    }

    /// The certified bound: `|x| < 2^bits_bound()`. It can be larger than the
    /// true bit length (bounds only grow); see [`Rns::tighten`].
    #[inline]
    #[must_use]
    pub fn bits_bound(&self) -> u32 {
        self.bits
    }

    /// Replace the bound with the exact bit length of `|x|` (costs one
    /// reconstruction). Useful after long computations whose bound became
    /// pessimistic, e.g. after subtracting nearly equal numbers.
    #[must_use]
    pub fn tighten(self) -> Self {
        let (_, mag) = self.to_sign_magnitude();
        Self::raw(self.r, self.red, limbs::bit_len(&mag))
    }

    /// `ξ_i = r_i·y_i mod m_i` and the exact overflow coefficient `α'`.
    ///
    /// The residues are in Montgomery form `r_i·R`; a Montgomery product with
    /// the *ordinary* constant `y_i` gives `r_i·y_i` in ordinary form, so the
    /// `R^{-1}` factor is absorbed at no extra cost.
    pub(crate) fn crt_parts(&self) -> ([u32; N], u64) {
        let base = Self::base();
        let ch = base.channels();
        let y = base.crt_y();
        let xi: [u32; N] = core::array::from_fn(|i| ch[i].mul_mont(self.r[i], y[i]));
        let mi64 = base.crt_mi_mod64();
        let mut s64: u64 = 0;
        for i in 0..N {
            s64 = s64.wrapping_add((xi[i] as u64).wrapping_mul(mi64[i]));
        }
        let alpha = s64.wrapping_sub(self.red).wrapping_mul(base.m_inv64());
        debug_assert!(alpha <= N as u64, "invariant broken: alpha' out of range");
        (xi, alpha)
    }

    /// Exact two's-complement value `S − α'·M` over the first `L` limbs (higher
    /// limbs stay zero), where `L = ⌈(bits + 1)/64⌉` follows from the certified
    /// bound and never exceeds `limbs_used`. The value fits `64·L` bits with its
    /// sign, so arithmetic modulo `2^(64L)` is exact; limbs above `L` would only
    /// repeat the sign and are not computed.
    ///
    /// `S = Σ ξ_i·M_i` is built one limb position at a time (product
    /// scanning): position `j` sums `ξ_i·M_i[j]` over all channels in a `u128`,
    /// reading one contiguous column of the base's table, and a single carry
    /// runs through the result instead of one carry chain per channel.
    fn reconstruct(&self) -> ([u64; N], usize) {
        let base = Self::base();
        let l = (self.bits as usize + 1).div_ceil(64).min(base.limbs_used());
        let (xi, alpha) = self.crt_parts();
        let mut t = [0u64; N];
        let mut carry: u128 = 0;
        for (limb, col) in t[..l].iter_mut().zip(base.crt_cols()) {
            let mut s = carry;
            for (&x, &m) in xi.iter().zip(col) {
                s += x as u128 * m as u128;
            }
            *limb = s as u64;
            carry = s >> 64;
        }
        limbs::sub_mul_small(&mut t[..l], &base.modulus_limbs()[..l], alpha);
        (t, l)
    }

    /// `(is_negative, |x|)` with `|x|` as little-endian `u64` limbs.
    #[must_use]
    pub fn to_sign_magnitude(&self) -> (bool, [u64; N]) {
        if self.bits <= 63 {
            let v = self.red as i64;
            let mut mag = [0u64; N];
            mag[0] = v.unsigned_abs();
            return (v < 0, mag);
        }
        let (mut t, l) = self.reconstruct();
        let negative = t[l - 1] >> 63 == 1;
        if negative {
            limbs::neg_wrapping(&mut t[..l]);
        }
        (negative, t)
    }

    /// The value as `i128`, if it fits.
    #[must_use]
    pub fn to_i128(&self) -> Option<i128> {
        if self.bits <= 63 {
            return Some(self.red as i64 as i128);
        }
        let (t, l) = self.reconstruct();
        let ext = if (t[l.min(2) - 1] as i64) < 0 {
            u64::MAX
        } else {
            0
        };
        if t[2.min(l)..l].iter().any(|&x| x != ext) {
            return None;
        }
        let hi = if l >= 2 { t[1] } else { ext };
        Some((t[0] as u128 | (hi as u128) << 64) as i128)
    }

    /// `-1`, `0` or `1`. Always exact.
    ///
    /// Tier 1 (`O(1)`): if `|x| < 2^63` is certified, the redundant channel *is*
    /// `x`. Tier 2 (`O(N)`): `g = Σ ξ_i·⌊2^64/m_i⌋ − α'·2^64` satisfies
    /// `g ≤ 2^64·x/M ≤ g + Σ ξ_i`; if that interval excludes 0 the sign is
    /// decided. Tier 3 (`O(N²)`): exact reconstruction.
    #[inline]
    #[must_use]
    pub fn signum(&self) -> i32 {
        self.signum_tiered().0
    }

    /// Test hook: which tier (1, 2 or 3) decided the sign. Not part of the API.
    #[doc(hidden)]
    #[must_use]
    pub fn __sign_tier(&self) -> u8 {
        self.signum_tiered().1
    }

    fn signum_tiered(&self) -> (i32, u8) {
        if self.bits <= 63 {
            return ((self.red as i64).signum() as i32, 1);
        }
        if self.r.iter().all(|&v| v == 0) {
            return (0, 1);
        }
        let (xi, alpha) = self.crt_parts();
        let inv = Self::base().frac_inv64();
        let mut f: u128 = 0;
        let mut err: i128 = 0;
        for i in 0..N {
            f += xi[i] as u128 * inv[i] as u128;
            err += xi[i] as i128;
        }
        let g = f as i128 - ((alpha as i128) << 64);
        if g > 0 {
            return (1, 2);
        }
        if g + err < 0 {
            return (-1, 2);
        }
        let (t, l) = self.reconstruct();
        let s = if t[l - 1] >> 63 == 1 {
            -1
        } else if t.iter().all(|&l| l == 0) {
            0
        } else {
            1
        };
        (s, 3)
    }

    /// `true` iff `x < 0`.
    #[inline]
    #[must_use]
    pub fn is_negative(&self) -> bool {
        self.signum() < 0
    }

    /// `true` iff `x > 0`.
    #[inline]
    #[must_use]
    pub fn is_positive(&self) -> bool {
        self.signum() > 0
    }

    /// `true` iff `x == 0`. `O(N)`, no reconstruction.
    #[inline]
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.r.iter().all(|&v| v == 0)
    }

    /// `|x|`.
    #[must_use]
    pub fn abs(self) -> Self {
        if self.is_negative() {
            -self
        } else {
            self
        }
    }

    /// The ordinary residues `x mod m_i` (non-negative).
    #[must_use]
    pub fn residues(&self) -> [u32; N] {
        let ch = Self::base().channels();
        core::array::from_fn(|i| ch[i].from_mont(self.r[i]))
    }

    /// Forget the integer structure and view `x` as an element of `ℤ/Mℤ`.
    #[inline]
    #[must_use]
    pub fn to_rns_mod(&self) -> RnsMod<B, N> {
        RnsMod::from_mont_residues(self.r)
    }

    /// `self + rhs`, or `None` if the certified bound would exceed `max_bits`.
    #[must_use]
    pub fn checked_add(self, rhs: Self) -> Option<Self> {
        let bits = add_bits(self.bits, rhs.bits);
        if bits > Self::max_bits() {
            return None;
        }
        let ch = Self::base().channels();
        Some(Self::raw(
            core::array::from_fn(|i| ch[i].add(self.r[i], rhs.r[i])),
            self.red.wrapping_add(rhs.red),
            bits,
        ))
    }

    /// `self - rhs`, or `None` if the certified bound would exceed `max_bits`.
    #[must_use]
    pub fn checked_sub(self, rhs: Self) -> Option<Self> {
        let bits = add_bits(self.bits, rhs.bits);
        if bits > Self::max_bits() {
            return None;
        }
        Some(self.sub_with_bits(rhs, bits))
    }

    #[inline]
    fn sub_with_bits(self, rhs: Self, bits: u32) -> Self {
        let ch = Self::base().channels();
        Self::raw(
            core::array::from_fn(|i| ch[i].sub(self.r[i], rhs.r[i])),
            self.red.wrapping_sub(rhs.red),
            bits,
        )
    }

    /// `self * rhs`, or `None` if the certified bound would exceed `max_bits`.
    #[must_use]
    pub fn checked_mul(self, rhs: Self) -> Option<Self> {
        let bits = mul_bits(self.bits, rhs.bits);
        if bits > Self::max_bits() {
            return None;
        }
        let ch = Self::base().channels();
        Some(Self::raw(
            core::array::from_fn(|i| ch[i].mul_mont(self.r[i], rhs.r[i])),
            self.red.wrapping_mul(rhs.red),
            bits,
        ))
    }

    /// `self^exp`, or `None` on overflow.
    #[must_use]
    pub fn checked_pow(self, mut exp: u32) -> Option<Self> {
        let mut acc = Self::one();
        let mut b = self;
        while exp > 0 {
            if exp & 1 == 1 {
                acc = acc.checked_mul(b)?;
            }
            exp >>= 1;
            if exp > 0 {
                b = b.checked_mul(b)?;
            }
        }
        Some(acc)
    }
}

impl<B: RnsBase<N>, const N: usize> Rns<B, N> {
    /// The integer in `(−M/2, M/2)` with the given Montgomery residues, or
    /// `None` if it is not in range (`|x| >= 2^max_bits`).
    ///
    /// For values whose redundant channel is unknown (results computed purely
    /// mod `M`, such as a quotient candidate or a determinant): Garner's
    /// algorithm needs only the residues. `O(N²)`.
    pub(crate) fn from_mont_residues_signed(r: [u32; N]) -> Option<Self> {
        let x = RnsMod::<B, N>::from_mont_residues(r).to_limbs();
        let mut other = *Self::base().modulus_limbs();
        limbs::sub_assign(&mut other, &x);
        let (neg, mag) = match limbs::cmp(&x, &other) {
            Ordering::Less => (false, x),
            _ => (true, other),
        };
        Self::from_sign_magnitude(neg, &mag)
    }

    /// Truncated division with remainder, like Rust's `/` and `%` on integers:
    /// `x = q·y + r`, `q` rounded toward zero, `r` with the sign of `x` and
    /// `|r| < |y|`. `None` if `y == 0`. Never overflows (`|q| <= |x|`).
    ///
    /// Cost: `O(N²)`. The operands are reconstructed, divided in binary with
    /// Knuth's Algorithm D, and converted back.
    #[must_use]
    pub fn checked_div_rem(self, y: Self) -> Option<(Self, Self)> {
        if y.is_zero() {
            return None;
        }
        let (sx, mx) = self.to_sign_magnitude();
        let (sy, my) = y.to_sign_magnitude();
        let (q, r) = Self::divmod_magnitudes(&mx, &my);
        let q = Self::from_sign_magnitude(sx != sy, &q).expect("|q| <= |x| is in range");
        let r = Self::from_sign_magnitude(sx, &r).expect("|r| < |y| is in range");
        Some((q, r))
    }

    /// Unsigned `(|x| / |y|, |x| mod |y|)` on magnitudes of in-range values.
    fn divmod_magnitudes(mx: &[u64; N], my: &[u64; N]) -> ([u64; N], [u64; N]) {
        let mut q = [0u64; N];
        let mut r = [0u64; N];
        if Self::base().limbs_used() <= 2 {
            let a = limbs::to_u128(mx).expect("fits u128");
            let b = limbs::to_u128(my).expect("fits u128");
            let (qv, rv) = (a / b, a % b);
            q[0] = qv as u64;
            r[0] = rv as u64;
            if N > 1 {
                q[1] = (qv >> 64) as u64;
                r[1] = (rv >> 64) as u64;
            }
            return (q, r);
        }
        let (m, n) = (limbs::sig_len(mx), limbs::sig_len(my));
        debug_assert!(m < N);
        let mut un = [0u64; N];
        let mut vn = [0u64; N];
        limbs::divmod(
            &mx[..m],
            &my[..n],
            &mut q,
            &mut r,
            &mut un[..m + 1],
            &mut vn[..n],
        );
        (q, r)
    }

    /// `x / y` rounded toward zero, or `None` if `y == 0`. `O(N²)`.
    #[must_use]
    pub fn checked_div(self, y: Self) -> Option<Self> {
        self.checked_div_rem(y).map(|(q, _)| q)
    }

    /// `x % y` with the sign of `x`, or `None` if `y == 0`. `O(N²)`.
    #[must_use]
    pub fn checked_rem(self, y: Self) -> Option<Self> {
        self.checked_div_rem(y).map(|(_, r)| r)
    }

    /// Euclidean division: `x = q·y + r` with `0 <= r < |y|`. `None` if
    /// `y == 0`. Matches `i128::div_euclid` / `rem_euclid`. `O(N²)`.
    #[must_use]
    pub fn checked_div_rem_euclid(self, y: Self) -> Option<(Self, Self)> {
        if y.is_zero() {
            return None;
        }
        let (sx, mx) = self.to_sign_magnitude();
        let (sy, my) = y.to_sign_magnitude();
        let (mut q, mut r) = Self::divmod_magnitudes(&mx, &my);
        let r_zero = r.iter().all(|&l| l == 0);
        if sx && !r_zero {
            let mut ry = my;
            limbs::sub_assign(&mut ry, &r);
            r = ry;
            limbs::add_one(&mut q);
        }
        let q = Self::from_sign_magnitude(sx != sy, &q).expect("|q| + 1 <= |x| is in range");
        let r = Self::from_sign_magnitude(false, &r).expect("r < |y| is in range");
        Some((q, r))
    }

    /// Euclidean quotient. See [`Rns::checked_div_rem_euclid`].
    #[must_use]
    pub fn div_euclid(self, y: Self) -> Self {
        self.checked_div_rem_euclid(y)
            .unwrap_or_else(|| div_by_zero())
            .0
    }

    /// Euclidean remainder, always in `[0, |y|)`. See [`Rns::checked_div_rem_euclid`].
    #[must_use]
    pub fn rem_euclid(self, y: Self) -> Self {
        self.checked_div_rem_euclid(y)
            .unwrap_or_else(|| div_by_zero())
            .1
    }

    /// `Some(x / y)` if `y` divides `x` exactly, `None` otherwise (or if
    /// `y == 0`). The answer is exact, never probabilistic.
    ///
    /// When `y` is coprime to `M` (always, unless `y` is a multiple of one of
    /// the base's primes), the only candidate is `q ≡ x·y⁻¹ (mod M)`, computed
    /// channel by channel. Let `bq` and `by` be the exact bit lengths of `q`
    /// and `y`. If `y | x`, the true quotient is in range and has the same
    /// residues, so it is the candidate. If `bq + by ≥ bits(M)`, then
    /// `|q·y| ≥ 2^(bq+by−2) ≥ 2^max_bits > |x|`, so `q·y ≠ x`. Otherwise
    /// `|q·y| < M`, and `q·y ≡ x (mod M)` forces `q·y − x ∈ {−M, 0, M}`; since
    /// `M` is odd, the redundant channel (`mod 2^64`) tells `0` apart from `±M`.
    ///
    /// Cost: `O(N)` for the candidate and the check, plus one inverse per channel
    /// and one reconstruction (`O(N²)`) to get `q`'s exact size. If `y` shares a
    /// prime with `M`, falls back to [`Rns::checked_div_rem`].
    #[must_use]
    pub fn div_exact(self, y: Self) -> Option<Self> {
        if y.is_zero() {
            return None;
        }
        if self.is_zero() {
            return Some(Self::zero());
        }
        if y.r.contains(&0) {
            let (q, r) = self.checked_div_rem(y)?;
            return r.is_zero().then_some(q);
        }
        let base = Self::base();
        let ch = base.channels();
        let cand: [u32; N] =
            core::array::from_fn(|i| ch[i].mul_mont(self.r[i], ch[i].inv_mont(y.r[i])));

        let q = Self::from_mont_residues_signed(cand)?;
        let by = y.tighten().bits;
        if q.bits + by >= base.bits() {
            return None;
        }
        (q.red.wrapping_mul(y.red) == self.red).then_some(q)
    }
}

#[cold]
#[inline(never)]
fn div_by_zero() -> ! {
    panic!("residue: attempt to divide by zero")
}

impl<B: RnsBase<N>, const N: usize> core::ops::Div for Rns<B, N> {
    type Output = Self;
    /// Truncated division (`O(N²)`). Panics if `rhs == 0`.
    fn div(self, rhs: Self) -> Self {
        self.checked_div(rhs).unwrap_or_else(|| div_by_zero())
    }
}

impl<B: RnsBase<N>, const N: usize> core::ops::Rem for Rns<B, N> {
    type Output = Self;
    /// Remainder with the sign of `self` (`O(N²)`). Panics if `rhs == 0`.
    fn rem(self, rhs: Self) -> Self {
        self.checked_rem(rhs).unwrap_or_else(|| div_by_zero())
    }
}

impl<B: RnsBase<N>, const N: usize> Add for Rns<B, N> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        self.checked_add(rhs).unwrap_or_else(|| overflow())
    }
}

impl<B: RnsBase<N>, const N: usize> Sub for Rns<B, N> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self.checked_sub(rhs).unwrap_or_else(|| overflow())
    }
}

impl<B: RnsBase<N>, const N: usize> Mul for Rns<B, N> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.checked_mul(rhs).unwrap_or_else(|| overflow())
    }
}

impl<B: RnsBase<N>, const N: usize> Neg for Rns<B, N> {
    type Output = Self;
    /// Negation never overflows: the range is symmetric.
    #[inline]
    fn neg(self) -> Self {
        let ch = Self::base().channels();
        Self::raw(
            core::array::from_fn(|i| ch[i].neg(self.r[i])),
            self.red.wrapping_neg(),
            self.bits,
        )
    }
}

impl<B: RnsBase<N>, const N: usize> AddAssign for Rns<B, N> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<B: RnsBase<N>, const N: usize> SubAssign for Rns<B, N> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<B: RnsBase<N>, const N: usize> MulAssign for Rns<B, N> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<B: RnsBase<N>, const N: usize> Ord for Rns<B, N> {
    /// Exact comparison without leaving RNS.
    ///
    /// If the signs differ they decide. If they agree, `|a − b| < max(|a|, |b|)`,
    /// so `a − b` fits under the *existing* bound (no growth, never overflows)
    /// and its sign is the answer.
    fn cmp(&self, other: &Self) -> Ordering {
        let (sa, sb) = (self.signum(), other.signum());
        if sa != sb {
            return sa.cmp(&sb);
        }
        if sa == 0 {
            return Ordering::Equal;
        }
        let d = self.sub_with_bits(*other, self.bits.max(other.bits));
        d.signum().cmp(&0)
    }
}

impl<B: RnsBase<N>, const N: usize> PartialOrd for Rns<B, N> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<B: RnsBase<N>, const N: usize> PartialEq for Rns<B, N> {
    /// Values in range have unique residues, so this is exact and `O(N)`.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.r == other.r
    }
}
impl<B: RnsBase<N>, const N: usize> Eq for Rns<B, N> {}

impl<B: RnsBase<N>, const N: usize> Hash for Rns<B, N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.r.hash(state);
    }
}

impl<B: RnsBase<N>, const N: usize> Clone for Rns<B, N> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<B: RnsBase<N>, const N: usize> Copy for Rns<B, N> {}

impl<B: RnsBase<N>, const N: usize> Default for Rns<B, N> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<B: RnsBase<N>, const N: usize> fmt::Display for Rns<B, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (neg, mag) = self.to_sign_magnitude();
        if neg {
            f.write_str("-")?;
        }
        limbs::fmt_decimal(&mag, f)
    }
}

impl<B: RnsBase<N>, const N: usize> fmt::Debug for Rns<B, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Rns({} ; |x| < 2^{})", self, self.bits)
    }
}

impl<B: RnsBase<N>, const N: usize> TryFrom<i64> for Rns<B, N> {
    type Error = OutOfRange;
    fn try_from(x: i64) -> Result<Self, OutOfRange> {
        Self::from_i64(x).ok_or(OutOfRange)
    }
}

impl<B: RnsBase<N>, const N: usize> TryFrom<i128> for Rns<B, N> {
    type Error = OutOfRange;
    fn try_from(x: i128) -> Result<Self, OutOfRange> {
        Self::from_i128(x).ok_or(OutOfRange)
    }
}

/// The value does not fit the base's range `|x| < 2^max_bits`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutOfRange;

impl fmt::Display for OutOfRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("value out of the RNS base's range")
    }
}

impl core::error::Error for OutOfRange {}

/// Error returned when parsing an [`Rns`] from a string fails.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParseExactError {
    /// Empty input or a non-digit character.
    InvalidDigit,
    /// A valid integer that does not fit the base's range.
    OutOfRange,
}

impl fmt::Display for ParseExactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseExactError::InvalidDigit => f.write_str("invalid decimal integer"),
            ParseExactError::OutOfRange => f.write_str("integer out of the RNS base's range"),
        }
    }
}

impl core::error::Error for ParseExactError {}

impl<B: RnsBase<N>, const N: usize> FromStr for Rns<B, N> {
    type Err = ParseExactError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (neg, digits) = match s.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, s.strip_prefix('+').unwrap_or(s)),
        };
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return Err(ParseExactError::InvalidDigit);
        }
        let mut mag = [0u64; N];
        for chunk in digits.as_bytes().chunks(19) {
            let mut v = 0u64;
            for &b in chunk {
                v = v * 10 + (b - b'0') as u64;
            }
            let scale = 10u64.pow(chunk.len() as u32);
            mag = limbs::checked_mul_add_u64(&mag, scale, v).ok_or(ParseExactError::OutOfRange)?;
        }
        Self::from_sign_magnitude(neg, &mag).ok_or(ParseExactError::OutOfRange)
    }
}

#[cfg(feature = "num-bigint")]
mod bigint_interop {
    use super::*;
    use alloc::vec::Vec;
    use num_bigint::{BigInt, Sign};

    impl<B: RnsBase<N>, const N: usize> TryFrom<&BigInt> for Rns<B, N> {
        type Error = OutOfRange;
        fn try_from(x: &BigInt) -> Result<Self, OutOfRange> {
            Self::from_sign_magnitude(x.sign() == Sign::Minus, &x.magnitude().to_u64_digits())
                .ok_or(OutOfRange)
        }
    }

    impl<B: RnsBase<N>, const N: usize> Rns<B, N> {
        /// The exact value as a [`BigInt`].
        #[must_use]
        pub fn to_bigint(&self) -> BigInt {
            let (neg, mag) = self.to_sign_magnitude();
            let bytes: Vec<u8> = mag.iter().flat_map(|l| l.to_le_bytes()).collect();
            let sign = if neg { Sign::Minus } else { Sign::Plus };
            BigInt::from_bytes_le(sign, &bytes)
        }
    }
}
