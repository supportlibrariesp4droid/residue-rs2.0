//! [`RnsMod`]: an element of the ring `ℤ/Mℤ` in residue representation.
//!
//! This is the "raw" RNS type: arithmetic is modulo `M`, so it never overflows
//! and has no notion of sign or order (`ℤ/Mℤ` is not an ordered ring). Use it
//! for modular workloads (cryptography, hashing, number theory). For exact
//! signed integers with overflow checking, use [`Rns`](crate::Rns).
//!
//! Internally each residue is kept in Montgomery form, so multiplication costs
//! one Montgomery product per channel and no division.

use core::fmt;
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::str::FromStr;

use crate::base::{Base, RnsBase};
use crate::limbs;

/// An element of `ℤ/Mℤ`, where `M` is the product of the moduli of base `B`.
///
/// `N` is the number of channels and must match the base (the type system
/// enforces it). Values are `Copy` and occupy `4·N` bytes.
pub struct RnsMod<B: RnsBase<N>, const N: usize> {
    /// Residues in Montgomery form: `r[i] = (x · 2^32) mod m_i`.
    r: [u32; N],
    _base: PhantomData<B>,
}

impl<B: RnsBase<N>, const N: usize> RnsMod<B, N> {
    #[inline]
    fn base() -> &'static Base<N> {
        B::base()
    }

    #[inline]
    pub(crate) fn from_mont_residues(r: [u32; N]) -> Self {
        RnsMod {
            r,
            _base: PhantomData,
        }
    }

    /// Build from ordinary residues (`res[i]` is taken modulo `m_i`).
    #[must_use]
    pub fn from_residues(res: [u32; N]) -> Self {
        let ch = Self::base().channels();
        Self::from_mont_residues(core::array::from_fn(|i| {
            let p = ch[i].modulus();
            ch[i].to_mont(res[i] % p)
        }))
    }

    /// The ordinary residues `x mod m_i`.
    #[must_use]
    pub fn residues(&self) -> [u32; N] {
        let ch = Self::base().channels();
        core::array::from_fn(|i| ch[i].from_mont(self.r[i]))
    }

    /// The additive identity.
    #[inline]
    #[must_use]
    pub fn zero() -> Self {
        Self::from_mont_residues([0; N])
    }

    /// The multiplicative identity.
    #[must_use]
    pub fn one() -> Self {
        let ch = Self::base().channels();
        Self::from_mont_residues(core::array::from_fn(|i| ch[i].to_mont(1)))
    }

    /// `true` iff the value is `0` (mod `M`).
    #[inline]
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.r.iter().all(|&x| x == 0)
    }

    /// `x mod M`.
    #[must_use]
    pub fn from_u64(x: u64) -> Self {
        let ch = Self::base().channels();
        Self::from_mont_residues(core::array::from_fn(|i| {
            let p = ch[i].modulus();
            ch[i].to_mont((x % p as u64) as u32)
        }))
    }

    /// `x mod M`.
    #[must_use]
    pub fn from_u128(x: u128) -> Self {
        Self::from_limbs(&[x as u64, (x >> 64) as u64])
    }

    /// `x mod M`, so negative inputs map to `M - |x| mod M`.
    #[must_use]
    pub fn from_i128(x: i128) -> Self {
        let mag = Self::from_u128(x.unsigned_abs());
        if x < 0 {
            -mag
        } else {
            mag
        }
    }

    /// `x mod M` for an integer given as little-endian `u64` limbs of any length.
    #[must_use]
    pub fn from_limbs(limbs: &[u64]) -> Self {
        let base = Self::base();
        let sig = &limbs[..(limbs::bit_len_slice(limbs) as usize).div_ceil(64)];
        if sig.len() <= N {
            Self::from_mont_residues(core::array::from_fn(|i| base.mont_from_limbs(i, sig)))
        } else {
            let ch = base.channels();
            Self::from_mont_residues(core::array::from_fn(|i| {
                ch[i].to_mont(limbs::mod_small(sig, ch[i].modulus()))
            }))
        }
    }

    /// The canonical representative in `[0, M)` as little-endian `u64` limbs.
    ///
    /// Uses Garner's algorithm (mixed-radix conversion) followed by Horner
    /// evaluation: `O(N²)` word operations, no big-integer division.
    #[must_use]
    pub fn to_limbs(&self) -> [u64; N] {
        let base = Self::base();
        let ch = base.channels();
        let r = self.residues();

        let mut a = [0u32; N];
        for i in 0..N {
            let p = ch[i].modulus();
            let mut v = r[i];
            for (j, &aj) in a.iter().enumerate().take(i) {
                let (inv, inv_shoup) = base.garner(j, i);
                v = ch[i].sub(v, aj % p);
                v = ch[i].mul_shoup(v, inv, inv_shoup);
            }
            a[i] = v;
        }

        let mut x = [0u64; N];
        for i in (0..N).rev() {
            x = limbs::mul_small_add(x, ch[i].modulus(), a[i]);
        }
        x
    }

    /// The canonical representative in `[0, M)`, if it fits in a `u128`.
    #[must_use]
    pub fn to_u128(&self) -> Option<u128> {
        limbs::to_u128(&self.to_limbs())
    }

    /// `self^exp mod M`, square-and-multiply.
    #[must_use]
    pub fn pow(&self, mut exp: u64) -> Self {
        let mut acc = Self::one();
        let mut b = *self;
        while exp > 0 {
            if exp & 1 == 1 {
                acc *= b;
            }
            b *= b;
            exp >>= 1;
        }
        acc
    }

    /// Multiplicative inverse mod `M`, if it exists (i.e. `gcd(x, M) = 1`, which
    /// for a base of primes means no residue is zero). Uses Fermat's little
    /// theorem independently on each channel.
    #[must_use]
    pub fn inv(&self) -> Option<Self> {
        if self.r.contains(&0) {
            return None;
        }
        let ch = Self::base().channels();
        Some(Self::from_mont_residues(core::array::from_fn(|i| {
            ch[i].inv_mont(self.r[i])
        })))
    }

    /// Apply a binary per-channel operation.
    #[inline]
    fn zip(self, rhs: Self, op: impl Fn(&crate::Channel, u32, u32) -> u32) -> Self {
        let ch = Self::base().channels();
        Self::from_mont_residues(core::array::from_fn(|i| op(&ch[i], self.r[i], rhs.r[i])))
    }
}

impl<B: RnsBase<N>, const N: usize> Add for RnsMod<B, N> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        self.zip(rhs, |c, a, b| c.add(a, b))
    }
}

impl<B: RnsBase<N>, const N: usize> Sub for RnsMod<B, N> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self.zip(rhs, |c, a, b| c.sub(a, b))
    }
}

impl<B: RnsBase<N>, const N: usize> Mul for RnsMod<B, N> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.zip(rhs, |c, a, b| c.mul_mont(a, b))
    }
}

impl<B: RnsBase<N>, const N: usize> Neg for RnsMod<B, N> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        let ch = Self::base().channels();
        Self::from_mont_residues(core::array::from_fn(|i| ch[i].neg(self.r[i])))
    }
}

impl<B: RnsBase<N>, const N: usize> AddAssign for RnsMod<B, N> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<B: RnsBase<N>, const N: usize> SubAssign for RnsMod<B, N> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<B: RnsBase<N>, const N: usize> MulAssign for RnsMod<B, N> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<B: RnsBase<N>, const N: usize> Clone for RnsMod<B, N> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<B: RnsBase<N>, const N: usize> Copy for RnsMod<B, N> {}

impl<B: RnsBase<N>, const N: usize> PartialEq for RnsMod<B, N> {
    /// Montgomery form is a bijection, so comparing residues is exact.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.r == other.r
    }
}
impl<B: RnsBase<N>, const N: usize> Eq for RnsMod<B, N> {}

impl<B: RnsBase<N>, const N: usize> Hash for RnsMod<B, N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.r.hash(state);
    }
}

impl<B: RnsBase<N>, const N: usize> Default for RnsMod<B, N> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<B: RnsBase<N>, const N: usize> fmt::Display for RnsMod<B, N> {
    /// The canonical representative in `[0, M)`, in decimal.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        limbs::fmt_decimal(&self.to_limbs(), f)
    }
}

impl<B: RnsBase<N>, const N: usize> fmt::Debug for RnsMod<B, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RnsMod({} ; residues {:?})", self, self.residues())
    }
}

impl<B: RnsBase<N>, const N: usize> From<u64> for RnsMod<B, N> {
    fn from(x: u64) -> Self {
        Self::from_u64(x)
    }
}

impl<B: RnsBase<N>, const N: usize> From<i128> for RnsMod<B, N> {
    fn from(x: i128) -> Self {
        Self::from_i128(x)
    }
}

/// Error returned when parsing an [`RnsMod`] from a string fails.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParseRnsError;

impl fmt::Display for ParseRnsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("invalid decimal integer")
    }
}

impl core::error::Error for ParseRnsError {}

impl<B: RnsBase<N>, const N: usize> FromStr for RnsMod<B, N> {
    type Err = ParseRnsError;

    /// Parse a decimal integer (optionally with a leading `-`) and reduce it
    /// mod `M`. Digits are folded directly into the residues, so inputs of any
    /// length are accepted without building a big integer.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (neg, digits) = match s.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, s.strip_prefix('+').unwrap_or(s)),
        };
        if digits.is_empty() {
            return Err(ParseRnsError);
        }
        let ch = Self::base().channels();
        let mut res = [0u32; N];
        for byte in digits.bytes() {
            let d = match byte {
                b'0'..=b'9' => (byte - b'0') as u64,
                _ => return Err(ParseRnsError),
            };
            for i in 0..N {
                let p = ch[i].modulus() as u64;
                res[i] = ((res[i] as u64 * 10 + d) % p) as u32;
            }
        }
        let x = Self::from_residues(res);
        Ok(if neg { -x } else { x })
    }
}

#[cfg(feature = "num-bigint")]
mod bigint_interop {
    use super::*;
    use alloc::vec::Vec;
    use num_bigint::{BigInt, BigUint, Sign};

    impl<B: RnsBase<N>, const N: usize> From<&BigUint> for RnsMod<B, N> {
        fn from(x: &BigUint) -> Self {
            Self::from_limbs(&x.to_u64_digits())
        }
    }

    impl<B: RnsBase<N>, const N: usize> From<&BigInt> for RnsMod<B, N> {
        fn from(x: &BigInt) -> Self {
            let mag = Self::from_limbs(&x.magnitude().to_u64_digits());
            if x.sign() == Sign::Minus {
                -mag
            } else {
                mag
            }
        }
    }

    impl<B: RnsBase<N>, const N: usize> RnsMod<B, N> {
        /// The canonical representative in `[0, M)` as a [`BigUint`].
        #[must_use]
        pub fn to_biguint(&self) -> BigUint {
            let bytes: Vec<u8> = self
                .to_limbs()
                .iter()
                .flat_map(|l| l.to_le_bytes())
                .collect();
            BigUint::from_bytes_le(&bytes)
        }
    }
}
