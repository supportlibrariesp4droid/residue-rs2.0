//! Arithmetic modulo a large odd number with RNS Montgomery multiplication.
//!
//! The usual use of residue number systems in cryptography: numbers modulo an
//! odd `m` of hundreds or thousands of bits (an RSA modulus, a prime field),
//! multiplied without ever forming a big-integer product. The method is that
//! of Bajard, Didier and Kornerup and of Kawamura et al., with the exact
//! second base extension of Shenoy and Kumaresan that the rest of the crate
//! already uses.
//!
//! # Method
//!
//! Two bases with no prime in common, `B` with product `M` and `B'` with
//! product `M'`, and the redundant residue modulo `2^64`. Elements are kept in
//! Montgomery form `x̃ = x·M mod m`, in all channels at once. The product of
//! `a` and `b` is `a·b·M⁻¹ mod m`, computed as
//!
//! 1. `c = a·b` in every channel;
//! 2. `q = c·(−m⁻¹) mod M`, in `B`;
//! 3. `q̂ = Σ ξ_i·M_i` in `B'` and modulo `2^64`, where `ξ_i = q_i·M_i⁻¹ mod
//!    m_i`. This extension is *inexact*: `q̂ = q + α·M` for some `0 ≤ α < n`,
//!    but `q̂` is an exactly defined integer, so its residues, including the
//!    one modulo `2^64`, are exact;
//! 4. `r = (c + q̂·m)/M` in `B'` and modulo `2^64`. The division is exact,
//!    because `c + q̂·m ≡ c + q·m ≡ 0 (mod M)`, and `r ≡ a·b·M⁻¹ (mod m)`;
//! 5. `r` is extended from `B'` to `B` exactly, with the redundant residue.
//!
//! # Bounds
//!
//! Elements are not reduced to `[0, m)`; each carries a *factor* `f` with
//! value `< f·m`. If `a < f_a·m` and `b < f_b·m`, then, since `q̂ < n·M`,
//! `r < c/M + n·m < (f_a·f_b·m/M + n)·m`. With `M ≥ 2^s·m` the product's
//! factor is at most `⌊f_a·f_b/2^s⌋ + 1 + n`, so with enough room in `M` the
//! factor of a product stays near `n + 2` however many products are chained,
//! and additions and subtractions need no reduction at all (they add factors).
//!
//! The constructor computes the largest factor `F` such that the product of
//! any two elements of factor at most `F` still fits both bases, and every
//! element keeps a factor of at most `F`: when a sum would exceed it, its
//! operands are first *normalized* (multiplied by the Montgomery form of 1,
//! which brings the factor back to at most `n + 2`). No operation can fail. The
//! bases must leave room for `F ≥ 4(n + 2)`, which takes about
//! `bits(m) + 2·log2(n) + 5` bits in each base.
//!
//! **Not constant time.** Branches and table lookups depend on the data. Do
//! not use this module on secret values until a constant-time variant exists.

use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use core::marker::PhantomData;
use core::ops::{Add, Mul, Neg, Sub};

use crate::base::RnsBase;
use crate::exact::Rns;
use crate::extend::Extender;
use crate::limbs;

/// Why a [`MontRns`] context could not be built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MontRnsError {
    /// The modulus is even, or below 3.
    InvalidModulus,
    /// The modulus is divisible by this prime of the first base.
    SharesPrime(u32),
    /// The two bases have this prime in common.
    BasesOverlap(u32),
    /// A base is too small for this modulus: both bases need at least this
    /// many bits (`bits(M)` and `bits(M')`).
    BaseTooSmall {
        /// Bits each base needs.
        needed_bits: u32,
    },
}

impl fmt::Display for MontRnsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MontRnsError::InvalidModulus => f.write_str("modulus must be odd and at least 3"),
            MontRnsError::SharesPrime(p) => write!(f, "modulus is divisible by the base prime {p}"),
            MontRnsError::BasesOverlap(p) => write!(f, "the two bases share the prime {p}"),
            MontRnsError::BaseTooSmall { needed_bits } => {
                write!(
                    f,
                    "each base needs at least {needed_bits} bits for this modulus"
                )
            }
        }
    }
}

impl core::error::Error for MontRnsError {}

/// Residues of one element: first base, second base, modulo `2^64`.
#[derive(Clone, Copy)]
pub(crate) struct Parts<const N1: usize, const N2: usize> {
    pub(crate) r1: [u32; N1],
    pub(crate) r2: [u32; N2],
    pub(crate) red: u64,
}

/// A context for arithmetic modulo an odd `m`, with first base `B1` (`N1`
/// primes) and second base `B2` (`N2` primes). Build it once per modulus;
/// elements borrow it.
///
/// Both bases need about `bits(m) + 2·log2(N1) + 4` bits, and no prime in
/// common; the [`rns_base!`](crate::rns_base) macro declares a second base
/// disjoint from a first one with `largest N after K`. With 6 + 6 primes (185
/// bits) there is room for moduli up to 160 bits; the example works modulo the
/// prime `2^127 − 1` and checks Fermat's little theorem, `a^(m−1) = 1`:
///
/// ```
/// use residue::{rns_base, MontRns};
///
/// rns_base!(B1: largest 6);
/// rns_base!(B2: largest 6 after 6);
///
/// let m = [u64::MAX, u64::MAX >> 1];
/// let ctx = MontRns::<B1, 6, B2, 6>::new(&m).unwrap();
/// let a = ctx.from_u128(3);
/// let b = ctx.from_u128(5);
/// assert_eq!((a * b).to_u128(), Some(15));
/// assert_eq!(a.pow(&[u64::MAX - 1, u64::MAX >> 1]).to_u128(), Some(1));
/// ```
///
/// Elements are not reduced after every operation: each carries a bound
/// (its *factor*), and the context normalizes them when a bound would grow
/// too large, so no operation can fail. DESIGN.md has the method and the
/// proofs of the bounds.
///
/// **Not constant time.** Branches and table lookups depend on the data. Do
/// not use these types on secret values.
pub struct MontRns<B1: RnsBase<N1>, const N1: usize, B2: RnsBase<N2>, const N2: usize> {
    /// The modulus, trimmed little-endian limbs.
    pub(crate) m: Vec<u64>,
    pub(crate) m_bits: u32,
    /// Step 2 and the first half of step 3 in one product per channel:
    /// `w_i = −m⁻¹·M_i⁻¹ mod m_i` (ordinary form), so that `ξ_i = REDC(c̃_i·w_i)`.
    pub(crate) w: [u32; N1],
    /// Step 3: `(M_i mod m'_j)·R²` and, below, `M_i mod 2^64`.
    pub(crate) ext1: Extender<B1, N1, B2, N2>,
    /// Step 4: `m mod m'_j` and `M⁻¹ mod m'_j`, both in Montgomery form.
    pub(crate) m2: [u32; N2],
    pub(crate) minv2: [u32; N2],
    pub(crate) m64: u64,
    pub(crate) minv64: u64,
    /// Step 5.
    pub(crate) ext2: Extender<B2, N2, B1, N1>,
    /// `M ≥ 2^s·m`.
    pub(crate) s: u32,
    /// Largest factor an element may have: the product of two such elements
    /// fits both bases.
    pub(crate) fmax: u64,
    /// `M² mod m` as an element of factor 1 (enters Montgomery form).
    pub(crate) r2: Parts<N1, N2>,
    /// The integer 1 (leaves Montgomery form).
    pub(crate) one_plain: Parts<N1, N2>,
    /// `M mod m`: the Montgomery form of 1.
    pub(crate) one_mont: Parts<N1, N2>,
    /// `m` itself, as residues and modulo `2^64` (for subtraction).
    pub(crate) m_parts: Parts<N1, N2>,
    /// The bases as types only; `fn() -> _` keeps the context `Send` and
    /// `Sync` whatever the marker types are.
    pub(crate) _bases: PhantomData<fn() -> (B1, B2)>,
}

/// `x mod m` for limbs of any length (`m` trimmed, non-zero).
fn rem(x: &[u64], m: &[u64]) -> Vec<u64> {
    let (_, mut r, _) = limbs::__divmod_limbs(x, m);
    r.resize(m.len(), 0);
    r
}

/// Schoolbook product of two limb slices.
fn mul_limbs(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut out = vec![0u64; a.len() + b.len()];
    for (i, &x) in a.iter().enumerate() {
        let mut carry: u128 = 0;
        for (j, &y) in b.iter().enumerate() {
            let t = out[i + j] as u128 + x as u128 * y as u128 + carry;
            out[i + j] = t as u64;
            carry = t >> 64;
        }
        out[i + b.len()] = carry as u64;
    }
    out
}

impl<B1: RnsBase<N1>, const N1: usize, B2: RnsBase<N2>, const N2: usize> MontRns<B1, N1, B2, N2> {
    /// Build the context for the odd modulus `m`, given as little-endian
    /// limbs.
    ///
    /// # Errors
    /// See [`MontRnsError`]: an even or tiny modulus, a modulus divisible by a
    /// prime of `B1`, bases that share a prime, or bases too small for `m`.
    pub fn new(m: &[u64]) -> Result<Self, MontRnsError> {
        let m: Vec<u64> = m[..limbs::sig_len(m)].to_vec();
        if m.is_empty() || m[0] & 1 == 0 || (m.len() == 1 && m[0] < 3) {
            return Err(MontRnsError::InvalidModulus);
        }
        let (b1, b2) = (B1::base(), B2::base());
        let (ch1, ch2) = (b1.channels(), b2.channels());
        for c in ch1 {
            if ch2.iter().any(|d| d.modulus() == c.modulus()) {
                return Err(MontRnsError::BasesOverlap(c.modulus()));
            }
        }
        let m_bits = limbs::bit_len_slice(&m) as u32;
        let fbits = 64 - (N1 as u64 + 2).leading_zeros();
        let needed_bits = m_bits + 2 * fbits + 5;
        if b1.bits() < needed_bits || b2.bits() < needed_bits {
            return Err(MontRnsError::BaseTooSmall { needed_bits });
        }
        let s = b1.bits() - 1 - m_bits;
        let room = b1.max_bits().min(b2.max_bits()) - m_bits;
        let t = ((room - 2 + s) / 2).min(62);
        let fmax = 1u64 << t;
        debug_assert!(fmax >= 4 * (N1 as u64 + 2));

        let mut w = [0u32; N1];
        for (i, c) in ch1.iter().enumerate() {
            let mi = limbs::mod_small(&m, c.modulus());
            if mi == 0 {
                return Err(MontRnsError::SharesPrime(c.modulus()));
            }
            let minv = c.from_mont(c.inv_mont(c.to_mont(mi)));
            let y = b1.crt_y()[i];
            let prod = c.from_mont(c.mul_mont(c.to_mont(minv), c.to_mont(y)));
            w[i] = c.neg(prod);
        }

        let m_mod1 = |i: usize| limbs::mod_small(&m, ch1[i].modulus());
        let m2 = core::array::from_fn(|j| ch2[j].to_mont(limbs::mod_small(&m, ch2[j].modulus())));
        let minv2 = core::array::from_fn(|j| {
            let c = ch2[j];
            c.inv_mont(c.to_mont(limbs::mod_small(b1.modulus_limbs(), c.modulus())))
        });
        let m64 = m[0];
        let minv64 = b1.m_inv64();

        let parts_of = |x: &[u64]| -> Parts<N1, N2> {
            let x = &x[..limbs::sig_len(x)];
            Parts {
                r1: core::array::from_fn(|i| b1.mont_from_limbs(i, x)),
                r2: core::array::from_fn(|j| b2.mont_from_limbs(j, x)),
                red: x.first().copied().unwrap_or(0),
            }
        };
        let mm = rem(b1.modulus_limbs(), &m);
        let r2 = parts_of(&rem(&mul_limbs(&mm, &mm), &m));
        let one_mont = parts_of(&mm);
        let one_plain = parts_of(&[1]);
        let m_parts = Parts {
            r1: core::array::from_fn(|i| ch1[i].to_mont(m_mod1(i))),
            r2: core::array::from_fn(|j| ch2[j].to_mont(limbs::mod_small(&m, ch2[j].modulus()))),
            red: m64,
        };

        Ok(MontRns {
            m,
            m_bits,
            w,
            ext1: Extender::new(),
            m2,
            minv2,
            m64,
            minv64,
            ext2: Extender::new(),
            s,
            fmax,
            r2,
            one_plain,
            one_mont,
            m_parts,
            _bases: PhantomData,
        })
    }

    /// The modulus, as trimmed little-endian limbs.
    #[must_use]
    pub fn modulus(&self) -> &[u64] {
        &self.m
    }

    /// Factor of a product of elements of factors `fa, fb ≤ F`:
    /// `r < (fa·fb·m/M + n)·m ≤ (⌊fa·fb/2^s⌋ + 1 + n)·m`.
    pub(crate) fn product_factor(&self, fa: u64, fb: u64) -> u64 {
        debug_assert!(fa <= self.fmax && fb <= self.fmax);
        let q = (fa as u128 * fb as u128).checked_shr(self.s).unwrap_or(0);
        q as u64 + 1 + N1 as u64
    }

    /// The five steps of the module documentation, on residues.
    pub(crate) fn mul_parts(
        &self,
        a: &Parts<N1, N2>,
        b: &Parts<N1, N2>,
        f_out: u64,
    ) -> Parts<N1, N2> {
        let (b1, b2) = (B1::base(), B2::base());
        let (ch1, ch2) = (b1.channels(), b2.channels());
        let xi: [u32; N1] =
            core::array::from_fn(|i| ch1[i].mul_mont(ch1[i].mul_mont(a.r1[i], b.r1[i]), self.w[i]));
        let c2: [u32; N2] = core::array::from_fn(|j| ch2[j].mul_mont(a.r2[j], b.r2[j]));
        let c_red = a.red.wrapping_mul(b.red);
        let q2: [u32; N2] = core::array::from_fn(|j| {
            let (lo, hi) = crate::word::dot_halves(&xi, &self.ext1.cols[j]);
            ch2[j].reduce_halves(lo, hi)
        });
        let mi64 = b1.crt_mi_mod64();
        let q_red = xi.iter().zip(mi64).fold(0u64, |s, (&x, &mi)| {
            s.wrapping_add((x as u64).wrapping_mul(mi))
        });
        let r2: [u32; N2] = core::array::from_fn(|j| {
            let c = ch2[j];
            c.mul_mont(c.add(c2[j], c.mul_mont(q2[j], self.m2[j])), self.minv2[j])
        });
        let red = c_red
            .wrapping_add(q_red.wrapping_mul(self.m64))
            .wrapping_mul(self.minv64);
        let bits = (64 - f_out.leading_zeros()) + self.m_bits;
        let r = Rns::<B2, N2>::from_parts_unchecked(r2, red, bits);
        #[cfg(debug_assertions)]
        {
            let (neg, mag) = r.to_sign_magnitude();
            let limit = mul_limbs(&[f_out], &self.m);
            assert!(
                !neg && limbs::cmp(&mag, &limit) == core::cmp::Ordering::Less,
                "invariant broken: product above its factor bound"
            );
        }
        let (r1, _, _) = self
            .ext2
            .extend(&r)
            .expect("factors at most F always fit")
            .into_parts();
        Parts { r1, r2, red }
    }

    pub(crate) fn elem(&self, p: Parts<N1, N2>, factor: u64) -> ModElem<'_, B1, N1, B2, N2> {
        ModElem {
            ctx: self,
            p,
            factor,
        }
    }

    /// The element `x mod m`, from little-endian limbs of any length.
    #[must_use]
    pub fn from_limbs(&self, x: &[u64]) -> ModElem<'_, B1, N1, B2, N2> {
        let (b1, b2) = (B1::base(), B2::base());
        let x = rem(&x[..limbs::sig_len(x)], &self.m);
        let x = &x[..limbs::sig_len(&x)];
        let p = Parts {
            r1: core::array::from_fn(|i| b1.mont_from_limbs(i, x)),
            r2: core::array::from_fn(|j| b2.mont_from_limbs(j, x)),
            red: x.first().copied().unwrap_or(0),
        };
        let f = self.product_factor(1, 1);
        self.elem(self.mul_parts(&p, &self.r2, f), f)
    }

    /// The element `x mod m`.
    #[must_use]
    pub fn from_u128(&self, x: u128) -> ModElem<'_, B1, N1, B2, N2> {
        self.from_limbs(&[x as u64, (x >> 64) as u64])
    }

    /// Zero.
    #[must_use]
    pub fn zero(&self) -> ModElem<'_, B1, N1, B2, N2> {
        self.elem(
            Parts {
                r1: [0; N1],
                r2: [0; N2],
                red: 0,
            },
            1,
        )
    }

    /// One.
    #[must_use]
    pub fn one(&self) -> ModElem<'_, B1, N1, B2, N2> {
        self.elem(self.one_mont, 1)
    }
}

/// An element of `ℤ/mℤ` in a [`MontRns`] context, in Montgomery form, not
/// necessarily reduced (see [`MontRns`] and "Bounds" in DESIGN.md).
///
/// Operators never fail, except that they panic if the operands belong to
/// different contexts.
#[derive(Clone, Copy)]
pub struct ModElem<'c, B1: RnsBase<N1>, const N1: usize, B2: RnsBase<N2>, const N2: usize> {
    pub(crate) ctx: &'c MontRns<B1, N1, B2, N2>,
    pub(crate) p: Parts<N1, N2>,
    /// The value is below `factor·m`.
    pub(crate) factor: u64,
}

impl<'c, B1: RnsBase<N1>, const N1: usize, B2: RnsBase<N2>, const N2: usize>
    ModElem<'c, B1, N1, B2, N2>
{
    fn same_ctx(&self, other: &Self) {
        assert!(
            core::ptr::eq(self.ctx, other.ctx),
            "residue: elements of different MontRns contexts"
        );
    }

    fn product(&self, other: &Self) -> Self {
        self.same_ctx(other);
        let f = self.ctx.product_factor(self.factor, other.factor);
        self.ctx.elem(self.ctx.mul_parts(&self.p, &other.p, f), f)
    }

    /// Sum of residues, assuming the factors already add up to at most `F`.
    fn sum(&self, other: &Self) -> Self {
        let (b1, b2) = (B1::base(), B2::base());
        let (ch1, ch2) = (b1.channels(), b2.channels());
        let p = Parts {
            r1: core::array::from_fn(|i| ch1[i].add(self.p.r1[i], other.p.r1[i])),
            r2: core::array::from_fn(|j| ch2[j].add(self.p.r2[j], other.p.r2[j])),
            red: self.p.red.wrapping_add(other.p.red),
        };
        self.ctx.elem(p, self.factor + other.factor)
    }

    /// `self + other`, normalizing the operands first if their factors would
    /// add up to more than `F`.
    fn plus(&self, other: &Self) -> Self {
        self.same_ctx(other);
        if self.factor + other.factor <= self.ctx.fmax {
            self.sum(other)
        } else {
            self.normalize().sum(&other.normalize())
        }
    }

    /// `factor·m − self`: the same class negated, still non-negative.
    fn negated(&self) -> Parts<N1, N2> {
        let (b1, b2) = (B1::base(), B2::base());
        let (ch1, ch2) = (b1.channels(), b2.channels());
        let mp = &self.ctx.m_parts;
        let f = self.factor;
        Parts {
            r1: core::array::from_fn(|i| {
                let c = ch1[i];
                let fm = c.mul_mont(mp.r1[i], c.to_mont((f % c.modulus() as u64) as u32));
                c.sub(fm, self.p.r1[i])
            }),
            r2: core::array::from_fn(|j| {
                let c = ch2[j];
                let fm = c.mul_mont(mp.r2[j], c.to_mont((f % c.modulus() as u64) as u32));
                c.sub(fm, self.p.r2[j])
            }),
            red: f.wrapping_mul(mp.red).wrapping_sub(self.p.red),
        }
    }

    /// `self²`.
    #[must_use]
    pub fn square(&self) -> Self {
        *self * *self
    }

    /// `self^e`, the exponent as little-endian limbs (`0^0 = 1`), with a
    /// window of 4 bits: 14 products to build a table, then for every 4 bits
    /// of the exponent 4 squarings and one product.
    #[must_use]
    pub fn pow(&self, e: &[u64]) -> Self {
        let e = &e[..limbs::sig_len(e)];
        let one = self.ctx.one();
        if e.is_empty() {
            return one;
        }
        let mut table = [one; 16];
        table[1] = *self;
        for k in 2..16 {
            table[k] = table[k - 1] * *self;
        }
        let nibbles = e.len() * 16;
        let nib = |i: usize| ((e[i / 16] >> (4 * (i % 16))) & 15) as usize;
        let mut top = nibbles;
        while top > 0 && nib(top - 1) == 0 {
            top -= 1;
        }
        let mut acc = table[nib(top - 1)];
        for i in (0..top - 1).rev() {
            for _ in 0..4 {
                acc = acc.square();
            }
            acc = acc * table[nib(i)];
        }
        acc
    }

    /// `self⁻¹`, assuming `m` is prime (Fermat: `self^(m−2)`). For a composite
    /// modulus the result is meaningless. The inverse of zero is zero.
    #[must_use]
    pub fn inv_prime(&self) -> Self {
        let mut e = self.ctx.m.clone();
        let (v, b) = e[0].overflowing_sub(2);
        e[0] = v;
        if b {
            for l in e.iter_mut().skip(1) {
                let (v, b) = l.overflowing_sub(1);
                *l = v;
                if !b {
                    break;
                }
            }
        }
        self.pow(&e)
    }

    /// The same element with its factor reset to at most `N1 + 2`, at the cost
    /// of one product. Sums call it when needed; it is public for callers who
    /// want small factors before handing elements on.
    #[must_use]
    pub fn normalize(&self) -> Self {
        self.product(&self.ctx.one())
    }

    /// The canonical representative in `[0, m)`, as little-endian limbs
    /// (as many as the modulus has).
    #[must_use]
    pub fn to_limbs(&self) -> Vec<u64> {
        let ctx = self.ctx;
        let f = ctx.product_factor(self.factor, 1);
        let x = ctx.mul_parts(&self.p, &ctx.one_plain, f);
        let bits = (64 - f.leading_zeros()) + ctx.m_bits;
        let (_, mag) = Rns::<B2, N2>::from_parts_unchecked(x.r2, x.red, bits).to_sign_magnitude();
        rem(&mag, &ctx.m)
    }

    /// The canonical representative, if it fits a `u128`.
    #[must_use]
    pub fn to_u128(&self) -> Option<u128> {
        let l = self.to_limbs();
        if l.iter().skip(2).any(|&x| x != 0) {
            return None;
        }
        Some(l[0] as u128 | (l.get(1).copied().unwrap_or(0) as u128) << 64)
    }

    /// `true` if the element is `0 mod m`.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.to_limbs().iter().all(|&l| l == 0)
    }

    /// The factor bounding the stored value (`< factor·m`); for tests and
    /// diagnostics.
    #[must_use]
    pub fn factor(&self) -> u64 {
        self.factor
    }
}

impl<B1: RnsBase<N1>, const N1: usize, B2: RnsBase<N2>, const N2: usize> Mul
    for ModElem<'_, B1, N1, B2, N2>
{
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        self.product(&rhs)
    }
}

impl<B1: RnsBase<N1>, const N1: usize, B2: RnsBase<N2>, const N2: usize> Add
    for ModElem<'_, B1, N1, B2, N2>
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        self.plus(&rhs)
    }
}

impl<B1: RnsBase<N1>, const N1: usize, B2: RnsBase<N2>, const N2: usize> Sub
    for ModElem<'_, B1, N1, B2, N2>
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self.same_ctx(&rhs);
        self.plus(&self.ctx.elem(rhs.negated(), rhs.factor))
    }
}

impl<B1: RnsBase<N1>, const N1: usize, B2: RnsBase<N2>, const N2: usize> Neg
    for ModElem<'_, B1, N1, B2, N2>
{
    type Output = Self;
    fn neg(self) -> Self {
        self.ctx.elem(self.negated(), self.factor)
    }
}

impl<B1: RnsBase<N1>, const N1: usize, B2: RnsBase<N2>, const N2: usize> PartialEq
    for ModElem<'_, B1, N1, B2, N2>
{
    /// Equality of the classes modulo `m` (costs two conversions).
    fn eq(&self, other: &Self) -> bool {
        self.same_ctx(other);
        self.to_limbs() == other.to_limbs()
    }
}
impl<B1: RnsBase<N1>, const N1: usize, B2: RnsBase<N2>, const N2: usize> Eq
    for ModElem<'_, B1, N1, B2, N2>
{
}

impl<B1: RnsBase<N1>, const N1: usize, B2: RnsBase<N2>, const N2: usize> fmt::Debug
    for ModElem<'_, B1, N1, B2, N2>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ModElem(0x")?;
        let l = self.to_limbs();
        let top = limbs::sig_len(&l).max(1);
        write!(f, "{:x}", l[top - 1])?;
        for x in l[..top - 1].iter().rev() {
            write!(f, "{x:016x}")?;
        }
        write!(f, ", factor {})", self.factor)
    }
}
