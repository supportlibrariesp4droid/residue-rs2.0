//! RNS bases: a fixed set of distinct primes plus every derived constant,
//! all computed at compile time.

use crate::limbs;
use crate::primes::is_prime;
use crate::word::Channel;

/// `a^{-1} mod m` via the extended Euclidean algorithm. Panics if not invertible.
const fn modinv(a: u32, m: u32) -> u32 {
    let (mut r0, mut r1) = (m as i64, (a % m) as i64);
    let (mut t0, mut t1) = (0i64, 1i64);
    while r1 != 0 {
        let q = r0 / r1;
        (r0, r1) = (r1, r0 - q * r1);
        (t0, t1) = (t1, t0 - q * t1);
    }
    assert!(r0 == 1, "value is not invertible modulo m");
    (if t0 < 0 { t0 + m as i64 } else { t0 }) as u32
}

/// A set of `N` distinct odd primes below `2^31` and the constants derived from
/// them. Build one with [`Base::new`] in a `const`/`static` context (or, more
/// conveniently, with the [`rns_base!`](crate::rns_base) macro): invalid bases
/// are then rejected at compile time.
#[derive(Debug)]
pub struct Base<const N: usize> {
    channels: [Channel; N],
    /// `garner_inv[i][j] = m_i^{-1} mod m_j` for `i < j` (ordinary form).
    garner_inv: [[u32; N]; N],
    /// Shoup factors of `garner_inv`, for division-free multiplication.
    garner_shoup: [[u32; N]; N],
    /// `M = ∏ m_i`, little-endian limbs.
    modulus: [u64; N],
    /// Bit length of `M`.
    bits: u32,

    /// The cofactors `M_i = M / m_i`, stored by columns: `crt_cols[j][i]` is
    /// limb `j` of `M_i`. Reconstruction sums `ξ_i·M_i` one limb position at a
    /// time, so each position reads one contiguous column.
    crt_cols: [[u64; N]; N],
    /// `y_i = M_i^{-1} mod m_i` (ordinary form).
    crt_y: [u32; N],
    /// `M_i mod 2^64`.
    crt_mi_mod64: [u64; N],
    /// `M^{-1} mod 2^64` (`M` is odd).
    m_inv64: u64,
    /// `floor(2^64 / m_i)`: fixed-point reciprocals for the fast sign test.
    frac_inv64: [u64; N],

    /// Number of 64-bit limbs that can be non-zero for any in-range value:
    /// `ceil(bits(M) / 64)`. Limb loops stop here instead of at `N`.
    limbs_used: usize,
    /// `fwd[i][j] = [R²·2^(64j) mod m_i, R³·2^(64j) mod m_i]` with `R = 2^32`.
    /// A limb `ℓ = h·2^32 + l` at position `j` then has Montgomery form
    /// `REDC(l·fwd[i][j][0]) + REDC(h·fwd[i][j][1])`: two Montgomery reductions,
    /// no division (see [`Base::mont_from_limbs`]).
    fwd: [[[u32; 2]; N]; N],
}

/// `a^{-1} mod 2^64` for odd `a`, Newton-Hensel lifting (3, 6, 12, ..., 96 correct bits).
const fn inv_mod_2_64(a: u64) -> u64 {
    let mut x = a;
    let mut i = 0;
    while i < 6 {
        x = x.wrapping_mul(2u64.wrapping_sub(a.wrapping_mul(x)));
        i += 1;
    }
    x
}

impl<const N: usize> Base<N> {
    /// Validate the moduli and precompute all constants.
    ///
    /// # Panics
    /// If `N == 0`, or any modulus is not a prime in `[3, 2^31)`, or two moduli
    /// are equal. In a `const`/`static` initializer this is a compile error.
    #[must_use]
    pub const fn new(moduli: [u32; N]) -> Self {
        assert!(N >= 1, "a base needs at least one modulus");
        let mut i = 0;
        while i < N {
            let p = moduli[i];
            assert!(p >= 3 && p < (1 << 31), "moduli must lie in [3, 2^31)");
            assert!(is_prime(p), "moduli must be prime");
            let mut j = 0;
            while j < i {
                assert!(moduli[j] != p, "moduli must be distinct");
                j += 1;
            }
            i += 1;
        }

        let mut channels = [Channel::new(3); N];
        let mut i = 0;
        while i < N {
            channels[i] = Channel::new(moduli[i]);
            i += 1;
        }

        let mut garner_inv = [[0u32; N]; N];
        let mut garner_shoup = [[0u32; N]; N];
        let mut i = 0;
        while i < N {
            let mut j = i + 1;
            while j < N {
                let inv = modinv(moduli[i], moduli[j]);
                garner_inv[i][j] = inv;
                garner_shoup[i][j] = channels[j].shoup_factor(inv);
                j += 1;
            }
            i += 1;
        }

        let mut modulus = [0u64; N];
        modulus[0] = 1;
        let mut i = 0;
        while i < N {
            modulus = limbs::mul_small_add(modulus, moduli[i], 0);
            i += 1;
        }
        let bits = limbs::bit_len(&modulus);

        let mut crt_cols = [[0u64; N]; N];
        let mut crt_y = [0u32; N];
        let mut crt_mi_mod64 = [0u64; N];
        let mut frac_inv64 = [0u64; N];
        let mut i = 0;
        while i < N {
            let mut mi = [0u64; N];
            mi[0] = 1;
            let mut mi_mod_mi: u64 = 1;
            let mut k = 0;
            while k < N {
                if k != i {
                    mi = limbs::mul_small_add(mi, moduli[k], 0);
                    mi_mod_mi = mi_mod_mi * (moduli[k] % moduli[i]) as u64 % moduli[i] as u64;
                }
                k += 1;
            }
            let mut j = 0;
            while j < N {
                crt_cols[j][i] = mi[j];
                j += 1;
            }
            crt_y[i] = modinv(mi_mod_mi as u32, moduli[i]);
            crt_mi_mod64[i] = mi[0];
            frac_inv64[i] = ((1u128 << 64) / moduli[i] as u128) as u64;
            i += 1;
        }
        let m_inv64 = inv_mod_2_64(modulus[0]);

        let limbs_used = (bits as usize).div_ceil(64);
        let mut fwd = [[[0u32; 2]; N]; N];
        let mut i = 0;
        while i < N {
            let p = moduli[i] as u64;
            let r = (1u64 << 32) % p;
            let r2 = r * r % p;
            let r3 = r2 * r % p;
            let two64 = r * r % p;
            let mut pow = 1u64;
            let mut j = 0;
            while j < N {
                fwd[i][j] = [(r2 * pow % p) as u32, (r3 * pow % p) as u32];
                pow = pow * two64 % p;
                j += 1;
            }
            i += 1;
        }

        Base {
            channels,
            garner_inv,
            garner_shoup,
            modulus,
            bits,
            crt_cols,
            crt_y,
            crt_mi_mod64,
            m_inv64,
            frac_inv64,
            limbs_used,
            fwd,
        }
    }

    /// The two Montgomery power constants for limb position `j` in channel `i`.
    #[cfg(feature = "alloc")]
    #[inline(always)]
    pub(crate) fn fwd(&self, i: usize, j: usize) -> [u32; 2] {
        if j < N {
            self.fwd[i][j]
        } else {
            [0, 0]
        }
    }

    /// Limbs that can be non-zero for in-range values (`ceil(bits(M)/64)`).
    #[inline]
    pub(crate) fn limbs_used(&self) -> usize {
        self.limbs_used
    }

    /// Montgomery residue in channel `i` of the integer given by `limbs`
    /// (little-endian, at most `limbs_used()` significant limbs), without any
    /// division.
    ///
    /// Each limb `ℓ = h·2^32 + l` at position `j` contributes
    /// `REDC(l·fwd[i][j][0]) + REDC(h·fwd[i][j][1])`. REDC is linear modulo `p`,
    /// so instead of reducing every product, the products are accumulated in
    /// two halves and reduced once with [`Channel::reduce_halves`]: three
    /// reductions per channel instead of two per limb. Each product is below
    /// `2^63`, so the halves cannot overflow.
    #[inline(always)]
    pub(crate) fn mont_from_limbs(&self, i: usize, limbs: &[u64]) -> u32 {
        debug_assert!(limbs.len() <= N);
        let (mut lo, mut hi) = (0u64, 0u64);
        for (&l, &[c_lo, c_hi]) in limbs.iter().zip(&self.fwd[i]) {
            let a = (l & 0xFFFF_FFFF) * c_lo as u64;
            let b = (l >> 32) * c_hi as u64;
            lo += (a & 0xFFFF_FFFF) + (b & 0xFFFF_FFFF);
            hi += (a >> 32) + (b >> 32);
        }
        self.channels[i].reduce_halves(lo, hi)
    }

    /// Largest bit length an exact integer ([`Rns`](crate::Rns)) may have on this
    /// base: values satisfy `|x| < 2^max_bits`. Equal to `bits() - 2`, which
    /// guarantees `|x| < M/2`, so the signed representation is unique.
    #[inline]
    #[must_use]
    pub const fn max_bits(&self) -> u32 {
        self.bits - 2
    }

    /// Limbs of the cofactors `M_i` by column; see the field documentation.
    #[inline]
    pub(crate) fn crt_cols(&self) -> &[[u64; N]; N] {
        &self.crt_cols
    }

    /// `M_i mod q` for a modulus `q < 2^32`, by Horner's rule over the digits.
    pub(crate) fn crt_mi_mod(&self, i: usize, q: u32) -> u32 {
        let q = q as u64;
        let mut r = 0u64;
        let mut j = N;
        while j > 0 {
            j -= 1;
            let limb = self.crt_cols[j][i];
            r = ((r << 32) | (limb >> 32)) % q;
            r = ((r << 32) | (limb & 0xFFFF_FFFF)) % q;
        }
        r as u32
    }
    #[inline]
    pub(crate) fn crt_y(&self) -> &[u32; N] {
        &self.crt_y
    }
    #[inline]
    pub(crate) fn crt_mi_mod64(&self) -> &[u64; N] {
        &self.crt_mi_mod64
    }
    #[inline]
    pub(crate) fn m_inv64(&self) -> u64 {
        self.m_inv64
    }
    #[inline]
    pub(crate) fn frac_inv64(&self) -> &[u64; N] {
        &self.frac_inv64
    }

    /// Number of channels.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        N
    }

    /// Always `false`: a base has at least one channel.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// The per-channel arithmetic engines.
    #[inline]
    #[must_use]
    pub const fn channels(&self) -> &[Channel; N] {
        &self.channels
    }

    /// The moduli, in channel order.
    #[must_use]
    pub fn moduli(&self) -> [u32; N] {
        self.channels.map(|c| c.modulus())
    }

    /// The dynamic range `M = ∏ m_i` as little-endian `u64` limbs.
    #[inline]
    #[must_use]
    pub const fn modulus_limbs(&self) -> &[u64; N] {
        &self.modulus
    }

    /// Bit length of `M`: every value in `[0, M)` fits in this many bits.
    #[inline]
    #[must_use]
    pub const fn bits(&self) -> u32 {
        self.bits
    }

    #[inline]
    pub(crate) fn garner(&self, i: usize, j: usize) -> (u32, u32) {
        (self.garner_inv[i][j], self.garner_shoup[i][j])
    }
}

/// A type-level handle to a `'static` [`Base`]. Implemented by the zero-sized
/// marker types that [`rns_base!`](crate::rns_base) generates, so numbers on
/// different bases are different types and cannot be mixed by accident.
pub trait RnsBase<const N: usize>: Copy + 'static {
    /// The base this marker stands for.
    fn base() -> &'static Base<N>;
}

/// Declare a zero-sized marker type for an RNS base.
///
/// A base is given as the `N` largest primes below `2^31` (`largest 4`, about
/// 124 bits of dynamic range), as an explicit list of primes, or as the `N`
/// largest primes after the first `K`, which gives a second base disjoint
/// from the first:
///
/// ```
/// use residue::{rns_base, RnsMod};
///
/// rns_base!(pub B4: largest 4);
/// rns_base!(Small: 3 = [101, 103, 107]);
/// rns_base!(pub B4b: largest 4 after 4);
///
/// let x = RnsMod::<B4, 4>::from_u64(42);
/// assert_eq!(x.to_u128(), Some(42));
/// ```
///
/// The base is checked and its constants are computed at compile time; an
/// invalid list (a composite such as 102, a duplicate, a value `>= 2^31`) does
/// not compile:
///
/// ```compile_fail
/// residue::rns_base!(Bad: 2 = [101, 102]);
/// let _ = <Bad as residue::RnsBase<2>>::base();
/// ```
///
/// ```compile_fail
/// residue::rns_base!(Dup: 2 = [101, 101]);
/// let _ = <Dup as residue::RnsBase<2>>::base();
/// ```
#[macro_export]
macro_rules! rns_base {
    ($vis:vis $name:ident : largest $n:literal after $skip:literal) => {
        $crate::rns_base!($vis $name : $n = $crate::primes::largest_primes_after::<$n>($skip));
    };
    ($vis:vis $name:ident : largest $n:literal) => {
        $crate::rns_base!($vis $name : $n = $crate::primes::largest_primes::<$n>());
    };
    ($vis:vis $name:ident : $n:literal = $moduli:expr) => {
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
        $vis struct $name;
        impl $crate::RnsBase<$n> for $name {
            #[inline]
            fn base() -> &'static $crate::Base<$n> {
                static BASE: $crate::Base<$n> = $crate::Base::new($moduli);
                &BASE
            }
        }
    };
}
