//! Exact base extension: move an [`Rns`] to another base without leaving RNS.
//!
//! For any target modulus `q`,
//! `x mod q = (Σ ξ_i·(M_i mod q) − α'·(M mod q)) mod q`,
//! which is exact because `α'` is exact (see [`crate::exact`]). With the table
//! `M_i mod q_j` precomputed once, each extension costs `O(N_s · N_t)` word
//! operations: no reconstruction, no big-integer arithmetic, no sequential chain.
//!
//! The table is stored already multiplied by `R² mod q_j` (Montgomery form
//! squared). The sum for one target channel is then a delayed-reduction dot
//! product followed by a single [`Channel::reduce_halves`](crate::Channel::reduce_halves),
//! whose `R⁻¹` turns `R²` into the `R` of the Montgomery form: the result
//! comes out in Montgomery form with no division and no separate conversion.

use core::marker::PhantomData;

use crate::base::RnsBase;
use crate::exact::Rns;
use crate::limbs;

/// Precomputed tables to convert exact integers from base `S` (`NS` channels)
/// to base `T` (`NT` channels). Build once, reuse for many values.
///
/// The target may be larger (widening, to make room for bigger results),
/// smaller (narrowing), or simply different. Moduli may overlap. Here the
/// value moves from 4 primes (122-bit integers) to 9 (277-bit integers):
///
/// ```
/// use residue::{rns_base, Extender, Rns};
///
/// rns_base!(Narrow: largest 4);
/// rns_base!(Wide: largest 9);
///
/// let ext = Extender::<Narrow, 4, Wide, 9>::new();
/// let x: Rns<Narrow, 4> = "-123456789012345678901234567890".parse().unwrap();
/// let y: Rns<Wide, 9> = ext.extend(&x).unwrap();
/// assert_eq!(x.to_string(), y.to_string());
/// ```
pub struct Extender<S: RnsBase<NS>, const NS: usize, T: RnsBase<NT>, const NT: usize> {
    /// `cols[j][i] = (M_i mod q_j)·R² mod q_j`, one contiguous row per target.
    pub(crate) cols: [[u32; NS]; NT],
    /// `(M mod q_j)·R² mod q_j`.
    pub(crate) m_r2: [u32; NT],
    _bases: PhantomData<fn() -> (S, T)>,
}

impl<S: RnsBase<NS>, const NS: usize, T: RnsBase<NT>, const NT: usize> Extender<S, NS, T, NT> {
    /// Precompute the extension tables: `O(NS² · NT)` word operations, once.
    #[must_use]
    pub fn new() -> Self {
        let src = S::base();
        let dst = T::base().channels();
        let r2 = |j: usize, c: u32| dst[j].to_mont(dst[j].to_mont(c));
        let cols = core::array::from_fn(|j| {
            core::array::from_fn(|i| r2(j, src.crt_mi_mod(i, dst[j].modulus())))
        });
        let m_r2 = core::array::from_fn(|j| {
            r2(j, limbs::mod_small(src.modulus_limbs(), dst[j].modulus()))
        });
        Extender {
            cols,
            m_r2,
            _bases: PhantomData,
        }
    }

    /// Convert `x` to the target base. `None` if `x`'s certified bound exceeds
    /// the target's `max_bits` (only possible when narrowing).
    #[must_use]
    pub fn extend(&self, x: &Rns<S, NS>) -> Option<Rns<T, NT>> {
        if x.bits_bound() > T::base().max_bits() {
            return None;
        }
        let (xi, alpha) = x.crt_parts();
        let dst = T::base().channels();
        let r = core::array::from_fn(|j| {
            let ch = dst[j];
            let (lo, hi) = crate::word::dot_halves(&xi, &self.cols[j]);
            let pos = ch.reduce_halves(lo, hi);
            let neg = ch.redc(alpha * self.m_r2[j] as u64);
            ch.sub(pos, neg)
        });
        Some(Rns::from_parts_unchecked(r, x.redundant(), x.bits_bound()))
    }
}

impl<S: RnsBase<NS>, const NS: usize, T: RnsBase<NT>, const NT: usize> Default
    for Extender<S, NS, T, NT>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<B: RnsBase<N>, const N: usize> Rns<B, N> {
    /// Convert to another base in one call. Convenience wrapper that builds an
    /// [`Extender`] each time; for repeated conversions build it once.
    #[must_use]
    pub fn rebase<T: RnsBase<NT>, const NT: usize>(&self) -> Option<Rns<T, NT>> {
        Extender::<B, N, T, NT>::new().extend(self)
    }
}
