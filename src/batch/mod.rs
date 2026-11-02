//! [`RnsVec`]: many exact integers stored channel by channel (structure of arrays).
//!
//! In a scalar [`Rns`] the `N` residues of one number sit side by side, each
//! with a different modulus. Here it is the other way round: channel `i` is one
//! contiguous `Vec<u32>` holding residue `i` of *every* element. A whole channel
//! is then processed with a single modulus, which is the shape SIMD needs:
//! broadcast `m_i` once, then handle 8 (AVX2) or 16 (AVX-512) elements per
//! instruction.
//!
//! Range tracking is per vector: one certified bound `|x_k| < 2^bits` for all
//! elements, updated by the same rules as for [`Rns`].

mod convert;
mod kernels;
mod matrix;
mod mont_vec;
mod par;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm;
#[cfg(all(feature = "std", target_arch = "x86_64"))]
mod x86;

pub use matrix::RnsMatrix;
pub use mont_vec::ModVec;
#[cfg(feature = "parallel")]
#[doc(hidden)]
pub use par::__set_parallel_threshold;
#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[doc(hidden)]
pub use x86::__set_simd_level;

use alloc::vec::Vec;
use core::fmt;
use core::marker::PhantomData;
use core::ops::{Add, Mul, Neg, Sub};

use crate::base::{Base, RnsBase};
use crate::exact::{OutOfRange, Rns};
use crate::word::Channel;

#[derive(Clone, Copy)]
enum Elementwise {
    Add,
    Sub,
    Mul,
}

struct ZipOp<'a, const N: usize> {
    kind: Elementwise,
    ch: &'a [Channel; N],
    res: &'a mut [Vec<u32>; N],
    red: &'a mut [u64],
    other_res: &'a [Vec<u32>; N],
    other_red: &'a [u64],
}

impl<const N: usize> kernels::Op for ZipOp<'_, N> {
    type Out = ();
    #[inline(always)]
    fn run(self) {
        match self.kind {
            Elementwise::Add => {
                for i in 0..N {
                    kernels::add_assign(self.ch[i], &mut self.res[i], &self.other_res[i]);
                }
                kernels::red_add_assign(self.red, self.other_red);
            }
            Elementwise::Sub => {
                for i in 0..N {
                    kernels::sub_assign(self.ch[i], &mut self.res[i], &self.other_res[i]);
                }
                kernels::red_sub_assign(self.red, self.other_red);
            }
            Elementwise::Mul => {
                for i in 0..N {
                    kernels::mul_assign(self.ch[i], &mut self.res[i], &self.other_res[i]);
                }
                kernels::red_mul_assign(self.red, self.other_red);
            }
        }
    }
}

/// One channel of an element-wise product (the parallel path).
struct MulChannelOp<'a> {
    ch: Channel,
    res: &'a mut [u32],
    other: &'a [u32],
}

impl kernels::Op for MulChannelOp<'_> {
    type Out = ();
    #[inline(always)]
    fn run(self) {
        kernels::mul_assign(self.ch, self.res, self.other);
    }
}

/// The redundant channel of an element-wise product (the parallel path).
struct RedMulOp<'a> {
    red: &'a mut [u64],
    other: &'a [u64],
}

impl kernels::Op for RedMulOp<'_> {
    type Out = ();
    #[inline(always)]
    fn run(self) {
        kernels::red_mul_assign(self.red, self.other);
    }
}

/// One channel of a dot product (the parallel path).
struct DotChannelOp<'a> {
    a: &'a [u32],
    b: &'a [u32],
}

impl kernels::Op for DotChannelOp<'_> {
    type Out = (u64, u64);
    #[inline(always)]
    fn run(self) -> (u64, u64) {
        kernels::dot(self.a, self.b)
    }
}

/// The redundant channel of a dot product (the parallel path).
struct RedDotOp<'a> {
    a: &'a [u64],
    b: &'a [u64],
}

impl kernels::Op for RedDotOp<'_> {
    type Out = u64;
    #[inline(always)]
    fn run(self) -> u64 {
        kernels::red_dot(self.a, self.b)
    }
}

struct ScaleOp<'a, const N: usize> {
    ch: &'a [Channel; N],
    res: &'a mut [Vec<u32>; N],
    red: &'a mut [u64],
    k: [u32; N],
    kred: u64,
}

impl<const N: usize> kernels::Op for ScaleOp<'_, N> {
    type Out = ();
    #[inline(always)]
    fn run(self) {
        for i in 0..N {
            kernels::scale_assign(self.ch[i], &mut self.res[i], self.k[i]);
        }
        kernels::red_scale_assign(self.red, self.kred);
    }
}

struct NegOp<'a, const N: usize> {
    ch: &'a [Channel; N],
    res: &'a mut [Vec<u32>; N],
    red: &'a mut [u64],
}

impl<const N: usize> kernels::Op for NegOp<'_, N> {
    type Out = ();
    #[inline(always)]
    fn run(self) {
        for i in 0..N {
            kernels::neg_assign(self.ch[i], &mut self.res[i]);
        }
        kernels::red_neg_assign(self.red);
    }
}

struct DotOp<'a, const N: usize> {
    a: &'a [Vec<u32>; N],
    b: &'a [Vec<u32>; N],
    a_red: &'a [u64],
    b_red: &'a [u64],
}

#[allow(clippy::needless_range_loop)]
impl<const N: usize> kernels::Op for DotOp<'_, N> {
    type Out = ([(u64, u64); N], u64);
    #[inline(always)]
    fn run(self) -> Self::Out {
        let mut partial = [(0u64, 0u64); N];
        for i in 0..N {
            partial[i] = kernels::dot(&self.a[i], &self.b[i]);
        }
        (partial, kernels::red_dot(self.a_red, self.b_red))
    }
}

/// One channel of [`RnsVec::from_i128s`].
struct FromI128ChannelOp<'a> {
    ch: Channel,
    c: [[u32; 2]; 2],
    lo: &'a [u64],
    hi: &'a [u64],
    neg: &'a [u32],
    out: &'a mut [u32],
}

impl kernels::Op for FromI128ChannelOp<'_> {
    type Out = ();
    #[inline(always)]
    fn run(self) {
        if self.ch.modulus() > 4 {
            kernels::from_two_limbs(self.ch, self.c, self.lo, self.hi, self.neg, self.out);
        } else {
            kernels::from_two_limbs_small(self.ch, self.c, self.lo, self.hi, self.neg, self.out);
        }
    }
}

struct SumOp<'a, const N: usize> {
    a: &'a [Vec<u32>; N],
    a_red: &'a [u64],
}

#[allow(clippy::needless_range_loop)]
impl<const N: usize> kernels::Op for SumOp<'_, N> {
    type Out = ([u64; N], u64);
    #[inline(always)]
    fn run(self) -> Self::Out {
        let mut partial = [0u64; N];
        for i in 0..N {
            partial[i] = kernels::sum(&self.a[i]);
        }
        (partial, kernels::red_sum(self.a_red))
    }
}

/// A vector of exact integers in structure-of-arrays layout. See the module docs.
pub struct RnsVec<B: RnsBase<N>, const N: usize> {
    /// `res[i][k]` = Montgomery residue of element `k` in channel `i`.
    res: [Vec<u32>; N],
    /// `red[k]` = element `k` mod 2^64.
    red: Vec<u64>,
    /// Certified bound for every element: `|x_k| < 2^bits`.
    bits: u32,
    _base: PhantomData<B>,
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

/// `ceil(log2(n))`, with `ceil_log2(0) = ceil_log2(1) = 0`.
#[inline]
fn ceil_log2(n: usize) -> u32 {
    if n <= 1 {
        0
    } else {
        usize::BITS - (n - 1).leading_zeros()
    }
}

#[cold]
#[inline(never)]
fn overflow() -> ! {
    panic!("residue: integer overflow (result may exceed the base's range ±2^max_bits)")
}

#[cold]
#[inline(never)]
fn length_mismatch(a: usize, b: usize) -> ! {
    panic!("residue: RnsVec length mismatch ({a} vs {b})")
}

impl<B: RnsBase<N>, const N: usize> RnsVec<B, N> {
    #[inline]
    fn base() -> &'static Base<N> {
        B::base()
    }

    /// An empty vector.
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    /// An empty vector with room for `cap` elements.
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        RnsVec {
            res: core::array::from_fn(|_| Vec::with_capacity(cap)),
            red: Vec::with_capacity(cap),
            bits: 0,
            _base: PhantomData,
        }
    }

    /// Build from a slice of scalars.
    #[must_use]
    pub fn from_slice(xs: &[Rns<B, N>]) -> Self {
        let mut v = Self::with_capacity(xs.len());
        for &x in xs {
            v.push(x);
        }
        v
    }

    /// Build from `i128` values.
    ///
    /// Vectorized: one pass splits the inputs into magnitude limbs, then each
    /// channel is filled by a SIMD kernel (three Montgomery reductions per
    /// element, no division).
    ///
    /// # Errors
    /// [`OutOfRange`] if any value does not fit the base.
    pub fn from_i128s(xs: &[i128]) -> Result<Self, OutOfRange> {
        let n = xs.len();
        let mut lo = Vec::with_capacity(n);
        let mut hi = Vec::with_capacity(n);
        let mut neg = Vec::with_capacity(n);
        let mut red = Vec::with_capacity(n);
        let mut max_bits = 0u32;
        for &x in xs {
            let m = x.unsigned_abs();
            max_bits = max_bits.max(128 - m.leading_zeros());
            lo.push(m as u64);
            hi.push((m >> 64) as u64);
            neg.push((x < 0) as u32);
            red.push(x as u64);
        }
        if max_bits > Rns::<B, N>::max_bits() {
            return Err(OutOfRange);
        }
        let mut res: [Vec<u32>; N] = core::array::from_fn(|_| alloc::vec![0u32; n]);
        let base = Self::base();
        let ch = base.channels();
        let (lo, hi, neg) = (&lo, &hi, &neg);
        par::for_each_mut(&mut res, par::enabled(4 * N * n), |i, out| {
            kernels::dispatch(FromI128ChannelOp {
                ch: ch[i],
                c: [base.fwd(i, 0), base.fwd(i, 1)],
                lo,
                hi,
                neg,
                out,
            });
        });
        Ok(RnsVec {
            res,
            red,
            bits: max_bits,
            _base: PhantomData,
        })
    }

    /// `n` copies of zero.
    #[must_use]
    pub fn zeros(n: usize) -> Self {
        RnsVec {
            res: core::array::from_fn(|_| alloc::vec![0; n]),
            red: alloc::vec![0; n],
            bits: 0,
            _base: PhantomData,
        }
    }

    /// Append one element.
    pub fn push(&mut self, x: Rns<B, N>) {
        let (r, red, bits) = x.into_parts();
        for (ch, v) in self.res.iter_mut().zip(r) {
            ch.push(v);
        }
        self.red.push(red);
        self.bits = self.bits.max(bits);
    }

    /// Number of elements.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.red.len()
    }

    /// `true` if there are no elements.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.red.is_empty()
    }

    /// The certified bound shared by all elements: `|x_k| < 2^bits_bound()`.
    #[inline]
    #[must_use]
    pub fn bits_bound(&self) -> u32 {
        self.bits
    }

    /// Element `k` as a scalar, carrying the vector's bound.
    #[must_use]
    pub fn get(&self, k: usize) -> Option<Rns<B, N>> {
        if k >= self.len() {
            return None;
        }
        let r = core::array::from_fn(|i| self.res[i][k]);
        Some(Rns::from_parts_unchecked(r, self.red[k], self.bits))
    }

    /// Iterate over the elements as scalars.
    #[must_use]
    pub fn iter(&self) -> impl ExactSizeIterator<Item = Rns<B, N>> + '_ {
        (0..self.len()).map(move |k| self.get(k).unwrap())
    }

    /// Copy out into a `Vec` of scalars.
    #[must_use]
    pub fn to_vec(&self) -> Vec<Rns<B, N>> {
        self.iter().collect()
    }

    /// Replace the shared bound with the exact maximum bit length over all
    /// elements, with one vectorized conversion to binary.
    pub fn tighten(&mut self) {
        if self.bits <= 63 {
            self.bits = self
                .red
                .iter()
                .map(|&r| 64 - (r as i64).unsigned_abs().leading_zeros())
                .max()
                .unwrap_or(0);
            return;
        }
        let w = self.twos_complement_width();
        let mut limbs = alloc::vec![0u64; self.len() * w];
        self.to_twos_complement(w, &mut limbs)
            .expect("the bound's own width always fits");
        let mut mag = alloc::vec![0u64; w];
        self.bits = limbs
            .chunks_exact(w)
            .map(|x| {
                mag.copy_from_slice(x);
                if (x[w - 1] as i64) < 0 {
                    crate::limbs::neg_wrapping(&mut mag);
                }
                crate::limbs::bit_len_slice(&mag) as u32
            })
            .max()
            .unwrap_or(0);
    }

    #[inline]
    fn check_len(&self, other: &Self) {
        if self.len() != other.len() {
            length_mismatch(self.len(), other.len());
        }
    }

    /// `self[k] += other[k]` for all `k`.
    ///
    /// # Errors
    /// [`OutOfRange`], with `self` unchanged, if the certified bound would
    /// exceed `max_bits`.
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn checked_add_assign(&mut self, other: &Self) -> Result<(), OutOfRange> {
        self.check_len(other);
        let bits = add_bits(self.bits, other.bits);
        if bits > Rns::<B, N>::max_bits() {
            return Err(OutOfRange);
        }
        kernels::dispatch(ZipOp {
            kind: Elementwise::Add,
            ch: Self::base().channels(),
            res: &mut self.res,
            red: &mut self.red,
            other_res: &other.res,
            other_red: &other.red,
        });
        self.bits = bits;
        Ok(())
    }

    /// `self[k] -= other[k]` for all `k`.
    ///
    /// # Errors
    /// [`OutOfRange`], with `self` unchanged, if the certified bound would
    /// exceed `max_bits`.
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn checked_sub_assign(&mut self, other: &Self) -> Result<(), OutOfRange> {
        self.check_len(other);
        let bits = add_bits(self.bits, other.bits);
        if bits > Rns::<B, N>::max_bits() {
            return Err(OutOfRange);
        }
        kernels::dispatch(ZipOp {
            kind: Elementwise::Sub,
            ch: Self::base().channels(),
            res: &mut self.res,
            red: &mut self.red,
            other_res: &other.res,
            other_red: &other.red,
        });
        self.bits = bits;
        Ok(())
    }

    /// `self[k] *= other[k]` for all `k`.
    ///
    /// # Errors
    /// [`OutOfRange`], with `self` unchanged, if the certified bound would
    /// exceed `max_bits`.
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn checked_mul_assign(&mut self, other: &Self) -> Result<(), OutOfRange> {
        self.check_len(other);
        let bits = mul_bits(self.bits, other.bits);
        if bits > Rns::<B, N>::max_bits() {
            return Err(OutOfRange);
        }
        if par::enabled(N * self.len()) {
            let ch = Self::base().channels();
            let (res, red) = (&mut self.res, &mut self.red);
            let (other_res, other_red) = (&other.res, &other.red);
            par::join(
                true,
                || {
                    par::for_each_mut(res, true, |i, r| {
                        kernels::dispatch(MulChannelOp {
                            ch: ch[i],
                            res: r,
                            other: &other_res[i],
                        });
                    });
                },
                || {
                    kernels::dispatch(RedMulOp {
                        red,
                        other: other_red,
                    })
                },
            );
        } else {
            kernels::dispatch(ZipOp {
                kind: Elementwise::Mul,
                ch: Self::base().channels(),
                res: &mut self.res,
                red: &mut self.red,
                other_res: &other.res,
                other_red: &other.red,
            });
        }
        self.bits = bits;
        Ok(())
    }

    /// `self[k] *= k_scalar` for all `k`.
    ///
    /// # Errors
    /// [`OutOfRange`], with `self` unchanged, if the certified bound would
    /// exceed `max_bits`.
    pub fn checked_scale_assign(&mut self, k: Rns<B, N>) -> Result<(), OutOfRange> {
        let (kr, kred, kbits) = k.into_parts();
        let bits = mul_bits(self.bits, kbits);
        if bits > Rns::<B, N>::max_bits() {
            return Err(OutOfRange);
        }
        kernels::dispatch(ScaleOp {
            ch: Self::base().channels(),
            res: &mut self.res,
            red: &mut self.red,
            k: kr,
            kred,
        });
        self.bits = bits;
        Ok(())
    }

    /// Negate every element (never overflows).
    pub fn neg_assign(&mut self) {
        kernels::dispatch(NegOp {
            ch: Self::base().channels(),
            res: &mut self.res,
            red: &mut self.red,
        });
    }

    /// Panicking version of [`RnsVec::checked_add_assign`].
    pub fn add_assign(&mut self, other: &Self) {
        self.checked_add_assign(other)
            .unwrap_or_else(|_| overflow())
    }

    /// Panicking version of [`RnsVec::checked_sub_assign`].
    pub fn sub_assign(&mut self, other: &Self) {
        self.checked_sub_assign(other)
            .unwrap_or_else(|_| overflow())
    }

    /// Panicking version of [`RnsVec::checked_mul_assign`].
    pub fn mul_assign(&mut self, other: &Self) {
        self.checked_mul_assign(other)
            .unwrap_or_else(|_| overflow())
    }

    /// Panicking version of [`RnsVec::checked_scale_assign`].
    pub fn scale_assign(&mut self, k: Rns<B, N>) {
        self.checked_scale_assign(k).unwrap_or_else(|_| overflow())
    }

    /// Exact dot product `Σ self[k]·other[k]`, or `None` if the certified bound
    /// `bits_a + bits_b + ⌈log2 len⌉` exceeds `max_bits`.
    ///
    /// Lazy reduction: each channel accumulates raw 64-bit products split into
    /// 32-bit halves (no modular reduction inside the loop) and reduces once
    /// at the end, followed by one Montgomery reduction.
    ///
    /// # Panics
    /// If the lengths differ, or with `2^32` or more elements.
    #[must_use]
    pub fn dot(&self, other: &Self) -> Option<Rns<B, N>> {
        self.check_len(other);
        assert!(
            (self.len() as u64) < (1u64 << 32),
            "dot product of 2^32 or more terms"
        );
        let bits = mul_bits(self.bits, other.bits);
        let bits = if bits == 0 {
            0
        } else {
            bits + ceil_log2(self.len())
        };
        if bits > Rns::<B, N>::max_bits() {
            return None;
        }
        let ch = Self::base().channels();
        let (partial, red) = if par::enabled(N * self.len()) {
            let (a, b) = (&self.res, &other.res);
            let (a_red, b_red) = (&self.red, &other.red);
            let (parts, red) = par::join(
                true,
                || {
                    par::map(N, true, |i| {
                        kernels::dispatch(DotChannelOp { a: &a[i], b: &b[i] })
                    })
                },
                || kernels::dispatch(RedDotOp { a: a_red, b: b_red }),
            );
            let parts: [(u64, u64); N] = parts.try_into().expect("one partial sum per channel");
            (parts, red)
        } else {
            kernels::dispatch(DotOp {
                a: &self.res,
                b: &other.res,
                a_red: &self.red,
                b_red: &other.red,
            })
        };
        let r = core::array::from_fn(|i| {
            let p = ch[i].modulus() as u64;
            let (lo, hi) = partial[i];
            let two32 = (1u64 << 32) % p;
            let v = ((hi % p) * two32 % p + lo % p) % p;
            ch[i].redc(v)
        });
        Some(Rns::from_parts_unchecked(r, red, bits))
    }

    /// Exact sum of all elements, or `None` on overflow of the certified bound.
    #[must_use]
    pub fn sum(&self) -> Option<Rns<B, N>> {
        let bits = if self.bits == 0 {
            0
        } else {
            self.bits + ceil_log2(self.len())
        };
        if bits > Rns::<B, N>::max_bits() {
            return None;
        }
        let ch = Self::base().channels();
        let (partial, red) = kernels::dispatch(SumOp {
            a: &self.res,
            a_red: &self.red,
        });
        let r = core::array::from_fn(|i| (partial[i] % ch[i].modulus() as u64) as u32);
        Some(Rns::from_parts_unchecked(r, red, bits))
    }
}

impl<B: RnsBase<N>, const N: usize> Add for &RnsVec<B, N> {
    type Output = RnsVec<B, N>;
    fn add(self, rhs: Self) -> RnsVec<B, N> {
        let mut out = self.clone();
        out.add_assign(rhs);
        out
    }
}

impl<B: RnsBase<N>, const N: usize> Sub for &RnsVec<B, N> {
    type Output = RnsVec<B, N>;
    fn sub(self, rhs: Self) -> RnsVec<B, N> {
        let mut out = self.clone();
        out.sub_assign(rhs);
        out
    }
}

impl<B: RnsBase<N>, const N: usize> Mul for &RnsVec<B, N> {
    type Output = RnsVec<B, N>;
    fn mul(self, rhs: Self) -> RnsVec<B, N> {
        let mut out = self.clone();
        out.mul_assign(rhs);
        out
    }
}

impl<B: RnsBase<N>, const N: usize> Neg for &RnsVec<B, N> {
    type Output = RnsVec<B, N>;
    fn neg(self) -> RnsVec<B, N> {
        let mut out = self.clone();
        out.neg_assign();
        out
    }
}

impl<B: RnsBase<N>, const N: usize> Clone for RnsVec<B, N> {
    fn clone(&self) -> Self {
        RnsVec {
            res: self.res.clone(),
            red: self.red.clone(),
            bits: self.bits,
            _base: PhantomData,
        }
    }
}

impl<B: RnsBase<N>, const N: usize> Default for RnsVec<B, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: RnsBase<N>, const N: usize> PartialEq for RnsVec<B, N> {
    /// Element-wise equality of values (bounds are not compared).
    fn eq(&self, other: &Self) -> bool {
        self.res == other.res
    }
}
impl<B: RnsBase<N>, const N: usize> Eq for RnsVec<B, N> {}

impl<B: RnsBase<N>, const N: usize> FromIterator<Rns<B, N>> for RnsVec<B, N> {
    fn from_iter<I: IntoIterator<Item = Rns<B, N>>>(iter: I) -> Self {
        let mut v = Self::new();
        v.extend(iter);
        v
    }
}

impl<B: RnsBase<N>, const N: usize> Extend<Rns<B, N>> for RnsVec<B, N> {
    fn extend<I: IntoIterator<Item = Rns<B, N>>>(&mut self, iter: I) {
        for x in iter {
            self.push(x);
        }
    }
}

impl<B: RnsBase<N>, const N: usize> fmt::Debug for RnsVec<B, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RnsVec(len {}, |x| < 2^{}) ", self.len(), self.bits)?;
        f.debug_list()
            .entries(self.iter().map(|x| alloc::format!("{x}")))
            .finish()
    }
}
