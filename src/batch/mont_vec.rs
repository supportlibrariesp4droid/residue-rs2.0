//! [`ModVec`]: many elements of `ℤ/mℤ` in a [`MontRns`] context, multiplied
//! together with SIMD across elements.
//!
//! A single RNS Montgomery product costs about `2·N1·N2` word products, more
//! than a big-integer library needs for one product of the same size, so one
//! product at a time is not where RNS wins. Many independent products are
//! different: stored channel by channel, a block of elements turns both base
//! extensions into matrix products,
//!
//! ```text
//! Q̂ (N2 × E) = T1 (N2 × N1) · Ξ  (N1 × E)
//! R  (N1 × E) = T2 (N1 × N2) · Ξ' (N2 × E)
//! ```
//!
//! where `Ξ` holds the `ξ_i` of every element of the block and `T1`, `T2` are
//! the extension tables. Those run on the register-tiled matrix kernels, with
//! one prime per output row; the element-wise steps are vectorized loops.

use alloc::vec;
use alloc::vec::Vec;
use core::ops::{Add, Mul, Sub};

use super::{kernels, par};
use crate::base::RnsBase;
use crate::mont::{ModElem, MontRns, Parts};
use crate::word::Channel;

/// Elements per block: the scratch matrices of a block stay in the L2 cache.
const BLOCK: usize = 256;

/// Many elements of `ℤ/mℤ` in one [`MontRns`] context, stored channel by
/// channel, with one shared factor bound.
///
/// ```
/// use residue::{rns_base, ModVec, MontRns};
///
/// rns_base!(B1: largest 6);
/// rns_base!(B2: largest 6 after 6);
/// let m = [u64::MAX, u64::MAX >> 1];
/// let ctx = MontRns::<B1, 6, B2, 6>::new(&m).unwrap();
///
/// let xs: Vec<_> = (2..10u128).map(|v| ctx.from_u128(v)).collect();
/// let v = ModVec::from_elems(&ctx, &xs);
/// let cubes = v.pow(&[3]);
/// assert_eq!(cubes.get(1).unwrap().to_u128(), Some(27));
/// ```
///
/// **Not constant time**: see the documentation of [`MontRns`].
pub struct ModVec<'c, B1: RnsBase<N1>, const N1: usize, B2: RnsBase<N2>, const N2: usize> {
    ctx: &'c MontRns<B1, N1, B2, N2>,
    r1: [Vec<u32>; N1],
    r2: [Vec<u32>; N2],
    red: Vec<u64>,
    /// Every value is below `factor·m`.
    factor: u64,
}

/// The output of one block of products.
struct BlockOut {
    r1: Vec<u32>,
    r2: Vec<u32>,
    red: Vec<u64>,
}

/// One block of element-wise RNS Montgomery products.
struct MulBlockOp<'a, B1: RnsBase<N1>, const N1: usize, B2: RnsBase<N2>, const N2: usize> {
    ctx: &'a MontRns<B1, N1, B2, N2>,
    a1: [&'a [u32]; N1],
    a2: [&'a [u32]; N2],
    a_red: &'a [u64],
    b1: [&'a [u32]; N1],
    b2: [&'a [u32]; N2],
    b_red: &'a [u64],
}

/// `out (rows × e) = table (rows × k) · x (k × e)`, row `r` reduced with
/// `chs[r]`, through the explicit kernel when available.
#[inline(always)]
fn extend_block(chs: &[Channel], table: &[u32], k: usize, x: &[u32], e: usize, out: &mut [u32]) {
    let fast = chs.iter().all(|c| (k as u64) < c.modulus() as u64);
    #[cfg(all(feature = "std", target_arch = "x86_64"))]
    if fast && super::x86::gemm(chs, chs.len(), k, e, table, x, out) {
        return;
    }
    let mut lo = vec![0u64; e];
    let mut hi = vec![0u64; e];
    for (r, ch) in chs.iter().enumerate() {
        kernels::gemm_row_acc(&table[r * k..(r + 1) * k], x, e, &mut lo, &mut hi);
        let o = &mut out[r * e..(r + 1) * e];
        if fast {
            kernels::gemm_row_reduce(*ch, &lo, &hi, o);
        } else {
            kernels::gemm_row_reduce_slow(*ch, &lo, &hi, o);
        }
    }
}

impl<B1: RnsBase<N1>, const N1: usize, B2: RnsBase<N2>, const N2: usize> kernels::Op
    for MulBlockOp<'_, B1, N1, B2, N2>
{
    type Out = BlockOut;

    #[allow(clippy::needless_range_loop)]
    #[inline(always)]
    fn run(self) -> BlockOut {
        let ctx = self.ctx;
        let (base1, base2) = (B1::base(), B2::base());
        let (ch1, ch2) = (base1.channels(), base2.channels());
        let e = self.a_red.len();

        let mut xi = vec![0u32; N1 * e];
        let mut q_red = vec![0u64; e];
        let mi64 = base1.crt_mi_mod64();
        for i in 0..N1 {
            let (c, w) = (ch1[i], ctx.w[i]);
            let row = &mut xi[i * e..(i + 1) * e];
            for ((o, &x), &y) in row.iter_mut().zip(self.a1[i]).zip(self.b1[i]) {
                *o = c.mul_mont(c.mul_mont(x, y), w);
            }
            kernels::red_mac(&mut q_red, row, mi64[i]);
        }
        let mut r2 = vec![0u32; N2 * e];
        for j in 0..N2 {
            let c = ch2[j];
            for ((o, &x), &y) in r2[j * e..(j + 1) * e]
                .iter_mut()
                .zip(self.a2[j])
                .zip(self.b2[j])
            {
                *o = c.mul_mont(x, y);
            }
        }
        let mut q2 = vec![0u32; N2 * e];
        extend_block(ch2, ctx.ext1.cols.as_flattened(), N1, &xi, e, &mut q2);
        for j in 0..N2 {
            let (c, m2, minv2) = (ch2[j], ctx.m2[j], ctx.minv2[j]);
            for (o, &q) in r2[j * e..(j + 1) * e]
                .iter_mut()
                .zip(&q2[j * e..(j + 1) * e])
            {
                *o = c.mul_mont(c.add(*o, c.mul_mont(q, m2)), minv2);
            }
        }
        let mut red = vec![0u64; e];
        for (((o, &x), &y), &q) in red.iter_mut().zip(self.a_red).zip(self.b_red).zip(&q_red) {
            *o = x
                .wrapping_mul(y)
                .wrapping_add(q.wrapping_mul(ctx.m64))
                .wrapping_mul(ctx.minv64);
        }
        let (y2, mi64_2, minv64_2) = (base2.crt_y(), base2.crt_mi_mod64(), base2.m_inv64());
        let mut xi2 = q2;
        let mut s2 = vec![0u64; e];
        for j in 0..N2 {
            let (c, y) = (ch2[j], y2[j]);
            let row = &mut xi2[j * e..(j + 1) * e];
            for (o, &r) in row.iter_mut().zip(&r2[j * e..(j + 1) * e]) {
                *o = c.mul_mont(r, y);
            }
            kernels::red_mac(&mut s2, row, mi64_2[j]);
        }
        let beta: Vec<u64> = s2
            .iter()
            .zip(&red)
            .map(|(&s, &r)| {
                let b = s.wrapping_sub(r).wrapping_mul(minv64_2);
                debug_assert!(b <= N2 as u64, "invariant broken: beta out of range");
                b
            })
            .collect();
        let mut r1 = xi;
        extend_block(ch1, ctx.ext2.cols.as_flattened(), N2, &xi2, e, &mut r1);
        for i in 0..N1 {
            let (c, mr2) = (ch1[i], ctx.ext2.m_r2[i] as u64);
            for (o, &b) in r1[i * e..(i + 1) * e].iter_mut().zip(&beta) {
                *o = c.sub(*o, c.redc(b * mr2));
            }
        }
        BlockOut { r1, r2, red }
    }
}

impl<'c, B1: RnsBase<N1>, const N1: usize, B2: RnsBase<N2>, const N2: usize>
    ModVec<'c, B1, N1, B2, N2>
{
    /// A vector of the given elements.
    ///
    /// # Panics
    /// If an element belongs to another context.
    #[must_use]
    pub fn from_elems(
        ctx: &'c MontRns<B1, N1, B2, N2>,
        xs: &[ModElem<'c, B1, N1, B2, N2>],
    ) -> Self {
        let mut v = ModVec {
            ctx,
            r1: core::array::from_fn(|_| Vec::with_capacity(xs.len())),
            r2: core::array::from_fn(|_| Vec::with_capacity(xs.len())),
            red: Vec::with_capacity(xs.len()),
            factor: 1,
        };
        for x in xs {
            assert!(
                core::ptr::eq(x.ctx, ctx),
                "residue: elements of different MontRns contexts"
            );
            for (c, &r) in v.r1.iter_mut().zip(&x.p.r1) {
                c.push(r);
            }
            for (c, &r) in v.r2.iter_mut().zip(&x.p.r2) {
                c.push(r);
            }
            v.red.push(x.p.red);
            v.factor = v.factor.max(x.factor);
        }
        v
    }

    /// `n` copies of one element.
    #[must_use]
    pub fn splat(x: ModElem<'c, B1, N1, B2, N2>, n: usize) -> Self {
        ModVec {
            ctx: x.ctx,
            r1: core::array::from_fn(|i| vec![x.p.r1[i]; n]),
            r2: core::array::from_fn(|j| vec![x.p.r2[j]; n]),
            red: vec![x.p.red; n],
            factor: x.factor,
        }
    }

    /// Elements `x mod m` from values given as `width` little-endian limbs each,
    /// element `k` at `input[k·width .. (k+1)·width]`.
    ///
    /// # Panics
    /// If `width == 0` or `input.len()` is not a multiple of `width`.
    #[must_use]
    pub fn from_limbs(ctx: &'c MontRns<B1, N1, B2, N2>, width: usize, input: &[u64]) -> Self {
        assert!(
            width > 0 && input.len().is_multiple_of(width),
            "residue: input length must be a multiple of width"
        );
        let xs: Vec<_> = input
            .chunks_exact(width)
            .map(|x| ctx.from_limbs(x))
            .collect();
        Self::from_elems(ctx, &xs)
    }

    /// Number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.red.len()
    }

    /// `true` if there are no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.red.is_empty()
    }

    /// Element `k`.
    #[must_use]
    pub fn get(&self, k: usize) -> Option<ModElem<'c, B1, N1, B2, N2>> {
        (k < self.len()).then(|| {
            let p = Parts {
                r1: core::array::from_fn(|i| self.r1[i][k]),
                r2: core::array::from_fn(|j| self.r2[j][k]),
                red: self.red[k],
            };
            self.ctx.elem(p, self.factor)
        })
    }

    /// All elements.
    #[must_use]
    pub fn to_elems(&self) -> Vec<ModElem<'c, B1, N1, B2, N2>> {
        (0..self.len())
            .map(|k| self.get(k).expect("in range"))
            .collect()
    }

    /// Canonical representatives in `[0, m)`, as many limbs per element as the
    /// modulus has, element `k` at `[k·w .. (k+1)·w]`.
    #[must_use]
    pub fn to_limbs(&self) -> Vec<u64> {
        self.to_elems().iter().flat_map(|x| x.to_limbs()).collect()
    }

    fn check(&self, other: &Self) {
        assert!(
            core::ptr::eq(self.ctx, other.ctx),
            "residue: vectors of different MontRns contexts"
        );
        assert_eq!(self.len(), other.len(), "residue: length mismatch");
    }

    /// Element-wise product.
    fn product(&self, other: &Self) -> Self {
        self.check(other);
        let ctx = self.ctx;
        let f = ctx.product_factor(self.factor, other.factor);
        let n = self.len();
        let par = par::enabled(N1 * N2 * n);
        let block = if par {
            n.div_ceil(par::threads())
                .next_multiple_of(16)
                .clamp(16, BLOCK)
        } else {
            BLOCK
        };
        let blocks = n.div_ceil(block);
        let outs = par::map(blocks, par, |bi| {
            let r = bi * block..(bi * block + block).min(n);
            kernels::dispatch(MulBlockOp {
                ctx,
                a1: core::array::from_fn(|i| &self.r1[i][r.clone()]),
                a2: core::array::from_fn(|j| &self.r2[j][r.clone()]),
                a_red: &self.red[r.clone()],
                b1: core::array::from_fn(|i| &other.r1[i][r.clone()]),
                b2: core::array::from_fn(|j| &other.r2[j][r.clone()]),
                b_red: &other.red[r.clone()],
            })
        });
        let mut v = ModVec {
            ctx,
            r1: core::array::from_fn(|_| Vec::with_capacity(n)),
            r2: core::array::from_fn(|_| Vec::with_capacity(n)),
            red: Vec::with_capacity(n),
            factor: f,
        };
        for o in &outs {
            let e = o.red.len();
            for (i, c) in v.r1.iter_mut().enumerate() {
                c.extend_from_slice(&o.r1[i * e..(i + 1) * e]);
            }
            for (j, c) in v.r2.iter_mut().enumerate() {
                c.extend_from_slice(&o.r2[j * e..(j + 1) * e]);
            }
            v.red.extend_from_slice(&o.red);
        }
        v
    }

    /// The same elements with the factor reset to at most `N1 + 2`.
    #[must_use]
    pub fn normalize(&self) -> Self {
        self.product(&Self::splat(self.ctx.one(), self.len()))
    }

    fn sum(&self, other: &Self) -> Self {
        let (ch1, ch2) = (B1::base().channels(), B2::base().channels());
        ModVec {
            ctx: self.ctx,
            r1: core::array::from_fn(|i| {
                self.r1[i]
                    .iter()
                    .zip(&other.r1[i])
                    .map(|(&x, &y)| ch1[i].add(x, y))
                    .collect()
            }),
            r2: core::array::from_fn(|j| {
                self.r2[j]
                    .iter()
                    .zip(&other.r2[j])
                    .map(|(&x, &y)| ch2[j].add(x, y))
                    .collect()
            }),
            red: self
                .red
                .iter()
                .zip(&other.red)
                .map(|(&x, &y)| x.wrapping_add(y))
                .collect(),
            factor: self.factor + other.factor,
        }
    }

    fn plus(&self, other: &Self) -> Self {
        self.check(other);
        if self.factor + other.factor <= self.ctx.fmax {
            self.sum(other)
        } else {
            self.normalize().sum(&other.normalize())
        }
    }

    /// `factor·m − self`, element-wise.
    fn negated(&self) -> Self {
        let (ch1, ch2) = (B1::base().channels(), B2::base().channels());
        let mp = &self.ctx.m_parts;
        let f = self.factor;
        let fm1: [u32; N1] = core::array::from_fn(|i| {
            ch1[i].mul_mont(
                mp.r1[i],
                ch1[i].to_mont((f % ch1[i].modulus() as u64) as u32),
            )
        });
        let fm2: [u32; N2] = core::array::from_fn(|j| {
            ch2[j].mul_mont(
                mp.r2[j],
                ch2[j].to_mont((f % ch2[j].modulus() as u64) as u32),
            )
        });
        ModVec {
            ctx: self.ctx,
            r1: core::array::from_fn(|i| {
                self.r1[i].iter().map(|&x| ch1[i].sub(fm1[i], x)).collect()
            }),
            r2: core::array::from_fn(|j| {
                self.r2[j].iter().map(|&x| ch2[j].sub(fm2[j], x)).collect()
            }),
            red: self
                .red
                .iter()
                .map(|&x| f.wrapping_mul(mp.red).wrapping_sub(x))
                .collect(),
            factor: f,
        }
    }

    /// Element-wise square.
    #[must_use]
    pub fn square(&self) -> Self {
        self.product(self)
    }

    /// Every element raised to the same exponent `e` (little-endian limbs),
    /// with a 4-bit window: the batch counterpart of [`ModElem::pow`], and the
    /// case where RNS pays, e.g. many RSA operations with one key.
    #[must_use]
    pub fn pow(&self, e: &[u64]) -> Self {
        let e = &e[..crate::limbs::sig_len(e)];
        let one = Self::splat(self.ctx.one(), self.len());
        if e.is_empty() {
            return one;
        }
        let mut table: Vec<Self> = Vec::with_capacity(16);
        table.push(one);
        table.push(self.clone());
        for k in 2..16 {
            let next = table[k - 1].product(self);
            table.push(next);
        }
        let nib = |i: usize| ((e[i / 16] >> (4 * (i % 16))) & 15) as usize;
        let mut top = e.len() * 16;
        while top > 0 && nib(top - 1) == 0 {
            top -= 1;
        }
        let mut acc = table[nib(top - 1)].clone();
        for i in (0..top - 1).rev() {
            for _ in 0..4 {
                acc = acc.square();
            }
            acc = acc.product(&table[nib(i)]);
        }
        acc
    }

    /// The shared factor bound (values are below `factor·m`).
    #[must_use]
    pub fn factor(&self) -> u64 {
        self.factor
    }
}

impl<B1: RnsBase<N1>, const N1: usize, B2: RnsBase<N2>, const N2: usize> Clone
    for ModVec<'_, B1, N1, B2, N2>
{
    fn clone(&self) -> Self {
        ModVec {
            ctx: self.ctx,
            r1: self.r1.clone(),
            r2: self.r2.clone(),
            red: self.red.clone(),
            factor: self.factor,
        }
    }
}

impl<'c, B1: RnsBase<N1>, const N1: usize, B2: RnsBase<N2>, const N2: usize> Mul
    for &ModVec<'c, B1, N1, B2, N2>
{
    type Output = ModVec<'c, B1, N1, B2, N2>;
    fn mul(self, rhs: Self) -> ModVec<'c, B1, N1, B2, N2> {
        self.product(rhs)
    }
}

impl<'c, B1: RnsBase<N1>, const N1: usize, B2: RnsBase<N2>, const N2: usize> Add
    for &ModVec<'c, B1, N1, B2, N2>
{
    type Output = ModVec<'c, B1, N1, B2, N2>;
    fn add(self, rhs: Self) -> ModVec<'c, B1, N1, B2, N2> {
        self.plus(rhs)
    }
}

impl<'c, B1: RnsBase<N1>, const N1: usize, B2: RnsBase<N2>, const N2: usize> Sub
    for &ModVec<'c, B1, N1, B2, N2>
{
    type Output = ModVec<'c, B1, N1, B2, N2>;
    fn sub(self, rhs: Self) -> ModVec<'c, B1, N1, B2, N2> {
        self.check(rhs);
        self.plus(&rhs.negated())
    }
}
