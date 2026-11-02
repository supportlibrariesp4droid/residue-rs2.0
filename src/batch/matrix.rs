//! [`RnsMatrix`]: dense matrices of exact integers.
//!
//! Storage is an [`RnsVec`] in row-major order, so each channel is one
//! contiguous `rows × cols` array of residues. Element-wise operations come for
//! free from `RnsVec`; this module adds the matrix product and the determinant.

use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use core::ops::Mul;

use super::{ceil_log2, kernels, mul_bits, overflow, RnsVec};
use crate::base::RnsBase;
use crate::exact::{OutOfRange, Rns};
use crate::word::Channel;

/// A dense `rows × cols` matrix of exact integers in RNS, with one certified
/// bound `|a_ij| < 2^bits` for all entries.
pub struct RnsMatrix<B: RnsBase<N>, const N: usize> {
    rows: usize,
    cols: usize,
    data: RnsVec<B, N>,
}

/// The matrix product in one channel.
///
/// Each channel is dispatched on its own rather than all channels in one
/// dispatched body. The type does not depend on the base, so one compiled copy
/// serves every base, and the loop no longer shares a function with `N` others:
/// in the Phase 0 experiment the same loop ran in 0.46 ms as one function per
/// channel and 0.59 to 0.74 ms inside a single body (see DESIGN.md).
struct GemmChannelOp<'a> {
    ch: Channel,
    m: usize,
    k: usize,
    p: usize,
    a: &'a [u32],
    b: &'a [u32],
    out: &'a mut [u32],
    lo: &'a mut [u64],
    hi: &'a mut [u64],
}

impl kernels::Op for GemmChannelOp<'_> {
    type Out = ();
    #[inline(always)]
    fn run(self) {
        let (m, k, p) = (self.m, self.k, self.p);
        let fast = (k as u64) < self.ch.modulus() as u64;
        for r in 0..m {
            kernels::gemm_row_acc(&self.a[r * k..(r + 1) * k], self.b, p, self.lo, self.hi);
            let o = &mut self.out[r * p..(r + 1) * p];
            if fast {
                kernels::gemm_row_reduce(self.ch, self.lo, self.hi, o);
            } else {
                kernels::gemm_row_reduce_slow(self.ch, self.lo, self.hi, o);
            }
        }
    }
}

/// The matrix product of the redundant channel, modulo `2^64`.
struct GemmRedOp<'a> {
    m: usize,
    k: usize,
    p: usize,
    a: &'a [u64],
    b: &'a [u64],
    out: &'a mut [u64],
}

impl kernels::Op for GemmRedOp<'_> {
    type Out = ();
    #[inline(always)]
    fn run(self) {
        let (m, k, p) = (self.m, self.k, self.p);
        for r in 0..m {
            kernels::gemm_row_red(
                &self.a[r * k..(r + 1) * k],
                self.b,
                p,
                &mut self.out[r * p..(r + 1) * p],
            );
        }
    }
}

/// Gaussian elimination in one channel. Dispatched per channel, like the
/// product, so that channels can run on different threads.
struct DetChannelOp<'a> {
    ch: Channel,
    n: usize,
    a: &'a [u32],
}

impl kernels::Op for DetChannelOp<'_> {
    type Out = u32;
    #[inline(always)]
    fn run(self) -> u32 {
        let mut work = self.a.to_vec();
        kernels::det_channel(self.ch, self.n, &mut work)
    }
}

impl<B: RnsBase<N>, const N: usize> RnsMatrix<B, N> {
    /// Wrap a vector of `rows · cols` entries in row-major order.
    ///
    /// # Panics
    /// If `data.len() != rows * cols`.
    #[must_use]
    pub fn from_vec(rows: usize, cols: usize, data: RnsVec<B, N>) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "residue: matrix data length mismatch"
        );
        RnsMatrix { rows, cols, data }
    }

    /// Build from `i128` entries in row-major order.
    ///
    /// # Errors
    /// [`OutOfRange`] if any entry does not fit the base.
    ///
    /// # Panics
    /// If `xs.len() != rows * cols`.
    pub fn from_i128s(rows: usize, cols: usize, xs: &[i128]) -> Result<Self, OutOfRange> {
        assert_eq!(
            xs.len(),
            rows * cols,
            "residue: matrix data length mismatch"
        );
        Ok(Self::from_vec(rows, cols, RnsVec::from_i128s(xs)?))
    }

    /// Build entry by entry.
    pub fn from_fn(rows: usize, cols: usize, mut f: impl FnMut(usize, usize) -> Rns<B, N>) -> Self {
        let mut data = RnsVec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                data.push(f(r, c));
            }
        }
        RnsMatrix { rows, cols, data }
    }

    /// The `rows × cols` zero matrix.
    #[must_use]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        RnsMatrix {
            rows,
            cols,
            data: RnsVec::zeros(rows * cols),
        }
    }

    /// The `n × n` identity matrix.
    #[must_use]
    pub fn identity(n: usize) -> Self {
        Self::from_fn(n, n, |r, c| if r == c { Rns::one() } else { Rns::zero() })
    }

    /// Number of rows.
    #[inline]
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    #[inline]
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Entry `(r, c)`, or `None` if out of bounds.
    #[must_use]
    pub fn get(&self, r: usize, c: usize) -> Option<Rns<B, N>> {
        if r < self.rows && c < self.cols {
            self.data.get(r * self.cols + c)
        } else {
            None
        }
    }

    /// The certified bound shared by all entries: `|a_ij| < 2^bits_bound()`.
    #[inline]
    #[must_use]
    pub fn bits_bound(&self) -> u32 {
        self.data.bits_bound()
    }

    /// The entries in row-major order.
    #[inline]
    #[must_use]
    pub fn as_vec(&self) -> &RnsVec<B, N> {
        &self.data
    }

    /// Consume the matrix, returning its entries in row-major order.
    #[inline]
    #[must_use]
    pub fn into_vec(self) -> RnsVec<B, N> {
        self.data
    }

    /// Replace the bound with the exact maximum bit length over all entries.
    pub fn tighten(&mut self) {
        self.data.tighten();
    }

    /// The transpose.
    #[must_use]
    pub fn transpose(&self) -> Self {
        Self::from_fn(self.cols, self.rows, |r, c| self.get(c, r).unwrap())
    }

    /// Exact matrix product `self · other`, or `None` if the certified bound
    /// `bits_a + bits_b + ⌈log2 k⌉` (with `k` the inner dimension) exceeds
    /// `max_bits`.
    ///
    /// Each channel is an independent `u32` matrix product with lazy reduction:
    /// raw 64-bit products are accumulated in two halves and every output entry
    /// is reduced once, with three Montgomery reductions and no division (when
    /// `k` is below the smallest modulus; otherwise a slower path is used).
    /// The inner loop runs along rows of `other`, vectorized across columns.
    ///
    /// # Panics
    /// If `self.cols() != other.rows()`.
    #[must_use]
    pub fn checked_matmul(&self, other: &Self) -> Option<Self> {
        assert_eq!(
            self.cols, other.rows,
            "residue: matrix product dimension mismatch ({}×{} · {}×{})",
            self.rows, self.cols, other.rows, other.cols
        );
        let (m, k, p) = (self.rows, self.cols, other.cols);
        let bits = mul_bits(self.bits_bound(), other.bits_bound());
        let bits = if bits == 0 { 0 } else { bits + ceil_log2(k) };
        if bits > Rns::<B, N>::max_bits() {
            return None;
        }
        let mut out = RnsVec::<B, N>::zeros(m * p);
        let ch = B::base().channels();
        let par = super::par::enabled(N * m * k * p);
        let blocks = if par {
            (2 * super::par::threads()).div_ceil(N).clamp(1, m.max(1))
        } else {
            1
        };
        let rows = m.div_ceil(blocks).max(1);
        let mut tasks: Vec<(usize, usize, &mut [u32])> = Vec::with_capacity(N * blocks);
        for (i, o) in out.res.iter_mut().enumerate() {
            for (bi, chunk) in o.chunks_mut((rows * p).max(1)).enumerate() {
                tasks.push((i, bi * rows, chunk));
            }
        }
        let (a_res, b_res) = (&self.data.res, &other.data.res);
        let (a_red, b_red) = (&self.data.red, &other.data.red);
        let red_out = &mut out.red;
        super::par::join(
            par,
            || {
                super::par::for_each_mut(&mut tasks, par, |_, (i, r0, o)| {
                    let mr = o.len() / p.max(1);
                    let a_rows = &a_res[*i][*r0 * k..(*r0 + mr) * k];
                    #[cfg(all(feature = "std", target_arch = "x86_64"))]
                    if (k as u64) < ch[*i].modulus() as u64
                        && super::x86::gemm(&ch[*i..=*i], mr, k, p, a_rows, &b_res[*i], o)
                    {
                        return;
                    }
                    let mut lo = vec![0u64; p];
                    let mut hi = vec![0u64; p];
                    kernels::dispatch(GemmChannelOp {
                        ch: ch[*i],
                        m: mr,
                        k,
                        p,
                        a: a_rows,
                        b: &b_res[*i],
                        out: o,
                        lo: &mut lo,
                        hi: &mut hi,
                    });
                });
            },
            || {
                kernels::dispatch(GemmRedOp {
                    m,
                    k,
                    p,
                    a: a_red,
                    b: b_red,
                    out: red_out,
                })
            },
        );
        out.bits = bits;
        Some(RnsMatrix {
            rows: m,
            cols: p,
            data: out,
        })
    }

    /// A certified bound on the bit length of the determinant, from Hadamard's
    /// inequality `|det A| <= ∏ ||row_i||`. With `n` rows and entries
    /// `|a_ij| < 2^b`: `|det A| < 2^(n·b + ⌈n·⌈log2 n⌉ / 2⌉)`.
    #[must_use]
    pub fn determinant_bits_bound(&self) -> u32 {
        let (n, b) = (self.rows as u32, self.bits_bound());
        if n == 0 || b == 0 {
            return 0;
        }
        n * b + (n * ceil_log2(self.rows)).div_ceil(2)
    }

    /// The exact determinant, or `None` if the Hadamard bound
    /// ([`RnsMatrix::determinant_bits_bound`]) exceeds the base's `max_bits`:
    /// then use a wider base (see [`crate::Extender`]).
    ///
    /// Multi-modular algorithm: Gaussian elimination is run independently in
    /// each channel, over the field `ℤ/m_iℤ`, each with its own pivots; the
    /// `N` determinants mod `m_i` are then combined by the Chinese Remainder
    /// Theorem. No big-integer arithmetic and no division of big integers
    /// happens during the elimination. Intermediate values never need to "fit":
    /// only the final determinant must lie in range, which Hadamard guarantees.
    ///
    /// Cost: `O(N·n³)` word operations (vectorized across columns) plus one
    /// reconstruction.
    ///
    /// # Panics
    /// If the matrix is not square.
    #[must_use]
    pub fn determinant(&self) -> Option<Rns<B, N>> {
        assert_eq!(
            self.rows, self.cols,
            "residue: determinant of a non-square matrix"
        );
        let n = self.rows;
        if n == 0 {
            return Some(Rns::one());
        }
        if self.determinant_bits_bound() > Rns::<B, N>::max_bits() {
            return None;
        }
        let ch = B::base().channels();
        let res = &self.data.res;
        let par = super::par::enabled(2 * N * n * n * n / 3);
        let det = super::par::map(N, par, |i| {
            kernels::dispatch(DetChannelOp {
                ch: ch[i],
                n,
                a: &res[i],
            })
        });
        let det: [u32; N] = det.try_into().expect("one determinant per channel");
        Some(Rns::from_mont_residues_signed(det).expect("Hadamard bound guarantees range"))
    }
}

/// Panicking matrix product. See [`RnsMatrix::checked_matmul`].
impl<B: RnsBase<N>, const N: usize> Mul for &RnsMatrix<B, N> {
    type Output = RnsMatrix<B, N>;
    fn mul(self, rhs: Self) -> RnsMatrix<B, N> {
        self.checked_matmul(rhs).unwrap_or_else(|| overflow())
    }
}

impl<B: RnsBase<N>, const N: usize> Clone for RnsMatrix<B, N> {
    fn clone(&self) -> Self {
        RnsMatrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.clone(),
        }
    }
}

impl<B: RnsBase<N>, const N: usize> PartialEq for RnsMatrix<B, N> {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows && self.cols == other.cols && self.data == other.data
    }
}
impl<B: RnsBase<N>, const N: usize> Eq for RnsMatrix<B, N> {}

impl<B: RnsBase<N>, const N: usize> fmt::Debug for RnsMatrix<B, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "RnsMatrix({}×{}, |a| < 2^{}) [",
            self.rows,
            self.cols,
            self.bits_bound()
        )?;
        for r in 0..self.rows {
            f.write_str("  [")?;
            for c in 0..self.cols {
                if c > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{}", self.get(r, c).unwrap())?;
            }
            f.write_str("]\n")?;
        }
        f.write_str("]")
    }
}
