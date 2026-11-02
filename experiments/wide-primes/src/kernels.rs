//! Per-channel kernels for the three widths. Every function here works on one
//! channel's residues; the `run_*` wrappers loop over the channels of a base.

use crate::ch50::Ch50;
use crate::ch62::Ch62;
use residue::Channel;

const MASK: u64 = 0xFFFF_FFFF;

/// The library's loop: one output row at a time, `a[i][k]` broadcast against
/// row `k` of `b`, products accumulated in two 32-bit halves held in memory.
pub fn gemm31_naive(
    ch: Channel,
    a: &[u32],
    b: &[u32],
    c: &mut [u32],
    n: usize,
    lo: &mut [u64],
    hi: &mut [u64],
) {
    for i in 0..n {
        lo.fill(0);
        hi.fill(0);
        for k in 0..n {
            let aik = a[i * n + k] as u64;
            let brow = &b[k * n..(k + 1) * n];
            for ((l, h), &bk) in lo.iter_mut().zip(hi.iter_mut()).zip(brow) {
                let t = aik * bk as u64;
                *l += t & MASK;
                *h += t >> 32;
            }
        }
        for j in 0..n {
            c[i * n + j] = ch.redc_halves(lo[j], hi[j]);
        }
    }
}

pub fn run_gemm31_naive(
    ch: &[Channel],
    a: &[Vec<u32>],
    b: &[Vec<u32>],
    c: &mut [Vec<u32>],
    n: usize,
) {
    let mut lo = vec![0u64; n];
    let mut hi = vec![0u64; n];
    for i in 0..ch.len() {
        gemm31_naive(ch[i], &a[i], &b[i], &mut c[i], n, &mut lo, &mut hi);
    }
}

/// Register tiling: a block of `RB × CB` outputs stays in registers for the
/// whole `k` loop. Four products are summed in one `u64` before splitting
/// (each is below `p² < 2^62`, so four fit), which divides the mask, shift and
/// add work of the delayed reduction by four. The `RB` coefficients of `a` for
/// one `k` are read from a transposed copy of the row block, so every load is
/// contiguous.
pub fn gemm31_tiled<const N: usize, const RB: usize, const CB: usize>(
    ch: Channel,
    a: &[u32],
    b: &[u32],
    c: &mut [u32],
    at: &mut [u64],
) {
    assert!(N % RB == 0 && N % CB == 0 && N % 4 == 0 && N / 4 < ch.modulus() as usize);
    assert!(RB * CB <= 64);
    let (a, b) = (&a[..N * N], &b[..N * N]);
    for i0 in (0..N).step_by(RB) {
        for k in 0..N {
            for r in 0..RB {
                at[k * RB + r] = a[(i0 + r) * N + k] as u64;
            }
        }
        for j0 in (0..N).step_by(CB) {
            let mut lo = [0u64; 64];
            let mut hi = [0u64; 64];
            for k0 in (0..N).step_by(4) {
                let mut t = [0u64; 64];
                for k in k0..k0 + 4 {
                    let brow: [u64; CB] = core::array::from_fn(|j| b[k * N + j0 + j] as u64);
                    let ar: [u64; RB] = at[k * RB..(k + 1) * RB].try_into().unwrap();
                    for r in 0..RB {
                        let tr: &mut [u64; CB] = (&mut t[r * CB..(r + 1) * CB]).try_into().unwrap();
                        for (tj, &y) in tr.iter_mut().zip(&brow) {
                            *tj += ar[r] * y;
                        }
                    }
                }
                for ((l, h), &v) in lo[..RB * CB]
                    .iter_mut()
                    .zip(hi[..RB * CB].iter_mut())
                    .zip(&t[..RB * CB])
                {
                    *l += v & MASK;
                    *h += v >> 32;
                }
            }
            for r in 0..RB {
                let out = &mut c[(i0 + r) * N + j0..(i0 + r) * N + j0 + CB];
                for (j, o) in out.iter_mut().enumerate() {
                    *o = ch.redc_halves(lo[r * CB + j], hi[r * CB + j]);
                }
            }
        }
    }
}

#[inline(never)]
pub fn run_gemm31_tiled<const RB: usize, const CB: usize>(
    ch: &[Channel],
    a: &[Vec<u32>],
    b: &[Vec<u32>],
    c: &mut [Vec<u32>],
) {
    let mut at = [0u64; 64 * 8];
    for i in 0..ch.len() {
        gemm31_tiled::<64, RB, CB>(ch[i], &a[i], &b[i], &mut c[i], &mut at);
    }
}

/// Row of `a` times column of `b` (via a transposed copy): each product is a
/// single 64×64→128 multiplication, summed in a `u128` plus a carry counter.
pub fn gemm62(ch: Ch62, a: &[u64], b: &[u64], c: &mut [u64], bt: &mut [u64], n: usize) {
    for k in 0..n {
        for j in 0..n {
            bt[j * n + k] = b[k * n + j];
        }
    }
    for i in 0..n {
        let arow = &a[i * n..(i + 1) * n];
        for j in 0..n {
            let bcol = &bt[j * n..(j + 1) * n];
            let mut acc: u128 = 0;
            let mut top: u64 = 0;
            for (&x, &y) in arow.iter().zip(bcol) {
                let (s, carry) = acc.overflowing_add(x as u128 * y as u128);
                acc = s;
                top += carry as u64;
            }
            c[i * n + j] = ch.reduce_sum(acc, top);
        }
    }
}

pub fn run_gemm62(
    ch: &[Ch62],
    a: &[Vec<u64>],
    b: &[Vec<u64>],
    c: &mut [Vec<u64>],
    bt: &mut [u64],
    n: usize,
) {
    for i in 0..ch.len() {
        gemm62(ch[i], &a[i], &b[i], &mut c[i], bt, n);
    }
}

/// Every product is reduced (an `f64` cannot hold an unreduced sum), but the
/// loop over `j` vectorizes with FMA.
pub fn gemm50(ch: Ch50, a: &[f64], b: &[f64], c: &mut [f64], n: usize) {
    for i in 0..n {
        let crow = &mut c[i * n..(i + 1) * n];
        crow.fill(0.0);
        for k in 0..n {
            let aik = a[i * n + k];
            let brow = &b[k * n..(k + 1) * n];
            for (o, &bk) in crow.iter_mut().zip(brow) {
                *o = ch.add(*o, ch.mul(aik, bk));
            }
        }
    }
}

pub fn run_gemm50(ch: &[Ch50], a: &[Vec<f64>], b: &[Vec<f64>], c: &mut [Vec<f64>], n: usize) {
    for i in 0..ch.len() {
        gemm50(ch[i], &a[i], &b[i], &mut c[i], n);
    }
}

pub trait Field: Copy {
    type E: Copy + PartialEq;
    fn zero(&self) -> Self::E;
    fn one(&self) -> Self::E;
    fn mul(&self, a: Self::E, b: Self::E) -> Self::E;
    fn sub(&self, a: Self::E, b: Self::E) -> Self::E;
    fn neg(&self, a: Self::E) -> Self::E;
    fn inv(&self, a: Self::E) -> Self::E;
}

impl Field for Channel {
    type E = u32;
    fn zero(&self) -> u32 {
        0
    }
    fn one(&self) -> u32 {
        self.to_mont(1)
    }
    #[inline(always)]
    fn mul(&self, a: u32, b: u32) -> u32 {
        self.mul_mont(a, b)
    }
    #[inline(always)]
    fn sub(&self, a: u32, b: u32) -> u32 {
        Channel::sub(self, a, b)
    }
    fn neg(&self, a: u32) -> u32 {
        Channel::neg(self, a)
    }
    fn inv(&self, a: u32) -> u32 {
        self.inv_mont(a)
    }
}

impl Field for Ch62 {
    type E = u64;
    fn zero(&self) -> u64 {
        0
    }
    fn one(&self) -> u64 {
        self.to_mont(1)
    }
    #[inline(always)]
    fn mul(&self, a: u64, b: u64) -> u64 {
        Ch62::mul(self, a, b)
    }
    #[inline(always)]
    fn sub(&self, a: u64, b: u64) -> u64 {
        Ch62::sub(self, a, b)
    }
    fn neg(&self, a: u64) -> u64 {
        Ch62::neg(self, a)
    }
    fn inv(&self, a: u64) -> u64 {
        Ch62::inv(self, a)
    }
}

impl Field for Ch50 {
    type E = f64;
    fn zero(&self) -> f64 {
        0.0
    }
    fn one(&self) -> f64 {
        1.0
    }
    #[inline(always)]
    fn mul(&self, a: f64, b: f64) -> f64 {
        Ch50::mul(self, a, b)
    }
    #[inline(always)]
    fn sub(&self, a: f64, b: f64) -> f64 {
        Ch50::sub(self, a, b)
    }
    fn neg(&self, a: f64) -> f64 {
        Ch50::neg(self, a)
    }
    fn inv(&self, a: f64) -> f64 {
        Ch50::inv(self, a)
    }
}

/// Gaussian elimination with row swaps, the library's algorithm.
pub fn det<F: Field>(f: F, n: usize, a: &mut [F::E]) -> F::E {
    let mut det = f.one();
    for c in 0..n {
        let Some(piv) = (c..n).find(|&r| a[r * n + c] != f.zero()) else {
            return f.zero();
        };
        if piv != c {
            let (top, bottom) = a.split_at_mut(piv * n);
            top[c * n..(c + 1) * n].swap_with_slice(&mut bottom[..n]);
            det = f.neg(det);
        }
        let pv = a[c * n + c];
        det = f.mul(det, pv);
        let inv = f.inv(pv);
        for r in c + 1..n {
            let factor = f.mul(a[r * n + c], inv);
            if factor != f.zero() {
                let (top, bottom) = a.split_at_mut(r * n);
                let pivot_row = &top[c * n + c + 1..(c + 1) * n];
                for (x, &y) in bottom[c + 1..n].iter_mut().zip(pivot_row) {
                    *x = f.sub(*x, f.mul(factor, y));
                }
            }
        }
    }
    det
}

fn run_det<F: Field>(ch: &[F], a: &[Vec<F::E>], work: &mut [F::E], n: usize) -> Vec<F::E> {
    ch.iter()
        .zip(a)
        .map(|(&f, m)| {
            work.copy_from_slice(m);
            det(f, n, work)
        })
        .collect()
}

pub fn run_det31(ch: &[Channel], a: &[Vec<u32>], work: &mut [u32], n: usize) -> Vec<u32> {
    run_det(ch, a, work, n)
}

pub fn run_det62(ch: &[Ch62], a: &[Vec<u64>], work: &mut [u64], n: usize) -> Vec<u64> {
    run_det(ch, a, work, n)
}

pub fn run_det50(ch: &[Ch50], a: &[Vec<f64>], work: &mut [f64], n: usize) -> Vec<f64> {
    run_det(ch, a, work, n)
}

/// The library's loop: 64-bit wrapping products. Vector instruction sets have
/// no cheap 64×64→64 multiplication (AVX-512DQ's `vpmullq` is 3 µops, AVX2
/// emulates it with three 32×32 products).
#[inline(never)]
pub fn red_gemm_u64(a: &[u64], b: &[u64], c: &mut [u64], n: usize) {
    for i in 0..n {
        let crow = &mut c[i * n..(i + 1) * n];
        crow.fill(0);
        for k in 0..n {
            let aik = a[i * n + k];
            for (o, &bk) in crow.iter_mut().zip(&b[k * n..(k + 1) * n]) {
                *o = o.wrapping_add(aik.wrapping_mul(bk));
            }
        }
    }
}

/// Same result from 32-bit pieces: `a·b mod 2^64 = a0·b0 + 2^32·(a0·b1 + a1·b0)`,
/// where the cross terms are needed only modulo `2^32`. The low products use
/// `vpmuludq`, the cross terms 32-bit `vpmulld` on twice as many lanes.
#[inline(never)]
pub fn red_gemm_split(
    a0: &[u32],
    a1: &[u32],
    b0: &[u32],
    b1: &[u32],
    c: &mut [u64],
    s1: &mut [u32],
    n: usize,
) {
    for i in 0..n {
        let crow = &mut c[i * n..(i + 1) * n];
        crow.fill(0);
        s1.fill(0);
        for k in 0..n {
            let (x0, x1) = (a0[i * n + k], a1[i * n + k]);
            let (r0, r1) = (&b0[k * n..(k + 1) * n], &b1[k * n..(k + 1) * n]);
            for (o, &y0) in crow.iter_mut().zip(r0) {
                *o = o.wrapping_add(x0 as u64 * y0 as u64);
            }
            for ((s, &y0), &y1) in s1.iter_mut().zip(r0).zip(r1) {
                *s = s
                    .wrapping_add(x0.wrapping_mul(y1))
                    .wrapping_add(x1.wrapping_mul(y0));
            }
        }
        for (o, &s) in crow.iter_mut().zip(s1.iter()) {
            *o = o.wrapping_add((s as u64) << 32);
        }
    }
}

/// The library's loop with both accumulators in one buffer, interleaved
/// (`acc[2j]` low half, `acc[2j+1]` high half), so a row's two sums share
/// cache lines and their addresses no longer depend on two separate heap
/// allocations.
pub fn gemm31_interleaved(
    ch: Channel,
    a: &[u32],
    b: &[u32],
    c: &mut [u32],
    n: usize,
    acc: &mut [u64],
) {
    for i in 0..n {
        acc.fill(0);
        for k in 0..n {
            let aik = a[i * n + k] as u64;
            let brow = &b[k * n..(k + 1) * n];
            for (pair, &bk) in acc.chunks_exact_mut(2).zip(brow) {
                let t = aik * bk as u64;
                pair[0] += t & MASK;
                pair[1] += t >> 32;
            }
        }
        for j in 0..n {
            c[i * n + j] = ch.redc_halves(acc[2 * j], acc[2 * j + 1]);
        }
    }
}

#[inline(never)]
pub fn run_gemm31_interleaved(
    ch: &[Channel],
    a: &[Vec<u32>],
    b: &[Vec<u32>],
    c: &mut [Vec<u32>],
    n: usize,
) {
    let mut acc = vec![0u64; 2 * n];
    for i in 0..ch.len() {
        gemm31_interleaved(ch[i], &a[i], &b[i], &mut c[i], n, &mut acc);
    }
}

/// Exact copy of the library's `GemmOp::run` body: all channels in one
/// function, then the redundant channel.
#[inline(never)]
pub fn library_body(
    ch: &[Channel],
    a: &[Vec<u32>],
    b: &[Vec<u32>],
    out: &mut [Vec<u32>],
    a_red: &[u64],
    b_red: &[u64],
    out_red: &mut [u64],
    n: usize,
) {
    let (m, k, p) = (n, n, n);
    let mut lo = vec![0u64; p];
    let mut hi = vec![0u64; p];
    for i in 0..ch.len() {
        let c = ch[i];
        let (a, b) = (&a[i], &b[i]);
        let o = &mut out[i];
        for r in 0..m {
            let a_row = &a[r * k..(r + 1) * k];
            lo.fill(0);
            hi.fill(0);
            for (kk, &aik) in a_row.iter().enumerate() {
                let brow = &b[kk * p..(kk + 1) * p];
                let x = aik as u64;
                for ((l, h), &bkj) in lo.iter_mut().zip(hi.iter_mut()).zip(brow) {
                    let t = x * bkj as u64;
                    *l += t & MASK;
                    *h += t >> 32;
                }
            }
            for ((oo, &l), &h) in o[r * p..(r + 1) * p].iter_mut().zip(&lo).zip(&hi) {
                *oo = c.redc_halves(l, h);
            }
        }
    }
    for r in 0..m {
        let orow = &mut out_red[r * p..(r + 1) * p];
        orow.fill(0);
        for (kk, &aik) in a_red[r * k..(r + 1) * k].iter().enumerate() {
            for (oo, &bkj) in orow.iter_mut().zip(&b_red[kk * p..(kk + 1) * p]) {
                *oo = oo.wrapping_add(aik.wrapping_mul(bkj));
            }
        }
    }
}
