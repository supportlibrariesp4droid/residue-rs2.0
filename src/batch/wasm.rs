//! WebAssembly `simd128` versions of the hottest portable loops.
//!
//! WebAssembly has no runtime feature detection: these are compiled only when
//! the whole crate is built with `-C target-feature=+simd128`, and then the
//! intrinsics of `core::arch::wasm32` are safe to call.
//!
//! The point is `i64x2.extmul_{low,high}_i32x4_u`, the 32×32→64 multiply that
//! runtimes map to one `pmuludq` (x86) or `umull` (ARM). Left to itself, LLVM
//! vectorizes the same loops with `i64x2.mul`, a full 64×64 multiply that no
//! SSE or NEON instruction implements and that runtimes expand into several.
//! Writing `extmul` explicitly is not enough either: LLVM turns it back into
//! `i64x2.mul` when an operand is a load (which it splits into two extending
//! loads) or a value hoisted out of the loop. Passing the operands through
//! [`core::hint::black_box`] keeps them opaque, at the price of a store and a
//! load each, which measured cheaper than the multiplies they replace.

use core::arch::wasm32::*;
use core::hint::black_box;

use crate::word::Channel;

#[inline(always)]
fn load(x: &[u32]) -> v128 {
    u32x4(x[0], x[1], x[2], x[3])
}

#[inline(always)]
fn store(v: v128, x: &mut [u32]) {
    x[0] = u32x4_extract_lane::<0>(v);
    x[1] = u32x4_extract_lane::<1>(v);
    x[2] = u32x4_extract_lane::<2>(v);
    x[3] = u32x4_extract_lane::<3>(v);
}

/// The two 64-bit products of the low and the high lane pairs of `a·b`.
#[inline(always)]
fn extmul(a: v128, b: v128) -> (v128, v128) {
    let (a, b) = (black_box(a), black_box(b));
    (u64x2_extmul_low_u32x4(a, b), u64x2_extmul_high_u32x4(a, b))
}

/// Montgomery product of four lanes, as [`Channel::mul_mont`].
#[inline(always)]
fn mul_mont4(a: v128, b: v128, p: v128, neg_pinv: v128) -> v128 {
    let (tl, th) = extmul(a, b);
    let m = i32x4_mul(i32x4_shuffle::<0, 2, 4, 6>(tl, th), neg_pinv);
    let (ml, mh) = extmul(m, p);
    let u = i32x4_shuffle::<1, 3, 5, 7>(i64x2_add(tl, ml), i64x2_add(th, mh));
    u32x4_min(u, i32x4_sub(u, p))
}

/// `a[k] = a[k]·b[k]` in Montgomery form.
#[inline(always)]
pub(crate) fn mul_assign(ch: Channel, a: &mut [u32], b: &[u32]) {
    let p = u32x4_splat(ch.modulus());
    let np = u32x4_splat(ch.neg_pinv());
    let mut ac = a.chunks_exact_mut(4);
    let mut bc = b.chunks_exact(4);
    for (x, y) in (&mut ac).zip(&mut bc) {
        store(mul_mont4(load(x), load(y), p, np), x);
    }
    for (x, &y) in ac.into_remainder().iter_mut().zip(bc.remainder()) {
        *x = ch.mul_mont(*x, y);
    }
}

/// `Σ a_k·b_k` as two sums of 32-bit halves, as [`crate::word::dot_halves`].
#[inline(always)]
pub(crate) fn dot(a: &[u32], b: &[u32]) -> (u64, u64) {
    let mask = u64x2_splat(0xFFFF_FFFF);
    let (mut lo, mut hi) = (u64x2_splat(0), u64x2_splat(0));
    let ac = a.chunks_exact(4);
    let bc = b.chunks_exact(4);
    let (ar, br) = (ac.remainder(), bc.remainder());
    for (x, y) in ac.zip(bc) {
        let (pl, ph) = extmul(load(x), load(y));
        lo = i64x2_add(lo, i64x2_add(v128_and(pl, mask), v128_and(ph, mask)));
        hi = i64x2_add(hi, i64x2_add(u64x2_shr(pl, 32), u64x2_shr(ph, 32)));
    }
    let (rl, rh) = crate::word::dot_halves(ar, br);
    (
        rl + u64x2_extract_lane::<0>(lo) + u64x2_extract_lane::<1>(lo),
        rh + u64x2_extract_lane::<0>(hi) + u64x2_extract_lane::<1>(hi),
    )
}

/// One row of a matrix product, accumulated in 32-bit halves, as
/// [`super::kernels::gemm_row_acc`].
#[inline(always)]
pub(crate) fn gemm_row_acc(a_row: &[u32], b: &[u32], p: usize, lo: &mut [u64], hi: &mut [u64]) {
    lo.fill(0);
    hi.fill(0);
    let mask = u64x2_splat(0xFFFF_FFFF);
    let full = p - p % 4;
    for (kk, &aik) in a_row.iter().enumerate() {
        let brow = &b[kk * p..(kk + 1) * p];
        let av = u32x4_splat(aik);
        for ((bj, l), h) in brow[..full]
            .chunks_exact(4)
            .zip(lo[..full].chunks_exact_mut(4))
            .zip(hi[..full].chunks_exact_mut(4))
        {
            let (pl, ph) = extmul(av, load(bj));
            let l0 = u64x2(l[0], l[1]);
            let l1 = u64x2(l[2], l[3]);
            let h0 = u64x2(h[0], h[1]);
            let h1 = u64x2(h[2], h[3]);
            let l0 = i64x2_add(l0, v128_and(pl, mask));
            let l1 = i64x2_add(l1, v128_and(ph, mask));
            let h0 = i64x2_add(h0, u64x2_shr(pl, 32));
            let h1 = i64x2_add(h1, u64x2_shr(ph, 32));
            l[0] = u64x2_extract_lane::<0>(l0);
            l[1] = u64x2_extract_lane::<1>(l0);
            l[2] = u64x2_extract_lane::<0>(l1);
            l[3] = u64x2_extract_lane::<1>(l1);
            h[0] = u64x2_extract_lane::<0>(h0);
            h[1] = u64x2_extract_lane::<1>(h0);
            h[2] = u64x2_extract_lane::<0>(h1);
            h[3] = u64x2_extract_lane::<1>(h1);
        }
        let a = aik as u64;
        for ((l, h), &bkj) in lo[full..]
            .iter_mut()
            .zip(&mut hi[full..])
            .zip(&brow[full..])
        {
            let t = a * bkj as u64;
            *l += t & 0xFFFF_FFFF;
            *h += t >> 32;
        }
    }
}
