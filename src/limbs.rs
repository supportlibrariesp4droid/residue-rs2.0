//! Minimal fixed-width multi-limb integer helpers (little-endian `u64` limbs).
//!
//! These are deliberately tiny: just what conversions in and out of RNS need,
//! in `const fn` where possible, with no allocation. A base with `N` channels of
//! < 31 bits has `M < 2^(31N)`, so `N` limbs of 64 bits always hold any value
//! in `[0, M)`.

/// `x * m + a`. The caller guarantees the result fits in `N` limbs.
pub(crate) const fn mul_small_add<const N: usize>(mut x: [u64; N], m: u32, a: u32) -> [u64; N] {
    let mut carry = a as u128;
    let mut i = 0;
    while i < N {
        let t = x[i] as u128 * m as u128 + carry;
        x[i] = t as u64;
        carry = t >> 64;
        i += 1;
    }
    debug_assert!(carry == 0, "multi-limb overflow");
    x
}

/// `x * m + a` with 64-bit multiplier and addend; `None` on overflow of `N` limbs.
pub(crate) fn checked_mul_add_u64<const N: usize>(
    x: &[u64; N],
    m: u64,
    a: u64,
) -> Option<[u64; N]> {
    let mut out = [0u64; N];
    let mut carry = a as u128;
    for i in 0..N {
        let t = x[i] as u128 * m as u128 + carry;
        out[i] = t as u64;
        carry = t >> 64;
    }
    if carry == 0 {
        Some(out)
    } else {
        None
    }
}

/// `acc -= x * m`, wrapping modulo `2^(64·acc.len())`. `x` must be at least as long.
pub(crate) fn sub_mul_small(acc: &mut [u64], x: &[u64], m: u64) {
    let mut borrow: u128 = 0;
    for i in 0..acc.len() {
        let prod = x[i] as u128 * m as u128 + borrow;
        let (v, under) = acc[i].overflowing_sub(prod as u64);
        acc[i] = v;
        borrow = (prod >> 64) + under as u128;
    }
}

/// Two's-complement negation modulo `2^(64·x.len())`.
pub(crate) fn neg_wrapping(x: &mut [u64]) {
    let mut carry = 1u64;
    for l in x.iter_mut() {
        let (v, c) = (!*l).overflowing_add(carry);
        *l = v;
        carry = c as u64;
    }
}

/// Number of significant bits of an arbitrary-length limb slice.
pub(crate) fn bit_len_slice(x: &[u64]) -> u64 {
    for i in (0..x.len()).rev() {
        if x[i] != 0 {
            return 64 * i as u64 + (64 - x[i].leading_zeros()) as u64;
        }
    }
    0
}

/// Number of significant bits (0 for zero).
pub(crate) const fn bit_len<const N: usize>(x: &[u64; N]) -> u32 {
    let mut i = N;
    while i > 0 {
        i -= 1;
        if x[i] != 0 {
            return 64 * i as u32 + (64 - x[i].leading_zeros());
        }
    }
    0
}

/// `x mod m` for an arbitrary-length limb slice (Horner from the top limb).
///
/// Two steps of 32 bits per limb, so every division is a native `u64` one
/// rather than a call to the much slower 128-bit division routine.
pub(crate) const fn mod_small(limbs: &[u64], m: u32) -> u32 {
    let m = m as u64;
    let mut r: u64 = 0;
    let mut i = limbs.len();
    while i > 0 {
        i -= 1;
        r = ((r << 32) | (limbs[i] >> 32)) % m;
        r = ((r << 32) | (limbs[i] & 0xFFFF_FFFF)) % m;
    }
    r as u32
}

/// `(x / d, x mod d)` for a non-zero `u64` divisor.
pub(crate) fn divmod_small<const N: usize>(x: &[u64; N], d: u64) -> ([u64; N], u64) {
    let mut q = [0u64; N];
    let mut rem: u128 = 0;
    for i in (0..N).rev() {
        let cur = (rem << 64) | x[i] as u128;
        q[i] = (cur / d as u128) as u64;
        rem = cur % d as u128;
    }
    (q, rem as u64)
}

/// Number of significant limbs (0 for zero).
#[inline]
pub(crate) fn sig_len(x: &[u64]) -> usize {
    (bit_len_slice(x) as usize).div_ceil(64)
}

/// `x += 1`. The caller guarantees no overflow out of the slice.
pub(crate) fn add_one(x: &mut [u64]) {
    for l in x.iter_mut() {
        let (v, c) = l.overflowing_add(1);
        *l = v;
        if !c {
            return;
        }
    }
    debug_assert!(false, "add_one overflow");
}

/// `a -= b` where `a >= b`; `b` may be shorter than `a`.
pub(crate) fn sub_assign(a: &mut [u64], b: &[u64]) {
    let mut borrow = false;
    for (i, ai) in a.iter_mut().enumerate() {
        let bi = b.get(i).copied().unwrap_or(0);
        let (v1, c1) = ai.overflowing_sub(bi);
        let (v2, c2) = v1.overflowing_sub(borrow as u64);
        *ai = v2;
        borrow = c1 || c2;
    }
    debug_assert!(!borrow, "sub_assign underflow");
}

/// Compare two limb slices as unsigned integers (any lengths).
pub(crate) fn cmp(a: &[u64], b: &[u64]) -> core::cmp::Ordering {
    let n = a.len().max(b.len());
    for i in (0..n).rev() {
        let (x, y) = (
            a.get(i).copied().unwrap_or(0),
            b.get(i).copied().unwrap_or(0),
        );
        if x != y {
            return x.cmp(&y);
        }
    }
    core::cmp::Ordering::Equal
}

/// Unsigned division with remainder, `u = q·v + r`, `0 <= r < v`.
///
/// `u` and `v` are given by their significant limbs (`v` non-empty with a
/// non-zero top limb). `q` must have room for `u.len() − v.len() + 1` limbs and
/// `r` for `v.len()`; both must be zeroed by the caller. `un` and `vn` are
/// scratch buffers of at least `u.len() + 1` and `v.len()` limbs.
///
/// Returns `true` if the rare "add back" correction of Algorithm D ran (a test
/// hook: that branch is taken with probability about `2/2^64` on random inputs).
pub(crate) fn divmod(
    u: &[u64],
    v: &[u64],
    q: &mut [u64],
    r: &mut [u64],
    un: &mut [u64],
    vn: &mut [u64],
) -> bool {
    let (m, n) = (u.len(), v.len());
    debug_assert!(
        n >= 1 && v[n - 1] != 0,
        "divisor must be non-zero and trimmed"
    );
    if m < n {
        r[..m].copy_from_slice(u);
        return false;
    }
    if n == 1 {
        let d = v[0] as u128;
        let mut rem: u128 = 0;
        for i in (0..m).rev() {
            let cur = (rem << 64) | u[i] as u128;
            q[i] = (cur / d) as u64;
            rem = cur % d;
        }
        r[0] = rem as u64;
        return false;
    }
    divmod_knuth(u, v, q, r, un, vn)
}

/// Knuth, TAOCP vol. 2, §4.3.1, Algorithm D, with 64-bit digits (`b = 2^64`)
/// and `u128` intermediates. Requires `2 <= n <= m`.
fn divmod_knuth(
    u: &[u64],
    v: &[u64],
    q: &mut [u64],
    r: &mut [u64],
    un: &mut [u64],
    vn: &mut [u64],
) -> bool {
    let (m, n) = (u.len(), v.len());
    const B: u128 = 1 << 64;

    let s = v[n - 1].leading_zeros();
    if s == 0 {
        vn[..n].copy_from_slice(v);
        un[..m].copy_from_slice(u);
        un[m] = 0;
    } else {
        for i in (1..n).rev() {
            vn[i] = (v[i] << s) | (v[i - 1] >> (64 - s));
        }
        vn[0] = v[0] << s;
        un[m] = u[m - 1] >> (64 - s);
        for i in (1..m).rev() {
            un[i] = (u[i] << s) | (u[i - 1] >> (64 - s));
        }
        un[0] = u[0] << s;
    }

    let mut added_back = false;
    for j in (0..=m - n).rev() {
        let num = ((un[j + n] as u128) << 64) | un[j + n - 1] as u128;
        let mut qhat = num / vn[n - 1] as u128;
        let mut rhat = num % vn[n - 1] as u128;
        while qhat >= B || qhat * vn[n - 2] as u128 > ((rhat << 64) | un[j + n - 2] as u128) {
            qhat -= 1;
            rhat += vn[n - 1] as u128;
            if rhat >= B {
                break;
            }
        }

        let mut k: i128 = 0;
        for i in 0..n {
            let p = qhat * vn[i] as u128;
            let t = un[i + j] as i128 - k - (p as u64) as i128;
            un[i + j] = t as u64;
            k = (p >> 64) as i128 - (t >> 64);
        }
        let t = un[j + n] as i128 - k;
        un[j + n] = t as u64;

        q[j] = qhat as u64;
        if t < 0 {
            added_back = true;
            q[j] = q[j].wrapping_sub(1);
            let mut c: u128 = 0;
            for i in 0..n {
                let s2 = un[i + j] as u128 + vn[i] as u128 + c;
                un[i + j] = s2 as u64;
                c = s2 >> 64;
            }
            un[j + n] = un[j + n].wrapping_add(c as u64);
        }
    }

    if s == 0 {
        r[..n].copy_from_slice(&un[..n]);
    } else {
        for i in 0..n {
            r[i] = (un[i] >> s) | (un[i + 1] << (64 - s));
        }
    }
    added_back
}

/// Test hook: `(quotient, remainder, add_back_ran)` for arbitrary limb slices.
#[cfg(feature = "alloc")]
#[doc(hidden)]
#[must_use]
pub fn __divmod_limbs(u: &[u64], v: &[u64]) -> (alloc::vec::Vec<u64>, alloc::vec::Vec<u64>, bool) {
    use alloc::vec;
    let (u, v) = (&u[..sig_len(u)], &v[..sig_len(v)]);
    assert!(!v.is_empty(), "division by zero");
    let mut q = vec![0u64; u.len().max(1)];
    let mut r = vec![0u64; v.len()];
    let mut un = vec![0u64; u.len() + 1];
    let mut vn = vec![0u64; v.len()];
    let ab = divmod(u, v, &mut q, &mut r, &mut un, &mut vn);
    (q, r, ab)
}

/// `Some(value)` if the number fits in a `u128`.
pub(crate) fn to_u128<const N: usize>(x: &[u64; N]) -> Option<u128> {
    if x.iter().skip(2).any(|&l| l != 0) {
        return None;
    }
    let lo = x.first().copied().unwrap_or(0) as u128;
    let hi = x.get(1).copied().unwrap_or(0) as u128;
    Some(lo | (hi << 64))
}

/// Write `x` in decimal. Splits into base-10^19 chunks recursively, so no buffer
/// whose size depends on `N` is needed.
pub(crate) fn fmt_decimal<const N: usize>(
    x: &[u64; N],
    f: &mut core::fmt::Formatter<'_>,
) -> core::fmt::Result {
    const CHUNK: u64 = 10_000_000_000_000_000_000;
    if x.iter().skip(1).all(|&l| l == 0) {
        return write!(f, "{}", x.first().copied().unwrap_or(0));
    }
    let (q, r) = divmod_small(x, CHUNK);
    fmt_decimal(&q, f)?;
    write!(f, "{:019}", r)
}
