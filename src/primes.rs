//! Compile-time primality testing and prime-table generation.
//!
//! All functions here are `const fn`, so a base's moduli are chosen at compile
//! time. For moduli below `2^31` the deterministic Miller-Rabin witnesses
//! `{2, 3, 5, 7}` are sufficient (the first strong pseudoprime to all of them is
//! 3 215 031 751 > 2^31), so primality testing is exact, not probabilistic.
#![allow(clippy::manual_is_multiple_of)]

/// `(a * b) mod m` for 32-bit operands, via a 64-bit intermediate.
const fn mulmod(a: u32, b: u32, m: u32) -> u32 {
    ((a as u64 * b as u64) % m as u64) as u32
}

/// `(base ^ exp) mod m`, binary exponentiation.
const fn powmod(mut base: u32, mut exp: u32, m: u32) -> u32 {
    let mut acc: u32 = 1;
    base %= m;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = mulmod(acc, base, m);
        }
        base = mulmod(base, base, m);
        exp >>= 1;
    }
    acc
}

/// Deterministic Miller-Rabin primality test, exact for all `n < 2^31`.
#[must_use]
pub const fn is_prime(n: u32) -> bool {
    if n < 2 {
        return false;
    }
    let small = [2u32, 3, 5, 7];
    let mut i = 0;
    while i < 4 {
        if n == small[i] {
            return true;
        }
        if n % small[i] == 0 {
            return false;
        }
        i += 1;
    }
    let mut d = n - 1;
    let mut s = 0u32;
    while d & 1 == 0 {
        d >>= 1;
        s += 1;
    }
    let bases = [2u32, 3, 5, 7];
    let mut bi = 0;
    while bi < 4 {
        let a = bases[bi];
        let mut x = powmod(a, d, n);
        if x != 1 && x != n - 1 {
            let mut r = 1u32;
            let mut composite = true;
            while r < s {
                x = mulmod(x, x, n);
                if x == n - 1 {
                    composite = false;
                    break;
                }
                r += 1;
            }
            if composite {
                return false;
            }
        }
        bi += 1;
    }
    true
}

/// The largest prime strictly below `start | 1` (i.e. at or below an odd start),
/// searching downward. Returns 0 only if none exists (never for sane inputs).
#[must_use]
pub const fn prime_below(start: u32) -> u32 {
    let mut cand = if start & 1 == 1 { start } else { start - 1 };
    loop {
        if is_prime(cand) {
            return cand;
        }
        if cand <= 2 {
            return 0;
        }
        cand -= 2;
    }
}

/// The `N` largest primes strictly below `2^31`, in descending order.
///
/// The first element is `2^31 - 1 = 2_147_483_647`, which is itself prime
/// (the Mersenne prime M31).
#[must_use]
pub const fn largest_primes<const N: usize>() -> [u32; N] {
    let mut out = [0u32; N];
    let mut count = 0;
    let mut cand: u32 = (1 << 31) - 1;
    while count < N {
        if is_prime(cand) {
            out[count] = cand;
            count += 1;
        }
        if cand < 2 {
            break;
        }
        cand -= 2;
    }
    out
}

/// The `N` largest primes below `2^31` after skipping the `skip` largest, in
/// descending order. `largest_primes_after::<N>(k)` and `largest_primes::<k>()`
/// share no prime, which is what two bases of RNS Montgomery multiplication
/// need (see [`crate::MontRns`]).
#[must_use]
pub const fn largest_primes_after<const N: usize>(skip: usize) -> [u32; N] {
    let mut out = [0u32; N];
    let mut seen = 0;
    let mut count = 0;
    let mut cand: u32 = (1 << 31) - 1;
    while count < N {
        if is_prime(cand) {
            if seen >= skip {
                out[count] = cand;
                count += 1;
            }
            seen += 1;
        }
        if cand < 2 {
            break;
        }
        cand -= 2;
    }
    out
}
