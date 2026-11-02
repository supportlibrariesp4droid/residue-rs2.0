//! A 62-bit prime channel in Montgomery form with `R = 2^64`.
//!
//! Residues are below `p < 2^62`, so `a + b` never overflows a `u64` and the
//! branchless tricks of the 31-bit channel carry over. Products are
//! `u64 × u64 → u128`, which x86-64 does in one instruction but no vector
//! instruction set offers, so these kernels are scalar.

#[derive(Clone, Copy, Debug)]
pub struct Ch62 {
    pub p: u64,
    /// `−p⁻¹ mod 2^64`.
    ninv: u64,
    /// `R² mod p`.
    r2: u64,
}

impl Ch62 {
    pub fn new(p: u64) -> Self {
        assert!(p & 1 == 1 && p < 1 << 62);
        let mut x = p;
        for _ in 0..6 {
            x = x.wrapping_mul(2u64.wrapping_sub(p.wrapping_mul(x)));
        }
        let r = ((1u128 << 64) % p as u128) as u64;
        let r2 = ((r as u128 * r as u128) % p as u128) as u64;
        Ch62 {
            p,
            ninv: x.wrapping_neg(),
            r2,
        }
    }

    /// `t·R⁻¹ mod p` for `t < p·2^64`.
    #[inline(always)]
    pub fn redc(&self, t: u128) -> u64 {
        let m = (t as u64).wrapping_mul(self.ninv);
        let u = ((t + m as u128 * self.p as u128) >> 64) as u64;
        if u >= self.p {
            u - self.p
        } else {
            u
        }
    }

    #[inline(always)]
    pub fn mul(&self, a: u64, b: u64) -> u64 {
        self.redc(a as u128 * b as u128)
    }

    #[inline(always)]
    pub fn to_mont(&self, a: u64) -> u64 {
        self.mul(a, self.r2)
    }

    #[inline(always)]
    pub fn from_mont(&self, a: u64) -> u64 {
        self.redc(a as u128)
    }

    #[inline(always)]
    pub fn add(&self, a: u64, b: u64) -> u64 {
        let s = a + b;
        let t = s.wrapping_sub(self.p);
        if s < t {
            s
        } else {
            t
        }
    }

    #[inline(always)]
    pub fn sub(&self, a: u64, b: u64) -> u64 {
        let d = a.wrapping_sub(b);
        let d2 = d.wrapping_add(self.p);
        if d < d2 {
            d
        } else {
            d2
        }
    }

    #[inline(always)]
    pub fn neg(&self, a: u64) -> u64 {
        if a == 0 {
            0
        } else {
            self.p - a
        }
    }

    pub fn inv(&self, a: u64) -> u64 {
        let mut e = self.p - 2;
        let mut acc = self.to_mont(1);
        let mut b = a;
        while e > 0 {
            if e & 1 == 1 {
                acc = self.mul(acc, b);
            }
            b = self.mul(b, b);
            e >>= 1;
        }
        acc
    }

    /// Reduce an unreduced sum of products `top·2^128 + acc`, where every term
    /// was a product of two Montgomery residues (`ab·R²`), to the Montgomery
    /// residue of the sum: `T·R⁻¹ = top·R + hi + lo·R⁻¹`. Requires `top < p`.
    #[inline(always)]
    pub fn reduce_sum(&self, acc: u128, top: u64) -> u64 {
        let lo = self.redc(acc as u64 as u128);
        let hi = self.to_mont(self.redc(acc >> 64));
        let tp = self.to_mont(top);
        self.add(self.add(lo, hi), tp)
    }
}

/// Deterministic Miller-Rabin for `u64` (these 12 witnesses are exact for all
/// 64-bit integers).
pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for p in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n % p == 0 {
            return n == p;
        }
    }
    let mulmod = |a: u64, b: u64| (a as u128 * b as u128 % n as u128) as u64;
    let (mut d, mut s) = (n - 1, 0);
    while d % 2 == 0 {
        d /= 2;
        s += 1;
    }
    'witness: for a in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        let mut x = 1u64;
        let (mut b, mut e) = (a, d);
        while e > 0 {
            if e & 1 == 1 {
                x = mulmod(x, b);
            }
            b = mulmod(b, b);
            e >>= 1;
        }
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 1..s {
            x = mulmod(x, x);
            if x == n - 1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

/// The `count` largest primes below `limit`, in decreasing order.
pub fn largest_primes_below(limit: u64, count: usize) -> Vec<u64> {
    let mut out = Vec::with_capacity(count);
    let mut n = (limit - 1) | 1;
    while out.len() < count {
        if is_prime(n) {
            out.push(n);
        }
        n -= 2;
    }
    out
}
