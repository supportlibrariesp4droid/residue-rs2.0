//! A prime channel below `2^50` in double precision, with fused multiply-add.
//!
//! The product `a·b` of two residues has about 100 bits, more than an `f64`
//! holds, but `h = a·b` (rounded) and `l = fma(a, b, −h)` represent it exactly
//! as `h + l`. The quotient `q = ⌊h/p⌋` is estimated with a precomputed `1/p`
//! and is off by at most one, so `fma(−q, p, h) + l = ab − qp` is an exact
//! integer in `[−p, 2p)`, brought into `[0, p)` by one correction. This is the
//! method NTL and FLINT use for word-size moduli in floating point; it
//! vectorizes on any instruction set with FMA (AVX2, AVX-512, NEON, WASM
//! relaxed SIMD), unlike 64×64→128 integer products.

#[derive(Clone, Copy, Debug)]
pub struct Ch50 {
    p: f64,
    pinv: f64,
}

impl Ch50 {
    pub fn new(p: u64) -> Self {
        assert!(p < 1 << 50);
        Ch50 {
            p: p as f64,
            pinv: 1.0 / p as f64,
        }
    }

    pub fn p_u64(&self) -> u64 {
        self.p as u64
    }

    #[inline(always)]
    pub fn mul(&self, a: f64, b: f64) -> f64 {
        let h = a * b;
        let l = a.mul_add(b, -h);
        let q = (h * self.pinv).floor();
        let r = (-q).mul_add(self.p, h) + l;
        let r = if r >= self.p { r - self.p } else { r };
        if r < 0.0 {
            r + self.p
        } else {
            r
        }
    }

    #[inline(always)]
    pub fn add(&self, a: f64, b: f64) -> f64 {
        let s = a + b;
        if s >= self.p {
            s - self.p
        } else {
            s
        }
    }

    #[inline(always)]
    pub fn sub(&self, a: f64, b: f64) -> f64 {
        let d = a - b;
        if d < 0.0 {
            d + self.p
        } else {
            d
        }
    }

    #[inline(always)]
    pub fn neg(&self, a: f64) -> f64 {
        if a == 0.0 {
            0.0
        } else {
            self.p - a
        }
    }

    pub fn inv(&self, a: f64) -> f64 {
        let mut e = self.p_u64() - 2;
        let mut acc = 1.0;
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
}
