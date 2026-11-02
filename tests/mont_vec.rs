//! Batch RNS Montgomery arithmetic (`ModVec`) against num-bigint, with every
//! kernel (explicit AVX-512, AVX2, portable) and, with the `parallel`
//! feature, the parallel path forced on.
#![cfg(feature = "alloc")]
#![allow(long_running_const_eval)]

use num_bigint::BigUint;
use num_traits::One;
use proptest::prelude::*;
use residue::{rns_base, ModVec, MontRns, MontRnsError};

fn big(limbs: &[u64]) -> BigUint {
    let bytes: Vec<u8> = limbs.iter().flat_map(|l| l.to_le_bytes()).collect();
    BigUint::from_bytes_le(&bytes)
}

fn modulus(bits: u32) -> impl Strategy<Value = BigUint> {
    let limbs = (bits as usize).div_ceil(64);
    prop::collection::vec(any::<u64>(), limbs).prop_map(move |l| {
        let mut x = big(&l) & ((BigUint::one() << bits) - 1u8);
        x |= BigUint::one() << (bits - 1);
        x |= BigUint::one();
        x
    })
}

fn value(bits: u32) -> impl Strategy<Value = BigUint> {
    let limbs = (bits as usize).div_ceil(64);
    prop::collection::vec(any::<u64>(), limbs)
        .prop_map(move |l| big(&l) & ((BigUint::one() << bits) - 1u8))
}

fn levels() -> Vec<u8> {
    if cfg!(all(feature = "std", target_arch = "x86_64")) {
        vec![0, 1, 2]
    } else {
        vec![0]
    }
}

fn set_level(_l: u8) {
    #[cfg(all(feature = "std", target_arch = "x86_64"))]
    residue::__set_simd_level(_l);
}

fn force_parallel() {
    #[cfg(feature = "parallel")]
    residue::__set_parallel_threshold(0);
}

/// Limbs of the canonical value of element `k` in a flat `to_limbs` output.
fn elem(limbs: &[u64], w: usize, k: usize) -> BigUint {
    big(&limbs[k * w..(k + 1) * w])
}

macro_rules! vec_tests {
    ($modname:ident, $b1:ident, $n1:literal, $b2:ident, $n2:literal, $maxbits:expr) => {
        mod $modname {
            use super::*;
            type Ctx = MontRns<$b1, $n1, $b2, $n2>;

            proptest! {
                #![proptest_config(ProptestConfig::with_cases(16))]

                #[test]
                fn batch_matches_oracle(
                    m in (2u32..=$maxbits).prop_flat_map(modulus),
                    xs in prop::collection::vec((value($maxbits + 20), value($maxbits + 20)), 0..600),
                    e in value(130),
                ) {
                    force_parallel();
                    let ml = m.to_u64_digits();
                    let ctx = match Ctx::new(&ml) {
                        Ok(c) => c,
                        Err(MontRnsError::SharesPrime(_)) => return Ok(()),
                        Err(err) => panic!("{err:?}"),
                    };
                    let w = ml.len();
                    let width = (($maxbits + 20) as usize).div_ceil(64);
                    let flat = |v: &Vec<BigUint>| -> Vec<u64> {
                        v.iter().flat_map(|x| { let mut d = x.to_u64_digits(); d.resize(width, 0); d }).collect()
                    };
                    let (a, b): (Vec<BigUint>, Vec<BigUint>) = xs.into_iter().unzip();
                    let va = ModVec::from_limbs(&ctx, width, &flat(&a));
                    let vb = ModVec::from_limbs(&ctx, width, &flat(&b));
                    for level in levels() {
                        set_level(level);
                        let prod = (&va * &vb).to_limbs();
                        let sum = (&va + &vb).to_limbs();
                        let diff = (&va - &vb).to_limbs();
                        let sq = va.square().to_limbs();
                        let pw = va.pow(&e.to_u64_digits()).to_limbs();
                        for k in 0..a.len() {
                            let (x, y) = (&a[k] % &m, &b[k] % &m);
                            prop_assert_eq!(elem(&prod, w, k), (&x * &y) % &m, "level {} product {}", level, k);
                            prop_assert_eq!(elem(&sum, w, k), (&x + &y) % &m, "level {} sum {}", level, k);
                            prop_assert_eq!(elem(&diff, w, k), (&x + &m - &y) % &m, "level {} difference {}", level, k);
                            prop_assert_eq!(elem(&sq, w, k), (&x * &x) % &m, "level {} square {}", level, k);
                            prop_assert_eq!(elem(&pw, w, k), x.modpow(&e, &m), "level {} power {}", level, k);
                        }
                    }
                    set_level(0);
                }
            }
        }
    };
}

rns_base!(S1: largest 6);
rns_base!(S2: largest 6 after 6);
rns_base!(U1: largest 7);
rns_base!(U2: largest 9 after 7);
rns_base!(L1: largest 20);
rns_base!(L2: largest 20 after 20);

vec_tests!(six_six, S1, 6, S2, 6, 170);
vec_tests!(seven_nine, U1, 7, U2, 9, 190);
vec_tests!(twenty, L1, 20, L2, 20, 600);

rns_base!(H1: largest 70);
rns_base!(H2: largest 70 after 70);

/// A 2048-bit modulus and 300 elements: batch against scalar elements.
#[test]
fn rsa_size_batch_matches_scalar() {
    force_parallel();
    let mut s = 0x0bad_5eed_1234_5678u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut ml: Vec<u64> = (0..32).map(|_| next()).collect();
    ml[0] |= 1;
    ml[31] |= 1 << 63;
    let ctx = MontRns::<H1, 70, H2, 70>::new(&ml).unwrap();
    let xs: Vec<_> = (0..300)
        .map(|_| {
            let l: Vec<u64> = (0..32).map(|_| next()).collect();
            ctx.from_limbs(&l)
        })
        .collect();
    let v = ModVec::from_elems(&ctx, &xs);
    let e = [next(), next()];
    for level in levels() {
        set_level(level);
        let p = v.pow(&e);
        for (k, x) in xs.iter().enumerate() {
            assert_eq!(
                p.get(k).unwrap().to_limbs(),
                x.pow(&e).to_limbs(),
                "level {level}, element {k}"
            );
        }
    }
    set_level(0);
}

/// Factors through additions and normalization, in batch.
#[test]
fn batch_factor_growth() {
    let m = [u64::MAX, u64::MAX >> 1];
    let ctx = MontRns::<S1, 6, S2, 6>::new(&m).unwrap();
    let xs: Vec<_> = (1..=40u128).map(|v| ctx.from_u128(v)).collect();
    let mut v = ModVec::from_elems(&ctx, &xs);
    let p = (1u128 << 127) - 1;
    for _ in 0..200 {
        v = &v + &v;
    }
    let two200 = BigUint::from(2u8).pow(200) % BigUint::from(p);
    for k in 0..40 {
        let want = (BigUint::from(k as u128 + 1) * &two200) % BigUint::from(p);
        assert_eq!(big(&v.get(k).unwrap().to_limbs()), want);
    }
    assert!(v.factor() < 1 << 62);
}

rns_base!(X1: largest 134);
rns_base!(X2: largest 134 after 134);

/// A 4096-bit modulus, the largest size the roadmap asks for: batch and scalar
/// exponentiation against num-bigint.
#[test]
fn rsa_4096_matches_num_bigint() {
    let mut s = 0x4096_4096_dead_beefu64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut ml: Vec<u64> = (0..64).map(|_| next()).collect();
    ml[0] |= 1;
    ml[63] |= 1 << 63;
    let m = big(&ml);
    let ctx = MontRns::<X1, 134, X2, 134>::new(&ml).unwrap();
    let raw: Vec<Vec<u64>> = (0..20).map(|_| (0..64).map(|_| next()).collect()).collect();
    let xs: Vec<_> = raw.iter().map(|l| ctx.from_limbs(l)).collect();
    let e = [next(), next()];
    let eb = big(&e);
    let v = ModVec::from_elems(&ctx, &xs);
    for level in levels() {
        set_level(level);
        let p = v.pow(&e);
        for (k, l) in raw.iter().enumerate() {
            let want = (big(l) % &m).modpow(&eb, &m);
            assert_eq!(
                big(&p.get(k).unwrap().to_limbs()),
                want,
                "level {level}, element {k}"
            );
        }
    }
    set_level(0);
    assert_eq!(
        big(&xs[0].pow(&e).to_limbs()),
        (big(&raw[0]) % &m).modpow(&eb, &m)
    );
}
