//! RNS Montgomery arithmetic modulo a large odd number, against num-bigint.
#![cfg(feature = "alloc")]

use num_bigint::BigUint;
use num_traits::{One, Zero};
use proptest::prelude::*;
use residue::{rns_base, MontRns, MontRnsError};

fn big(limbs: &[u64]) -> BigUint {
    let bytes: Vec<u8> = limbs.iter().flat_map(|l| l.to_le_bytes()).collect();
    BigUint::from_bytes_le(&bytes)
}

fn limbs_of(x: &BigUint) -> Vec<u64> {
    x.to_u64_digits()
}

/// An odd modulus of exactly `bits` bits (at least 2).
fn modulus(bits: u32) -> impl Strategy<Value = BigUint> {
    let limbs = (bits as usize).div_ceil(64);
    prop::collection::vec(any::<u64>(), limbs).prop_map(move |l| {
        let mut x = big(&l) & ((BigUint::one() << bits) - 1u8);
        x |= BigUint::one() << (bits - 1);
        x |= BigUint::one();
        x
    })
}

/// A value below `2^bits` (any size relative to the modulus).
fn value(bits: u32) -> impl Strategy<Value = BigUint> {
    let limbs = (bits as usize).div_ceil(64) + 1;
    (prop::collection::vec(any::<u64>(), limbs), 0..=bits)
        .prop_map(|(l, b)| big(&l) & ((BigUint::one() << b) - 1u8))
}

#[derive(Debug, Clone)]
enum Op {
    Mul(usize, usize),
    Add(usize, usize),
    Sub(usize, usize),
    Neg(usize),
    Square(usize),
}

fn ops(n: usize) -> impl Strategy<Value = Vec<Op>> {
    let op = prop_oneof![
        3 => (0..n, 0..n).prop_map(|(a, b)| Op::Mul(a, b)),
        2 => (0..n, 0..n).prop_map(|(a, b)| Op::Add(a, b)),
        2 => (0..n, 0..n).prop_map(|(a, b)| Op::Sub(a, b)),
        1 => (0..n).prop_map(Op::Neg),
        1 => (0..n).prop_map(Op::Square),
    ];
    prop::collection::vec(op, 1..40)
}

macro_rules! mont_tests {
    ($modname:ident, $b1:ident, $n1:literal, $b2:ident, $n2:literal, $maxbits:expr) => {
        mod $modname {
            use super::*;
            type Ctx = MontRns<$b1, $n1, $b2, $n2>;

            fn ctx(m: &BigUint) -> Option<Ctx> {
                match Ctx::new(&limbs_of(m)) {
                    Ok(c) => Some(c),
                    Err(MontRnsError::SharesPrime(_)) => None,
                    Err(e) => panic!("unexpected {e:?} for {m}"),
                }
            }

            proptest! {
                #![proptest_config(ProptestConfig::with_cases(48))]

                #[test]
                fn ring_operations(
                    m in (2u32..=$maxbits).prop_flat_map(modulus),
                    a in value($maxbits + 70),
                    b in value($maxbits + 70),
                ) {
                    let Some(ctx) = ctx(&m) else { return Ok(()); };
                    let (ea, eb) = (ctx.from_limbs(&limbs_of(&a)), ctx.from_limbs(&limbs_of(&b)));
                    let (ra, rb) = (&a % &m, &b % &m);
                    prop_assert_eq!(big(&ea.to_limbs()), ra.clone());
                    prop_assert_eq!(big(&(ea * eb).to_limbs()), (&ra * &rb) % &m);
                    prop_assert_eq!(big(&(ea + eb).to_limbs()), (&ra + &rb) % &m);
                    prop_assert_eq!(big(&(ea - eb).to_limbs()), (&ra + &m - &rb) % &m);
                    prop_assert_eq!(big(&(-ea).to_limbs()), (&m - &ra) % &m);
                    prop_assert_eq!(big(&ea.square().to_limbs()), (&ra * &ra) % &m);
                    prop_assert_eq!(big(&ctx.one().to_limbs()), BigUint::one() % &m);
                    prop_assert!(ctx.zero().is_zero());
                    prop_assert_eq!(ea.to_limbs().len(), limbs_of(&m).len());
                }

                #[test]
                fn power(
                    m in (2u32..=$maxbits).prop_flat_map(modulus),
                    a in value($maxbits),
                    e in value(300),
                ) {
                    let Some(ctx) = ctx(&m) else { return Ok(()); };
                    let x = ctx.from_limbs(&limbs_of(&a));
                    let got = big(&x.pow(&limbs_of(&e)).to_limbs());
                    prop_assert_eq!(got, (&a % &m).modpow(&e, &m));
                }

                /// Random chains: factors grow through additions and shrink
                /// through products; every intermediate result must stay exact.
                #[test]
                fn operation_chains(
                    m in (2u32..=$maxbits).prop_flat_map(modulus),
                    start in prop::collection::vec(value($maxbits), 4),
                    program in ops(4),
                ) {
                    let Some(ctx) = ctx(&m) else { return Ok(()); };
                    let mut el: Vec<_> = start.iter().map(|x| ctx.from_limbs(&limbs_of(x))).collect();
                    let mut or: Vec<BigUint> = start.iter().map(|x| x % &m).collect();
                    for (step, op) in program.iter().enumerate() {
                        let (e, o) = match *op {
                            Op::Mul(i, j) => (el[i] * el[j], (&or[i] * &or[j]) % &m),
                            Op::Add(i, j) => (el[i] + el[j], (&or[i] + &or[j]) % &m),
                            Op::Sub(i, j) => (el[i] - el[j], (&or[i] + &m - &or[j]) % &m),
                            Op::Neg(i) => (-el[i], (&m - &or[i]) % &m),
                            Op::Square(i) => (el[i].square(), (&or[i] * &or[i]) % &m),
                        };
                        prop_assert_eq!(big(&e.to_limbs()), o.clone(), "step {}: {:?}", step, op);
                        el[step % 4] = e;
                        or[step % 4] = o;
                    }
                }
            }
        }
    };
}

rns_base!(S1: largest 6);
rns_base!(S2: largest 6 after 6);
rns_base!(U1: largest 7);
rns_base!(U2: largest 9 after 7);
rns_base!(L1: largest 12);
rns_base!(L2: largest 12 after 12);

mont_tests!(six_six, S1, 6, S2, 6, 170);
mont_tests!(seven_nine, U1, 7, U2, 9, 190);
mont_tests!(twelve, L1, 12, L2, 12, 350);

rns_base!(H1: largest 70);
rns_base!(H2: largest 70 after 70);

/// A 2048-bit modulus, the size of an RSA key.
#[test]
fn rsa_size_modulus() {
    let mut s = 0x1234_5678_9abc_def1u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut ml: Vec<u64> = (0..32).map(|_| next()).collect();
    ml[0] |= 1;
    ml[31] |= 1 << 63;
    let m = big(&ml);
    let ctx = MontRns::<H1, 70, H2, 70>::new(&ml).unwrap();
    for _ in 0..4 {
        let a: Vec<u64> = (0..32).map(|_| next()).collect();
        let b: Vec<u64> = (0..32).map(|_| next()).collect();
        let e: Vec<u64> = (0..2).map(|_| next()).collect();
        let (x, y) = (ctx.from_limbs(&a), ctx.from_limbs(&b));
        assert_eq!(big(&(x * y).to_limbs()), (big(&a) * big(&b)) % &m);
        assert_eq!(big(&x.pow(&e).to_limbs()), big(&a).modpow(&big(&e), &m));
    }
}

/// Inverses modulo known primes.
#[test]
fn inverse_modulo_primes() {
    let p25519 = (BigUint::one() << 255u32) - 19u32;
    let p256 = BigUint::parse_bytes(
        b"ffffffff00000001000000000000000000000000ffffffffffffffffffffffff",
        16,
    )
    .unwrap();
    let m127 = (BigUint::one() << 127u32) - 1u32;
    for p in [
        p25519,
        p256,
        m127,
        BigUint::from(1_000_000_007u32),
        BigUint::from(3u32),
    ] {
        let ctx = MontRns::<L1, 12, L2, 12>::new(&limbs_of(&p)).unwrap();
        for v in [1u32, 2, 3, 12345, 987_654_321] {
            let x = ctx.from_u128(v as u128);
            if (BigUint::from(v) % &p).is_zero() {
                assert!(x.inv_prime().is_zero());
                continue;
            }
            let inv = x.inv_prime();
            assert_eq!(
                big(&(x * inv).to_limbs()),
                BigUint::one(),
                "p = {p}, v = {v}"
            );
        }
    }
}

/// Long chains of additions without products grow the factor; products and
/// normalize bring it back, and results stay exact throughout.
#[test]
fn factors_grow_and_shrink() {
    let m = (BigUint::one() << 127u32) - 1u32;
    let ctx = MontRns::<S1, 6, S2, 6>::new(&limbs_of(&m)).unwrap();
    let x = ctx.from_u128(123_456_789);
    let mut acc = x;
    for k in 2..=1000u32 {
        acc = acc + x;
        assert_eq!(
            acc.to_u128(),
            Some(123_456_789u128 * k as u128 % ((1u128 << 127) - 1))
        );
    }
    assert!(acc.factor() > 1000);
    let n = acc.normalize();
    assert!(n.factor() <= 6 + 2);
    assert_eq!(n, acc);
    let sq = acc * acc;
    assert!(sq.factor() <= 6 + 2);

    let mut d = x;
    let mut expect = 123_456_789u128;
    let p = (1u128 << 127) - 1;
    let mut largest = 0;
    for _ in 0..200 {
        d = d + d;
        expect = (expect * 2) % p;
        largest = largest.max(d.factor());
        assert_eq!(d.to_u128(), Some(expect));
    }
    assert!(largest < 1 << 62);
    let back = d - d - d;
    assert_eq!(back.to_u128(), Some((p - expect) % p));
}

#[test]
fn construction_errors() {
    assert_eq!(
        MontRns::<S1, 6, S2, 6>::new(&[10]).err(),
        Some(MontRnsError::InvalidModulus)
    );
    assert_eq!(
        MontRns::<S1, 6, S2, 6>::new(&[1]).err(),
        Some(MontRnsError::InvalidModulus)
    );
    assert_eq!(
        MontRns::<S1, 6, S2, 6>::new(&[]).err(),
        Some(MontRnsError::InvalidModulus)
    );
    let p = 2_147_483_647u64;
    assert_eq!(
        MontRns::<S1, 6, S2, 6>::new(&[p * 3]).err(),
        Some(MontRnsError::SharesPrime(p as u32))
    );
    assert_eq!(
        MontRns::<S1, 6, S1, 6>::new(&[1_000_003]).err(),
        Some(MontRnsError::BasesOverlap(p as u32))
    );
    let huge: Vec<u64> = vec![u64::MAX; 3];
    assert!(matches!(
        MontRns::<S1, 6, S2, 6>::new(&huge).err(),
        Some(MontRnsError::BaseTooSmall { .. })
    ));
}

#[test]
#[should_panic(expected = "different MontRns contexts")]
fn mixing_contexts_panics() {
    let c1 = MontRns::<S1, 6, S2, 6>::new(&[1_000_003]).unwrap();
    let c2 = MontRns::<S1, 6, S2, 6>::new(&[1_000_033]).unwrap();
    let _ = c1.one() * c2.one();
}
