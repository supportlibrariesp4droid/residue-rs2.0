//! Division: Knuth's Algorithm D on limbs, truncated and euclidean division on
//! `Rns`, and exact division. Oracle: num-bigint.

use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{One, Signed, Zero};
use proptest::prelude::*;
use residue::{rns_base, Rns, RnsBase};

fn big(limbs: &[u64]) -> BigUint {
    let bytes: Vec<u8> = limbs.iter().flat_map(|l| l.to_le_bytes()).collect();
    BigUint::from_bytes_le(&bytes)
}

/// Limbs that stress Algorithm D: boundary digits trigger the q̂ corrections.
fn limb() -> impl Strategy<Value = u64> {
    prop_oneof![
        3 => any::<u64>(),
        1 => Just(0u64),
        1 => Just(1u64),
        1 => Just(u64::MAX),
        1 => Just(u64::MAX - 1),
        1 => Just(1u64 << 63),
        1 => Just((1u64 << 63) - 1),
    ]
}

#[cfg(feature = "alloc")]
proptest! {
    #![proptest_config(ProptestConfig::with_cases(4000))]

    #[test]
    fn knuth_matches_bigint(
        u in prop::collection::vec(limb(), 0..9),
        v in prop::collection::vec(limb(), 1..7),
    ) {
        let (bu, bv) = (big(&u), big(&v));
        prop_assume!(!bv.is_zero());
        let (q, r, _) = residue::__divmod_limbs(&u, &v);
        prop_assert_eq!(big(&q), &bu / &bv);
        prop_assert_eq!(big(&r), &bu % &bv);
    }
}

/// The "add back" branch (probability ~2^-63 on random inputs) must be reached
/// and must be correct. Structured digits make it frequent.
#[cfg(feature = "alloc")]
#[test]
fn knuth_add_back_branch_is_exercised() {
    let digits = [
        0u64,
        1,
        2,
        u64::MAX,
        u64::MAX - 1,
        1 << 63,
        (1 << 63) - 1,
        (1 << 63) + 1,
    ];
    let mut hits = 0;
    let mut total = 0;
    for &a in &digits {
        for &b in &digits {
            for &c in &digits {
                for &d in &digits {
                    for &e in &digits {
                        let u = [a, b, c, d];
                        let v = [e, c, 1u64 << 63 | d >> 1];
                        let (bu, bv) = (big(&u), big(&v));
                        let (q, r, ab) = residue::__divmod_limbs(&u, &v);
                        assert_eq!(big(&q), &bu / &bv, "q for u={u:?} v={v:?}");
                        assert_eq!(big(&r), &bu % &bv, "r for u={u:?} v={v:?}");
                        hits += ab as usize;
                        total += 1;
                    }
                }
            }
        }
    }
    println!("add-back taken in {hits} of {total} structured cases");
    assert!(hits > 0, "the add-back branch was never exercised");
}

fn to_big<B: RnsBase<N>, const N: usize>(x: &Rns<B, N>) -> BigInt {
    let (neg, mag) = x.to_sign_magnitude();
    let m = BigInt::from(big(&mag));
    if neg {
        -m
    } else {
        m
    }
}

fn from_big<B: RnsBase<N>, const N: usize>(x: &BigInt) -> Rns<B, N> {
    Rns::from_sign_magnitude(x.sign() == Sign::Minus, &x.magnitude().to_u64_digits()).unwrap()
}

/// Signed values with bit length uniform in `[0, bits]`.
fn signed(bits: u32) -> impl Strategy<Value = BigInt> {
    let n = (bits as usize).div_ceil(64) + 1;
    (0..=bits, any::<bool>(), prop::collection::vec(limb(), n)).prop_map(|(bl, neg, noise)| {
        if bl == 0 {
            return BigInt::zero();
        }
        let mask = (BigUint::one() << bl) - 1u8;
        let v = BigInt::from((big(&noise) & mask) | (BigUint::one() << (bl - 1)));
        if neg {
            -v
        } else {
            v
        }
    })
}

/// Euclidean oracle: 0 <= r < |y|.
fn euclid(x: &BigInt, y: &BigInt) -> (BigInt, BigInt) {
    let ay = y.abs();
    let r = ((x % &ay) + &ay) % &ay;
    let q = (x - &r) / y;
    (q, r)
}

macro_rules! division_tests {
    ($modname:ident, $base:ident, $n:literal) => {
        mod $modname {
            use super::*;
            type Z = Rns<$base, $n>;
            fn z(x: &BigInt) -> Z { from_big::<$base, $n>(x) }
            fn maxb() -> u32 { Z::max_bits() }

            proptest! {
                #![proptest_config(ProptestConfig::with_cases(400))]

                #[test]
                fn truncated_div_rem(a in signed(maxb()), b in signed(maxb())) {
                    prop_assume!(!b.is_zero());
                    let (q, r) = z(&a).checked_div_rem(z(&b)).unwrap();
                    prop_assert_eq!(to_big(&q), &a / &b);
                    prop_assert_eq!(to_big(&r), &a % &b);
                    prop_assert_eq!(to_big(&(z(&a) / z(&b))), &a / &b);
                    prop_assert_eq!(to_big(&(z(&a) % z(&b))), &a % &b);
                }

                #[test]
                fn euclidean_div_rem(a in signed(maxb()), b in signed(maxb())) {
                    prop_assume!(!b.is_zero());
                    let (q, r) = z(&a).checked_div_rem_euclid(z(&b)).unwrap();
                    let (eq, er) = euclid(&a, &b);
                    prop_assert_eq!(to_big(&q), eq);
                    prop_assert_eq!(to_big(&r), er);
                }

                #[test]
                fn exact_division_of_products(q in signed(maxb() / 2), y in signed(maxb() / 2)) {
                    prop_assume!(!y.is_zero());
                    let x = &q * &y;
                    prop_assert_eq!(z(&x).div_exact(z(&y)).map(|v| to_big(&v)), Some(q));
                }

                #[test]
                fn exact_division_detects_non_divisors(a in signed(maxb()), b in signed(maxb() / 3)) {
                    prop_assume!(!b.is_zero());
                    let divisible = (&a % &b).is_zero();
                    match z(&a).div_exact(z(&b)) {
                        Some(q) => {
                            prop_assert!(divisible, "claimed {} / {} exact", a, b);
                            prop_assert_eq!(to_big(&q), &a / &b);
                        }
                        None => prop_assert!(!divisible, "missed exact {} / {}", a, b),
                    }
                }

                /// Off-by-one numerators: x = q·y ± 1 is never divisible (|y| > 1).
                #[test]
                fn exact_division_off_by_one(q in signed(maxb() / 2), y in signed(maxb() / 2), d in prop_oneof![Just(-1i32), Just(1)]) {
                    prop_assume!(y.abs() > BigInt::one());
                    let x = &q * &y + d;
                    prop_assume!(x.magnitude().bits() <= maxb() as u64);
                    prop_assert!(z(&x).div_exact(z(&y)).is_none());
                }
            }

            #[test]
            fn edges() {
                let one = BigInt::one();
                let top: BigInt = (BigInt::one() << maxb()) - 1;
                for (a, b) in [
                    (top.clone(), one.clone()),
                    (-top.clone(), one.clone()),
                    (top.clone(), -one.clone()),
                    (top.clone(), top.clone()),
                    (-top.clone(), top.clone()),
                    (BigInt::zero(), top.clone()),
                    (one.clone(), top.clone()),
                    (-one.clone(), top.clone()),
                ] {
                    let (q, r) = z(&a).checked_div_rem(z(&b)).unwrap();
                    assert_eq!((to_big(&q), to_big(&r)), (&a / &b, &a % &b), "{a} / {b}");
                    let (eq, er) = euclid(&a, &b);
                    assert_eq!(to_big(&z(&a).div_euclid(z(&b))), eq);
                    assert_eq!(to_big(&z(&a).rem_euclid(z(&b))), er);
                }
                assert!(z(&top).checked_div_rem(Z::zero()).is_none());
                assert!(z(&top).div_exact(Z::zero()).is_none());
                assert_eq!(Z::zero().div_exact(z(&top)), Some(Z::zero()));
            }
        }
    };
}

rns_base!(B1: largest 1);
rns_base!(B2: largest 2);
rns_base!(B4: largest 4);
rns_base!(B5: largest 5);
rns_base!(B9: largest 9);
rns_base!(B16: largest 16);
rns_base!(Small: 5 = [3, 5, 7, 11, 13]);
rns_base!(Mixed: 4 = [3, 2_147_483_647, 101, 65_537]);
rns_base!(Low: 5 = [1_000_000_007, 998_244_353, 1_000_000_009, 754_974_721, 167_772_161]);

division_tests!(base1, B1, 1);
division_tests!(base2, B2, 2);
division_tests!(base4, B4, 4);
division_tests!(base5, B5, 5);
division_tests!(base9, B9, 9);
division_tests!(base16, B16, 16);
division_tests!(small, Small, 5);
division_tests!(mixed, Mixed, 4);
division_tests!(low, Low, 5);

/// Divisors that share a prime with M take the fallback path of div_exact.
#[test]
fn exact_division_by_multiples_of_a_modulus() {
    type Z = Rns<Small, 5>;
    for y in [3i64, 5, 15, -21, 33, 39, 65] {
        for q in -30i64..=30 {
            let x = q * y;
            if (x.unsigned_abs() as u128) < (1 << Z::max_bits()) {
                let got = Z::from_i64(x).unwrap().div_exact(Z::from_i64(y).unwrap());
                assert_eq!(got.and_then(|v| v.to_i128()), Some(q as i128), "{x} / {y}");
                let off = Z::from_i64(x + 1).and_then(|v| v.div_exact(Z::from_i64(y).unwrap()));
                assert!(off.is_none(), "{} / {y} is not exact", x + 1);
            }
        }
    }
    type W = Rns<B4, 4>;
    let p = B4::base().moduli()[0] as i128;
    let x = W::from_i128(p * 123_456_789).unwrap();
    assert_eq!(
        x.div_exact(W::from_i128(p).unwrap()).unwrap().to_i128(),
        Some(123_456_789)
    );
    assert!(W::from_i128(p * 7 + 1)
        .unwrap()
        .div_exact(W::from_i128(p).unwrap())
        .is_none());
}

#[test]
#[should_panic(expected = "divide by zero")]
fn div_operator_by_zero_panics() {
    let _ = Rns::<B4, 4>::one() / Rns::<B4, 4>::zero();
}
