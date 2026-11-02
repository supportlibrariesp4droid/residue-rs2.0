//! Property tests: RnsMod must be a faithful image of ℤ/Mℤ.
//! Oracle: num-bigint, an independent arbitrary-precision implementation.

use num_bigint::{BigInt, BigUint};
use num_traits::{One, Zero};
use proptest::prelude::*;
use residue::{rns_base, RnsBase, RnsMod};

fn limbs_to_big(limbs: &[u64]) -> BigUint {
    let bytes: Vec<u8> = limbs.iter().flat_map(|l| l.to_le_bytes()).collect();
    BigUint::from_bytes_le(&bytes)
}

/// Strategy: integers from 0 up to ~2 limbs beyond M, to exercise reduction.
fn big(max_limbs: usize) -> impl Strategy<Value = BigUint> {
    prop_oneof![
        (0u64..1000).prop_map(BigUint::from),
        prop::collection::vec(any::<u64>(), 0..=max_limbs).prop_map(|v| limbs_to_big(&v)),
    ]
}

macro_rules! ring_tests {
    ($modname:ident, $base:ident, $n:literal) => {
        mod $modname {
            use super::*;
            type Z = RnsMod<$base, $n>;

            fn m() -> BigUint {
                limbs_to_big($base::base().modulus_limbs())
            }
            fn rns(x: &BigUint) -> Z {
                Z::from_limbs(&x.to_u64_digits())
            }
            fn back(x: Z) -> BigUint {
                limbs_to_big(&x.to_limbs())
            }

            #[test]
            fn modulus_and_bits_are_consistent() {
                let m = m();
                let expected: BigUint = $base::base()
                    .moduli()
                    .iter()
                    .map(|&p| BigUint::from(p))
                    .product();
                assert_eq!(m, expected);
                assert_eq!($base::base().bits() as u64, m.bits());
            }

            #[test]
            fn identities_and_edges() {
                let m = m();
                assert!(Z::zero().is_zero());
                assert_eq!(back(Z::one()), BigUint::one());
                assert_eq!(back(Z::from_i128(-1)), &m - 1u32);
                assert!(rns(&m).is_zero());
                assert_eq!(back(rns(&(&m - 1u32))), &m - 1u32);
                assert_eq!(Z::from_u64(7).pow(0), Z::one());
            }

            proptest! {
                #![proptest_config(ProptestConfig::with_cases(512))]

                #[test]
                fn roundtrip(a in big($n + 2)) {
                    prop_assert_eq!(back(rns(&a)), &a % m());
                }

                #[test]
                fn add_sub_mul_neg(a in big($n + 2), b in big($n + 2)) {
                    let m = m();
                    let (ra, rb) = (rns(&a), rns(&b));
                    prop_assert_eq!(back(ra + rb), (&a + &b) % &m);
                    prop_assert_eq!(back(ra * rb), (&a * &b) % &m);
                    let diff = (BigInt::from(a.clone()) - BigInt::from(b.clone()))
                        .modpow(&BigInt::one(), &BigInt::from(m.clone()));
                    prop_assert_eq!(back(ra - rb), diff.to_biguint().unwrap());
                    prop_assert_eq!(back(-ra), (&m - (&a % &m)) % &m);
                }

                #[test]
                fn ring_laws(a in big($n), b in big($n), c in big($n)) {
                    let (a, b, c) = (rns(&a), rns(&b), rns(&c));
                    prop_assert_eq!(a * (b + c), a * b + a * c);
                    prop_assert_eq!((a * b) * c, a * (b * c));
                    prop_assert_eq!(a + b - b, a);
                }

                #[test]
                fn signed_input(x in any::<i128>()) {
                    let m = BigInt::from(m());
                    let expected = BigInt::from(x).modpow(&BigInt::one(), &m);
                    prop_assert_eq!(back(Z::from_i128(x)), expected.to_biguint().unwrap());
                }

                #[test]
                fn pow_matches_modpow(a in big($n), e in 0u64..5000) {
                    let m = m();
                    prop_assert_eq!(back(rns(&a).pow(e)), a.modpow(&BigUint::from(e), &m));
                }

                #[test]
                fn inverse(a in big($n)) {
                    let m = m();
                    let ra = rns(&a);
                    let coprime = num_integer::Integer::gcd(&(&a % &m), &m).is_one();
                    match ra.inv() {
                        Some(i) => {
                            prop_assert!(coprime);
                            prop_assert_eq!(ra * i, Z::one());
                        }
                        None => prop_assert!(!coprime || (&a % &m).is_zero()),
                    }
                }

                #[test]
                fn decimal_parse_and_display(a in big($n + 2), neg in any::<bool>()) {
                    let m = BigInt::from(m());
                    let s = if neg { format!("-{}", a) } else { a.to_string() };
                    let parsed: Z = s.parse().unwrap();
                    let signed = if neg { -BigInt::from(a.clone()) } else { BigInt::from(a.clone()) };
                    let expected = signed.modpow(&BigInt::one(), &m).to_biguint().unwrap();
                    prop_assert_eq!(back(parsed), expected.clone());
                    prop_assert_eq!(parsed.to_string(), expected.to_string());
                }
            }
        }
    };
}

rns_base!(B1: largest 1);
rns_base!(B2: largest 2);
rns_base!(B4: largest 4);
rns_base!(B9: largest 9);
rns_base!(B16: largest 16);
rns_base!(Small: 5 = [3, 5, 7, 11, 13]);
rns_base!(Mixed: 4 = [3, 2_147_483_647, 101, 65_537]);

ring_tests!(base1, B1, 1);
ring_tests!(base2, B2, 2);
ring_tests!(base4, B4, 4);
ring_tests!(base9, B9, 9);
ring_tests!(base16, B16, 16);
ring_tests!(small, Small, 5);
ring_tests!(mixed, Mixed, 4);

#[test]
fn parse_rejects_garbage() {
    type Z = RnsMod<B4, 4>;
    assert!("".parse::<Z>().is_err());
    assert!("-".parse::<Z>().is_err());
    assert!("12a3".parse::<Z>().is_err());
    assert!(" 12".parse::<Z>().is_err());
    assert!("+12".parse::<Z>().is_ok());
}

#[test]
fn zero_not_invertible() {
    assert!(RnsMod::<B4, 4>::zero().inv().is_none());
    assert!(RnsMod::<Small, 5>::from_u64(3).inv().is_none());
}
