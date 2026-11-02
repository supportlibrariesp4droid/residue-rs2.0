//! Property tests for exact signed integers (`Rns`) and base extension.
//! Oracle: num-bigint.

use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{One, Signed, Zero};
use proptest::prelude::*;
use residue::{rns_base, Extender, ParseExactError, Rns, RnsBase};

fn limbs_to_biguint(limbs: &[u64]) -> BigUint {
    let bytes: Vec<u8> = limbs.iter().flat_map(|l| l.to_le_bytes()).collect();
    BigUint::from_bytes_le(&bytes)
}

fn to_big<B: RnsBase<N>, const N: usize>(x: &Rns<B, N>) -> BigInt {
    let (neg, mag) = x.to_sign_magnitude();
    let m = BigInt::from(limbs_to_biguint(&mag));
    if neg {
        -m
    } else {
        m
    }
}

fn from_big<B: RnsBase<N>, const N: usize>(x: &BigInt) -> Option<Rns<B, N>> {
    Rns::from_sign_magnitude(x.sign() == Sign::Minus, &x.magnitude().to_u64_digits())
}

fn bitlen(x: &BigInt) -> u64 {
    x.magnitude().bits()
}

/// A signed integer whose bit length is exactly `bl` (0 means zero).
fn with_bits(bl: u32, neg: bool, noise: &[u64]) -> BigInt {
    if bl == 0 {
        return BigInt::zero();
    }
    let mask = (BigUint::one() << bl) - 1u32;
    let top = BigUint::one() << (bl - 1);
    let mag = (limbs_to_biguint(noise) & mask) | top;
    let v = BigInt::from(mag);
    if neg {
        -v
    } else {
        v
    }
}

/// Strategy: signed values with bit length uniform in `[0, max_bits]`, so small,
/// medium and near-full-range values (and every sign tier) are all exercised.
fn signed(max_bits: u32) -> impl Strategy<Value = BigInt> {
    let limbs = (max_bits as usize).div_ceil(64) + 1;
    (
        0..=max_bits,
        any::<bool>(),
        prop::collection::vec(any::<u64>(), limbs),
    )
        .prop_map(|(bl, neg, noise)| with_bits(bl, neg, &noise))
}

macro_rules! exact_tests {
    ($modname:ident, $base:ident, $n:literal) => {
        mod $modname {
            use super::*;
            type Z = Rns<$base, $n>;

            fn maxb() -> u32 {
                Z::max_bits()
            }
            fn z(x: &BigInt) -> Z {
                from_big::<$base, $n>(x).expect("value in range")
            }

            #[test]
            fn edges() {
                let m = maxb();
                let top: BigInt = (BigInt::one() << m) - 1;
                let mut cases = vec![
                    BigInt::zero(),
                    BigInt::one(),
                    -BigInt::one(),
                    top.clone(),
                    -top.clone(),
                ];
                for e in [62u32, 63, 64, 65, 127, 128] {
                    if e < m {
                        let p: BigInt = BigInt::one() << e;
                        let below: BigInt = &p - 1;
                        let above: BigInt = &p + 1;
                        cases.extend([p.clone(), -p.clone(), below.clone(), -below, above]);
                    }
                }
                for c in &cases {
                    let x = z(c);
                    assert_eq!(&to_big(&x), c);
                    assert_eq!(x.to_string(), c.to_string());
                    assert_eq!(x.signum(), c.signum().to_string().parse::<i32>().unwrap());
                }
                let over = BigInt::one() << m;
                assert!(from_big::<$base, $n>(&over).is_none());
                assert!(from_big::<$base, $n>(&-over.clone()).is_none());
                assert_eq!(over.to_string().parse::<Z>(), Err(ParseExactError::OutOfRange));
            }

            #[test]
            #[should_panic(expected = "integer overflow")]
            fn operator_overflow_panics() {
                let top = z(&((BigInt::one() << maxb()) - 1));
                let _ = top * top;
            }

            proptest! {
                #![proptest_config(ProptestConfig::with_cases(512))]

                #[test]
                fn roundtrip_and_exact_bound(a in signed(maxb())) {
                    let x = z(&a);
                    prop_assert_eq!(to_big(&x), a.clone());
                    prop_assert_eq!(x.bits_bound() as u64, bitlen(&a));
                    prop_assert_eq!(x.to_string(), a.to_string());
                    prop_assert_eq!(x.to_string().parse::<Z>().unwrap(), x);
                    if let Ok(v) = i128::try_from(&a) {
                        prop_assert_eq!(x.to_i128(), Some(v));
                    } else {
                        prop_assert_eq!(x.to_i128(), None);
                    }
                }

                #[test]
                fn sign_is_exact(a in signed(maxb())) {
                    let x = z(&a);
                    let expected = if a.is_negative() { -1 } else if a.is_zero() { 0 } else { 1 };
                    prop_assert_eq!(x.signum(), expected);
                    prop_assert_eq!(to_big(&x.abs()), a.abs());
                }

                #[test]
                fn compare_is_exact(a in signed(maxb()), b in signed(maxb())) {
                    prop_assert_eq!(z(&a).cmp(&z(&b)), a.cmp(&b));
                }

                #[test]
                fn compare_near_equal(a in signed(maxb()), d in -3i64..=3) {
                    let b = &a + d;
                    if bitlen(&b) <= maxb() as u64 {
                        prop_assert_eq!(z(&a).cmp(&z(&b)), a.cmp(&b));
                    }
                }

                #[test]
                fn checked_arithmetic(a in signed(maxb()), b in signed(maxb())) {
                    let (x, y) = (z(&a), z(&b));
                    let (ba, bb) = (bitlen(&a), bitlen(&b));
                    let m = maxb() as u64;
                    let add_bound = if ba == 0 { bb } else if bb == 0 { ba } else { ba.max(bb) + 1 };
                    let mul_bound = if ba == 0 || bb == 0 { 0 } else { ba + bb };

                    match x.checked_add(y) {
                        Some(s) => prop_assert_eq!(to_big(&s), &a + &b),
                        None => prop_assert!(add_bound > m),
                    }
                    match x.checked_sub(y) {
                        Some(s) => prop_assert_eq!(to_big(&s), &a - &b),
                        None => prop_assert!(add_bound > m),
                    }
                    match x.checked_mul(y) {
                        Some(p) => prop_assert_eq!(to_big(&p), &a * &b),
                        None => prop_assert!(mul_bound > m),
                    }
                    prop_assert_eq!(to_big(&-x), -a.clone());
                }

                #[test]
                fn tighten_recovers_exact_bound(a in signed(maxb() - 2), b in signed(maxb() - 2)) {
                    let (x, y) = (z(&a), z(&b));
                    let back = (x + y) - y;
                    prop_assert_eq!(back, x);
                    prop_assert_eq!(back.tighten().bits_bound() as u64, bitlen(&a));
                }

                #[test]
                fn rns_mod_view_agrees(a in signed(maxb())) {
                    let x = z(&a);
                    let m = BigInt::from(limbs_to_biguint($base::base().modulus_limbs()));
                    let expected = ((&a % &m) + &m) % &m;
                    let got = limbs_to_biguint(&x.to_rns_mod().to_limbs());
                    prop_assert_eq!(BigInt::from(got), expected);
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
rns_base!(Low: 5 = [1_000_000_007, 998_244_353, 1_000_000_009, 754_974_721, 167_772_161]);

exact_tests!(base1, B1, 1);
exact_tests!(base2, B2, 2);
exact_tests!(base4, B4, 4);
exact_tests!(base9, B9, 9);
exact_tests!(base16, B16, 16);
exact_tests!(small, Small, 5);
exact_tests!(mixed, Mixed, 4);
exact_tests!(low, Low, 5);

/// Every sign tier must actually be reached by values of all magnitudes,
/// otherwise the tests above could be passing on the easy path only.
fn tier_counts<B: RnsBase<N>, const N: usize>() -> [usize; 3] {
    let maxb = Rns::<B, N>::max_bits();
    let mut counts = [0usize; 3];
    let mut seed = 0x1234_5678_9abc_def0u64;
    for bl in 0..=maxb {
        for neg in [false, true] {
            for _ in 0..4 {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let noise: Vec<u64> = (0..(maxb as usize / 64 + 2))
                    .map(|k| seed.rotate_left(k as u32 * 13) ^ (k as u64).wrapping_mul(0x9E37))
                    .collect();
                let a = with_bits(bl, neg, &noise);
                let x = from_big::<B, N>(&a).unwrap();
                let expected = if a.is_negative() {
                    -1
                } else if a.is_zero() {
                    0
                } else {
                    1
                };
                assert_eq!(x.signum(), expected, "wrong sign for {a}");
                counts[x.__sign_tier() as usize - 1] += 1;
            }
        }
    }
    counts
}

#[test]
fn all_sign_tiers_are_exercised() {
    for (name, c) in [
        ("B4", tier_counts::<B4, 4>()),
        ("B9", tier_counts::<B9, 9>()),
        ("B16", tier_counts::<B16, 16>()),
        ("Low", tier_counts::<Low, 5>()),
    ] {
        println!("{name}: tier1={} tier2={} tier3={}", c[0], c[1], c[2]);
        assert!(
            c.iter().all(|&k| k > 0),
            "{name}: some sign tier never reached: {c:?}"
        );
    }
    let c = tier_counts::<B2, 2>();
    assert_eq!((c[1], c[2]), (0, 0));
}

fn check_extension<S: RnsBase<NS>, const NS: usize, T: RnsBase<NT>, const NT: usize>(
    a: &BigInt,
    ext: &Extender<S, NS, T, NT>,
) -> Result<(), TestCaseError> {
    let x = from_big::<S, NS>(a).unwrap();
    let tmax = Rns::<T, NT>::max_bits() as u64;
    match ext.extend(&x) {
        Some(y) => {
            prop_assert_eq!(to_big(&y), a.clone());
            prop_assert_eq!(y.bits_bound(), x.bits_bound());
            prop_assert_eq!(y.signum(), x.signum());
            prop_assert_eq!(to_big(&-y), -a.clone());
            if let Some(d) = y.checked_sub(y) {
                prop_assert!(d.is_zero());
            }
        }
        None => prop_assert!(bitlen(a) > tmax),
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn widen_b4_to_b9(a in signed(Rns::<B4, 4>::max_bits())) {
        check_extension(&a, &Extender::<B4, 4, B9, 9>::new())?;
    }

    #[test]
    fn narrow_b9_to_b4(a in signed(Rns::<B9, 9>::max_bits())) {
        check_extension(&a, &Extender::<B9, 9, B4, 4>::new())?;
    }

    #[test]
    fn disjoint_b9_to_low(a in signed(Rns::<B9, 9>::max_bits())) {
        check_extension(&a, &Extender::<B9, 9, Low, 5>::new())?;
    }

    #[test]
    fn disjoint_low_to_b16(a in signed(Rns::<Low, 5>::max_bits())) {
        check_extension(&a, &Extender::<Low, 5, B16, 16>::new())?;
    }

    #[test]
    fn mixed_to_small(a in signed(Rns::<Mixed, 4>::max_bits())) {
        check_extension(&a, &Extender::<Mixed, 4, Small, 5>::new())?;
    }

    /// The point of widening: compute a product that does not fit the narrow
    /// base, exactly, and compare it.
    #[test]
    fn widen_then_multiply(a in signed(Rns::<B4, 4>::max_bits()), b in signed(Rns::<B4, 4>::max_bits())) {
        let ext = Extender::<B4, 4, B9, 9>::new();
        let (x, y) = (from_big::<B4, 4>(&a).unwrap(), from_big::<B4, 4>(&b).unwrap());
        let p = ext.extend(&x).unwrap() * ext.extend(&y).unwrap();
        prop_assert_eq!(to_big(&p), &a * &b);
        prop_assert_eq!(p.cmp(&ext.extend(&x).unwrap()), (&a * &b).cmp(&a));
    }
}

#[test]
fn rebase_convenience() {
    let x: Rns<B4, 4> = "-98765432109876543210987654321".parse().unwrap();
    let y: Rns<B9, 9> = x.rebase().unwrap();
    assert_eq!(y.to_string(), "-98765432109876543210987654321");
}
