//! RnsVec must agree element by element with scalar Rns (which is itself
//! verified against num-bigint), and its reductions must match num-bigint.
#![cfg(feature = "alloc")]

use num_bigint::{BigInt, BigUint, Sign};
use num_traits::Zero;
use proptest::prelude::*;
use residue::{rns_base, Rns, RnsBase, RnsVec};

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

fn from_big<B: RnsBase<N>, const N: usize>(x: &BigInt) -> Rns<B, N> {
    Rns::from_sign_magnitude(x.sign() == Sign::Minus, &x.magnitude().to_u64_digits()).unwrap()
}

/// Signed values with bit length uniform in `[0, bits]`.
fn signed(bits: u32) -> impl Strategy<Value = BigInt> {
    let limbs = (bits as usize).div_ceil(64) + 1;
    (
        0..=bits,
        any::<bool>(),
        prop::collection::vec(any::<u64>(), limbs),
    )
        .prop_map(|(bl, neg, noise)| {
            if bl == 0 {
                return BigInt::zero();
            }
            let mask = (BigUint::from(1u8) << bl) - 1u8;
            let top = BigUint::from(1u8) << (bl - 1);
            let v = BigInt::from((limbs_to_biguint(&noise) & mask) | top);
            if neg {
                -v
            } else {
                v
            }
        })
}

/// Two vectors of equal length (0..=67 so SIMD remainders are exercised).
fn pair(bits: u32) -> impl Strategy<Value = (Vec<BigInt>, Vec<BigInt>)> {
    (0usize..=67).prop_flat_map(move |n| {
        (
            prop::collection::vec(signed(bits), n),
            prop::collection::vec(signed(bits), n),
        )
    })
}

macro_rules! batch_tests {
    ($modname:ident, $base:ident, $n:literal) => {
        mod $modname {
            use super::*;
            type Z = Rns<$base, $n>;
            type V = RnsVec<$base, $n>;

            fn vec(xs: &[BigInt]) -> V {
                xs.iter().map(from_big::<$base, $n>).collect()
            }
            fn half() -> u32 {
                Z::max_bits() / 2 - 4
            }

            proptest! {
                #![proptest_config(ProptestConfig::with_cases(256))]

                #[test]
                fn elementwise_matches_scalar((a, b) in pair(half())) {
                    let (va, vb) = (vec(&a), vec(&b));
                    let sum = &va + &vb;
                    let diff = &va - &vb;
                    let prod = &va * &vb;
                    let neg = -&va;
                    for k in 0..a.len() {
                        let (x, y) = (from_big::<$base, $n>(&a[k]), from_big::<$base, $n>(&b[k]));
                        prop_assert_eq!(sum.get(k).unwrap(), x + y);
                        prop_assert_eq!(diff.get(k).unwrap(), x - y);
                        prop_assert_eq!(prod.get(k).unwrap(), x * y);
                        prop_assert_eq!(neg.get(k).unwrap(), -x);
                        prop_assert_eq!(to_big(&prod.get(k).unwrap()), &a[k] * &b[k]);
                    }
                }

                #[test]
                fn scale_matches_scalar((a, _b) in pair(half()), k in signed(half())) {
                    let mut va = vec(&a);
                    va.scale_assign(from_big::<$base, $n>(&k));
                    for (i, x) in a.iter().enumerate() {
                        prop_assert_eq!(to_big(&va.get(i).unwrap()), x * &k);
                    }
                }

                #[test]
                fn dot_and_sum_exact((a, b) in pair(half())) {
                    let (va, vb) = (vec(&a), vec(&b));
                    let expected_dot: BigInt = a.iter().zip(&b).map(|(x, y)| x * y).sum();
                    let expected_sum: BigInt = a.iter().sum();
                    prop_assert_eq!(to_big(&va.dot(&vb).unwrap()), expected_dot);
                    prop_assert_eq!(to_big(&va.sum().unwrap()), expected_sum);
                }

                #[test]
                fn overflow_is_refused_and_leaves_input_intact((a, b) in pair(Z::max_bits())) {
                    let (mut va, vb) = (vec(&a), vec(&b));
                    let before = va.clone();
                    match va.checked_mul_assign(&vb) {
                        Ok(()) => {
                            for k in 0..a.len() {
                                prop_assert_eq!(to_big(&va.get(k).unwrap()), &a[k] * &b[k]);
                            }
                        }
                        Err(_) => prop_assert_eq!(va, before),
                    }
                }
            }
        }
    };
}

rns_base!(B2: largest 2);
rns_base!(B4: largest 4);
rns_base!(B9: largest 9);
rns_base!(B16: largest 16);
rns_base!(Low: 5 = [1_000_000_007, 998_244_353, 1_000_000_009, 754_974_721, 167_772_161]);
rns_base!(Small: 5 = [3, 5, 7, 11, 13]);

batch_tests!(base2, B2, 2);
batch_tests!(base4, B4, 4);
batch_tests!(base9, B9, 9);
batch_tests!(base16, B16, 16);
batch_tests!(low, Low, 5);
batch_tests!(small, Small, 5);

/// The lazy dot-product accumulators must survive long vectors of maximal
/// residues (the worst case for the 32-bit-half sums).
#[test]
fn long_dot_worst_case_residues() {
    type V = RnsVec<B9, 9>;
    let n = 300_000;
    let minus_one = Rns::<B9, 9>::from_i64(-1).unwrap();
    let v: V = std::iter::repeat_n(minus_one, n).collect();
    assert_eq!(v.dot(&v).unwrap().to_i128(), Some(n as i128));
    assert_eq!(v.sum().unwrap().to_i128(), Some(-(n as i128)));
}

#[test]
#[should_panic(expected = "length mismatch")]
fn length_mismatch_panics() {
    let a = RnsVec::<B4, 4>::from_i128s(&[1, 2, 3]).unwrap();
    let b = RnsVec::<B4, 4>::from_i128s(&[1, 2]).unwrap();
    let _ = a.dot(&b);
}

#[test]
fn empty_and_roundtrip() {
    let e = RnsVec::<B4, 4>::new();
    assert!(e.is_empty());
    assert!(e.dot(&e).unwrap().is_zero());
    assert!(e.sum().unwrap().is_zero());
    let xs = [-5i128, 0, 7, i64::MAX as i128, -(1i128 << 100)];
    let v = RnsVec::<B4, 4>::from_i128s(&xs).unwrap();
    let back: Vec<i128> = v.iter().map(|x| x.to_i128().unwrap()).collect();
    assert_eq!(back, xs);
}

rns_base!(B1: largest 1);

macro_rules! from_i128s_tests {
    ($name:ident, $base:ident, $n:literal) => {
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(256))]
            #[test]
            fn $name(xs in prop::collection::vec(
                prop_oneof![any::<i128>(), any::<i64>().prop_map(|v| v as i128), (-1000i128..1000)],
                0..80,
            )) {
                let maxb = Rns::<$base, $n>::max_bits();
                let fits = xs.iter().all(|x| 128 - x.unsigned_abs().leading_zeros() <= maxb);
                match RnsVec::<$base, $n>::from_i128s(&xs) {
                    Ok(v) => {
                        prop_assert!(fits);
                        prop_assert_eq!(v.len(), xs.len());
                        for (k, &x) in xs.iter().enumerate() {
                            let s = Rns::<$base, $n>::from_i128(x).unwrap();
                            prop_assert_eq!(v.get(k).unwrap(), s);
                            prop_assert_eq!(v.get(k).unwrap().to_i128(), Some(x));
                        }
                    }
                    Err(_) => prop_assert!(!fits),
                }
            }
        }
    };
}

from_i128s_tests!(from_i128s_b1, B1, 1);
from_i128s_tests!(from_i128s_b4, B4, 4);
from_i128s_tests!(from_i128s_b9, B9, 9);
from_i128s_tests!(from_i128s_b16, B16, 16);

#[test]
fn from_i128s_extremes() {
    let xs = [
        i128::MIN,
        i128::MAX,
        0,
        -1,
        1,
        i64::MIN as i128,
        u64::MAX as i128,
    ];
    let v = RnsVec::<B9, 9>::from_i128s(&xs).unwrap();
    for (k, &x) in xs.iter().enumerate() {
        assert_eq!(v.get(k).unwrap().to_i128(), Some(x), "element {k}");
    }
}
