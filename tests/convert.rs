//! Batch conversions of `RnsVec` to and from binary, against num-bigint.
//!
//! Vectors reach 300 elements so that the 128-element blocks of the kernels
//! are crossed, and their lengths vary so that SIMD remainders are exercised.
#![cfg(feature = "alloc")]

use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{Signed, Zero};
use proptest::prelude::*;
use residue::{rns_base, OutOfRange, Rns, RnsBase, RnsVec};

/// Explicit SIMD kernels on (0) or off (2); see `__set_simd_level`. The
/// level is a global, and other tests may change it concurrently, which is
/// harmless: every level must give the same result.
fn simd_levels() -> impl Strategy<Value = u8> {
    prop_oneof![Just(0u8), Just(2u8)]
}

fn set_simd_level(_level: u8) {
    #[cfg(all(feature = "std", target_arch = "x86_64"))]
    residue::__set_simd_level(_level);
}

fn limbs_to_biguint(limbs: &[u64]) -> BigUint {
    let bytes: Vec<u8> = limbs.iter().flat_map(|l| l.to_le_bytes()).collect();
    BigUint::from_bytes_le(&bytes)
}

fn from_big<B: RnsBase<N>, const N: usize>(x: &BigInt) -> Rns<B, N> {
    Rns::from_sign_magnitude(x.sign() == Sign::Minus, &x.magnitude().to_u64_digits()).unwrap()
}

/// `x` as `width` limbs of two's complement, or `None` if it does not fit.
fn twos(x: &BigInt, width: usize) -> Option<Vec<u64>> {
    let bytes = x.to_signed_bytes_le();
    if bytes.len() > 8 * width {
        return None;
    }
    let fill = if x.is_negative() { 0xFF } else { 0 };
    let mut b = bytes;
    b.resize(8 * width, fill);
    Some(
        b.chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect(),
    )
}

fn from_twos(limbs: &[u64]) -> BigInt {
    let bytes: Vec<u8> = limbs.iter().flat_map(|l| l.to_le_bytes()).collect();
    BigInt::from_signed_bytes_le(&bytes)
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

/// Vectors whose values all have at most `bits` bits; sometimes all small,
/// sometimes near the limit.
fn values(bits: u32) -> impl Strategy<Value = Vec<BigInt>> {
    (1..=bits).prop_flat_map(|b| prop::collection::vec(signed(b), 0..=300))
}

macro_rules! convert_tests {
    ($modname:ident, $base:ident, $n:literal) => {
        mod $modname {
            use super::*;
            type Z = Rns<$base, $n>;
            type V = RnsVec<$base, $n>;

            fn vec(xs: &[BigInt]) -> V {
                xs.iter().map(from_big::<$base, $n>).collect()
            }
            fn max_len(xs: &[BigInt]) -> u32 {
                xs.iter().map(|x| x.bits() as u32).max().unwrap_or(0)
            }

            proptest! {
                #![proptest_config(ProptestConfig::with_cases(64))]

                /// At every width: exact where the values fit, an error where
                /// one does not, sign extension above the bound's width.
                #[test]
                fn to_twos_complement_any_width(xs in values(Z::max_bits()), level in simd_levels()) {
                    set_simd_level(level);
                    let v = vec(&xs);
                    let need = v.twos_complement_width();
                    for width in [1, 2, 3, need.saturating_sub(1).max(1), need, need + 2] {
                        let mut out = vec![0u64; xs.len() * width];
                        let fits = xs.iter().all(|x| twos(x, width).is_some());
                        match v.to_twos_complement(width, &mut out) {
                            Ok(()) => {
                                prop_assert!(fits, "width {} accepted a value too long", width);
                                for (k, x) in xs.iter().enumerate() {
                                    prop_assert_eq!(
                                        &out[k * width..(k + 1) * width],
                                        &twos(x, width).unwrap()[..],
                                        "width {}, element {}", width, k
                                    );
                                }
                            }
                            Err(OutOfRange) => prop_assert!(!fits, "width {} rejected values that fit", width),
                        }
                    }
                }

                #[test]
                fn to_i128s_and_i64s(xs in values(Z::max_bits().min(140)), level in simd_levels()) {
                    set_simd_level(level);
                    let v = vec(&xs);
                    let as128: Option<Vec<i128>> = xs.iter().map(|x| i128::try_from(x).ok()).collect();
                    prop_assert_eq!(v.to_i128s().ok(), as128);
                    let as64: Option<Vec<i64>> = xs.iter().map(|x| i64::try_from(x).ok()).collect();
                    prop_assert_eq!(v.to_i64s().ok(), as64);
                }

                /// The bound is pessimistic after arithmetic: (a·b + a) − a·b.
                #[test]
                fn pessimistic_bound(xs in values(Z::max_bits() / 2 - 2)) {
                    let a = vec(&xs);
                    let ab = &a * &a;
                    let back = &(&ab + &a) - &ab;
                    prop_assert!(back.bits_bound() >= a.bits_bound());
                    let w = back.twos_complement_width();
                    let mut out = vec![0u64; xs.len() * w];
                    back.to_twos_complement(w, &mut out).unwrap();
                    for (k, x) in xs.iter().enumerate() {
                        prop_assert_eq!(&from_twos(&out[k * w..(k + 1) * w]), x);
                        let s = back.get(k).unwrap();
                        prop_assert_eq!(s.to_i128(), i128::try_from(x).ok());
                        prop_assert_eq!(s.signum(), match x.sign() { Sign::Minus => -1, Sign::NoSign => 0, Sign::Plus => 1 });
                    }
                    let mut t = back.clone();
                    t.tighten();
                    prop_assert_eq!(t.bits_bound(), max_len(&xs));
                }

                #[test]
                fn from_twos_complement_round_trip(xs in values(Z::max_bits()), extra in 0usize..3) {
                    let width = xs.iter().map(|x| x.to_signed_bytes_le().len().div_ceil(8)).max().unwrap_or(1).max(1) + extra;
                    let input: Vec<u64> = xs.iter().flat_map(|x| twos(x, width).unwrap()).collect();
                    let v = V::from_twos_complement(width, &input).unwrap();
                    prop_assert_eq!(v.len(), xs.len());
                    prop_assert_eq!(v.bits_bound(), max_len(&xs));
                    for (k, x) in xs.iter().enumerate() {
                        prop_assert_eq!(v.get(k).unwrap(), from_big::<$base, $n>(x), "element {}", k);
                    }
                    let mut back = vec![0u64; input.len()];
                    v.to_twos_complement(width, &mut back).unwrap();
                    prop_assert_eq!(back, input);
                }

                #[test]
                fn tighten_matches_bit_length(xs in values(Z::max_bits())) {
                    let mut v = vec(&xs);
                    v.tighten();
                    prop_assert_eq!(v.bits_bound(), max_len(&xs));
                }
            }

            #[cfg(feature = "num-bigint")]
            proptest! {
                #![proptest_config(ProptestConfig::with_cases(64))]
                #[test]
                fn bigints_round_trip(xs in values(Z::max_bits())) {
                    let v = V::from_bigints(&xs).unwrap();
                    prop_assert_eq!(&v, &vec(&xs));
                    prop_assert_eq!(v.to_bigints(), xs);
                }
            }

            #[test]
            fn out_of_range_inputs() {
                let too_big = BigInt::from(1u8) << Z::max_bits();
                for x in [too_big.clone(), -too_big] {
                    let w = x.to_signed_bytes_le().len().div_ceil(8);
                    let input = twos(&x, w).unwrap();
                    assert!(V::from_twos_complement(w, &input).is_err());
                    #[cfg(feature = "num-bigint")]
                    assert!(V::from_bigints(&[BigInt::from(1), x]).is_err());
                }
            }

            #[test]
            fn extremes_of_the_range() {
                let m = (BigInt::from(1u8) << Z::max_bits()) - 1u8;
                let xs = vec![m.clone(), -m, BigInt::zero(), BigInt::from(-1), BigInt::from(1)];
                let v = vec(&xs);
                let w = v.twos_complement_width();
                let mut out = vec![0u64; xs.len() * w];
                v.to_twos_complement(w, &mut out).unwrap();
                for (k, x) in xs.iter().enumerate() {
                    assert_eq!(&from_twos(&out[k * w..(k + 1) * w]), x);
                }
            }
        }
    };
}

const fn three_and_large() -> [u32; 6] {
    let l = residue::primes::largest_primes::<5>();
    [3, l[0], l[1], l[2], l[3], l[4]]
}

rns_base!(B1: largest 1);
rns_base!(B2: largest 2);
rns_base!(B4: largest 4);
rns_base!(B9: largest 9);
rns_base!(B16: largest 16);
rns_base!(B40: largest 40);
rns_base!(Low: 5 = [1_000_000_007, 998_244_353, 1_000_000_009, 754_974_721, 167_772_161]);
rns_base!(Small: 5 = [3, 5, 7, 11, 13]);
rns_base!(Mixed: 4 = [3, 2_147_483_647, 101, 65_537]);
rns_base!(ThreeLarge: 6 = three_and_large());

convert_tests!(base1, B1, 1);
convert_tests!(base2, B2, 2);
convert_tests!(base4, B4, 4);
convert_tests!(base9, B9, 9);
convert_tests!(base16, B16, 16);
convert_tests!(base40, B40, 40);
convert_tests!(low, Low, 5);
convert_tests!(small, Small, 5);
convert_tests!(mixed, Mixed, 4);
convert_tests!(three_large, ThreeLarge, 6);

#[test]
fn empty_vector() {
    let v = RnsVec::<B9, 9>::new();
    assert_eq!(v.to_i128s(), Ok(vec![]));
    assert_eq!(v.to_i64s(), Ok(vec![]));
    let mut out = [];
    assert_eq!(v.to_twos_complement(3, &mut out), Ok(()));
    let e = RnsVec::<B9, 9>::from_twos_complement(2, &[]).unwrap();
    assert!(e.is_empty());
}

#[test]
#[should_panic(expected = "len · width")]
fn wrong_output_length_panics() {
    let v = RnsVec::<B9, 9>::from_i128s(&[1, 2]).unwrap();
    let mut out = [0u64; 3];
    let _ = v.to_twos_complement(2, &mut out);
}
