//! Conversions enabled by the `num-bigint` feature.
#![cfg(feature = "num-bigint")]

use num_bigint::{BigInt, BigUint};
use residue::{rns_base, RnsMod};

rns_base!(B9: largest 9);
type Z = RnsMod<B9, 9>;

#[test]
fn biguint_roundtrip() {
    let x: BigUint = "123456789012345678901234567890123456789012345678901234567890"
        .parse()
        .unwrap();
    let r = Z::from(&x);
    assert_eq!(r.to_biguint(), x);
}

#[test]
fn bigint_negative() {
    let x = BigInt::from(-5);
    assert_eq!(Z::from(&x), Z::from_i128(-5));
    assert_eq!(Z::from(&x) + Z::from_u64(5), Z::zero());
}

#[test]
fn exact_bigint_roundtrip() {
    use residue::Rns;
    let x: BigInt = "-123456789012345678901234567890123456789012345678901234567890"
        .parse()
        .unwrap();
    let r = Rns::<B9, 9>::try_from(&x).unwrap();
    assert_eq!(r.to_bigint(), x);
    assert!(r.is_negative());
}
