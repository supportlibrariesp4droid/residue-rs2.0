//! RnsMatrix: exact products and determinants against num-bigint.
#![cfg(feature = "alloc")]

use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{One, Zero};
use proptest::prelude::*;
use residue::{rns_base, Rns, RnsBase, RnsMatrix, RnsVec};

/// The explicit SIMD levels to test on this build (see `__set_simd_level`).
fn simd_levels() -> Vec<u8> {
    if cfg!(all(feature = "std", target_arch = "x86_64")) {
        vec![0, 1, 2]
    } else {
        vec![0]
    }
}

fn set_simd_level(_level: u8) {
    #[cfg(all(feature = "std", target_arch = "x86_64"))]
    residue::__set_simd_level(_level);
}

fn big(limbs: &[u64]) -> BigUint {
    let bytes: Vec<u8> = limbs.iter().flat_map(|l| l.to_le_bytes()).collect();
    BigUint::from_bytes_le(&bytes)
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

type Mat = Vec<Vec<BigInt>>;

/// Signed value with bit length uniform in [0, bits].
fn signed(bits: u32) -> impl Strategy<Value = BigInt> {
    let n = (bits as usize).div_ceil(64) + 1;
    (
        0..=bits,
        any::<bool>(),
        prop::collection::vec(any::<u64>(), n),
    )
        .prop_map(|(bl, neg, noise)| {
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

fn mat(rows: usize, cols: usize, bits: u32) -> impl Strategy<Value = Mat> {
    prop::collection::vec(prop::collection::vec(signed(bits), cols), rows)
}

fn to_rns<B: RnsBase<N>, const N: usize>(m: &Mat, cols: usize) -> RnsMatrix<B, N> {
    RnsMatrix::from_fn(m.len(), cols, |r, c| from_big::<B, N>(&m[r][c]))
}

fn matmul_oracle(a: &Mat, b: &Mat, k: usize, p: usize) -> Mat {
    a.iter()
        .map(|row| {
            (0..p)
                .map(|j| (0..k).map(|t| &row[t] * &b[t][j]).sum())
                .collect()
        })
        .collect()
}

/// Fraction-free Bareiss elimination over BigInt: an independent algorithm.
fn det_oracle(a: &Mat) -> BigInt {
    let n = a.len();
    if n == 0 {
        return BigInt::one();
    }
    let mut m = a.clone();
    let mut sign = BigInt::one();
    let mut prev = BigInt::one();
    for k in 0..n - 1 {
        if m[k][k].is_zero() {
            match (k + 1..n).find(|&r| !m[r][k].is_zero()) {
                Some(r) => {
                    m.swap(k, r);
                    sign = -sign;
                }
                None => return BigInt::zero(),
            }
        }
        for i in k + 1..n {
            for j in k + 1..n {
                m[i][j] = (&m[k][k] * &m[i][j] - &m[i][k] * &m[k][j]) / &prev;
            }
        }
        prev = m[k][k].clone();
    }
    sign * &m[n - 1][n - 1]
}

macro_rules! matrix_tests {
    ($modname:ident, $base:ident, $n:literal) => {
        mod $modname {
            use super::*;
            type M = RnsMatrix<$base, $n>;
            fn maxb() -> u32 { Rns::<$base, $n>::max_bits() }

            /// Dimensions and an entry size that keeps products in range.
            fn dims() -> impl Strategy<Value = (usize, usize, usize, u32)> {
                (0usize..=9, 0usize..=12, 0usize..=9).prop_map(|(m, k, p)| {
                    let room = maxb().saturating_sub(4) / 2;
                    (m, k, p, room)
                })
            }

            proptest! {
                #![proptest_config(ProptestConfig::with_cases(160))]

                #[test]
                fn product_matches_oracle(
                    (a, b, k, p) in dims().prop_flat_map(|(m, k, p, bits)| (mat(m, k, bits), mat(k, p, bits), Just(k), Just(p)))
                ) {
                    let (ra, rb) = (to_rns::<$base, $n>(&a, k), to_rns::<$base, $n>(&b, p));
                    let c = ra.checked_matmul(&rb).expect("sized to fit");
                    let expected = matmul_oracle(&a, &b, k, p);
                    prop_assert_eq!((c.rows(), c.cols()), (a.len(), p));
                    for (r, row) in expected.iter().enumerate() {
                        for (j, v) in row.iter().enumerate() {
                            prop_assert_eq!(&to_big(&c.get(r, j).unwrap()), v);
                        }
                    }
                }

                /// Shapes that cross the SIMD tiles (16 columns with AVX-512,
                /// 8 with AVX2), with leftover rows, columns and a `k` that is
                /// not a multiple of four.
                #[test]
                fn product_matches_oracle_large_shapes(
                    (a, b, k, p) in (0usize..=37, 0usize..=41, 0usize..=40)
                        .prop_flat_map(|(m, k, p)| {
                            let bits = maxb().saturating_sub(8) / 2;
                            (mat(m, k, bits), mat(k, p, bits), Just(k), Just(p))
                        })
                ) {
                    let (ra, rb) = (to_rns::<$base, $n>(&a, k), to_rns::<$base, $n>(&b, p));
                    let expected = matmul_oracle(&a, &b, k, p);
                    for level in simd_levels() {
                        set_simd_level(level);
                        let c = ra.checked_matmul(&rb).expect("sized to fit");
                        let twice = c.as_vec() + c.as_vec();
                        for (r, row) in expected.iter().enumerate() {
                            for (j, v) in row.iter().enumerate() {
                                prop_assert_eq!(&to_big(&c.get(r, j).unwrap()), v, "level {}, entry ({}, {})", level, r, j);
                                let t = twice.get(r * p + j).unwrap();
                                prop_assert_eq!(to_big(&t), v * 2u32, "level {}, doubled entry ({}, {})", level, r, j);
                            }
                        }
                    }
                    set_simd_level(0);
                }

                #[test]
                fn determinant_matches_bareiss(n in 0usize..=8, seed in any::<u64>()) {
                    let lg = if n <= 1 { 0 } else { usize::BITS - (n - 1).leading_zeros() };
                    let room = maxb().saturating_sub((n as u32 * lg).div_ceil(2));
                    let bits = if n == 0 { 0 } else { (room / n as u32).min(62) };
                    let mut s = seed | 1;
                    let mut next = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
                    let a: Mat = (0..n).map(|_| (0..n).map(|_| {
                        if bits == 0 { return BigInt::zero(); }
                        let v = (next() >> (64 - bits)) as i64;
                        BigInt::from(if next() & 1 == 1 { -v } else { v })
                    }).collect()).collect();
                    let ra = to_rns::<$base, $n>(&a, n);
                    let det = ra.determinant().expect("Hadamard bound fits by construction");
                    prop_assert_eq!(to_big(&det), det_oracle(&a));
                }
            }

            #[test]
            fn identity_and_shapes() {
                let a = M::from_i128s(2, 3, &[1, -2, 3, 4, 5, -6]).unwrap();
                assert_eq!(&M::identity(2) * &a, a);
                assert_eq!(&a * &M::identity(3), a);
                let at = a.transpose();
                assert_eq!((at.rows(), at.cols()), (3, 2));
                assert_eq!(at.get(2, 1).unwrap().to_i128(), Some(-6));
                let g = &a * &at;
                let got: Vec<i128> = g.as_vec().iter().map(|x| x.to_i128().unwrap()).collect();
                assert_eq!(got, vec![14, -24, -24, 77]);
                assert!(M::zeros(0, 0).determinant().unwrap() == Rns::one());
            }

            #[test]
            fn overflow_is_refused() {
                let top = (1i128 << (maxb() / 2 + 1).min(126)) - 1;
                if maxb() / 2 < 126 {
                    let a = M::from_i128s(1, 2, &[top, top]).unwrap();
                    let b = M::from_i128s(2, 1, &[top, top]).unwrap();
                    assert!(a.checked_matmul(&b).is_none());
                }
            }
        }
    };
}

rns_base!(B2: largest 2);
rns_base!(B4: largest 4);
rns_base!(B9: largest 9);
rns_base!(B16: largest 16);
rns_base!(Small: 5 = [3, 5, 7, 11, 13]);
rns_base!(Low: 5 = [1_000_000_007, 998_244_353, 1_000_000_009, 754_974_721, 167_772_161]);

matrix_tests!(base2, B2, 2);
matrix_tests!(base4, B4, 4);
matrix_tests!(base9, B9, 9);
matrix_tests!(base16, B16, 16);
matrix_tests!(small, Small, 5);
matrix_tests!(low, Low, 5);

#[test]
fn determinant_special_cases() {
    type M = RnsMatrix<B9, 9>;
    let s = M::from_i128s(3, 3, &[1, 2, 3, 4, 5, 6, 1, 2, 3]).unwrap();
    assert!(s.determinant().unwrap().is_zero());
    let p = M::from_i128s(2, 2, &[0, 1, 1, 0]).unwrap();
    assert_eq!(p.determinant().unwrap().to_i128(), Some(-1));
    let c = M::from_i128s(4, 4, &[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0]).unwrap();
    assert_eq!(c.determinant().unwrap().to_i128(), Some(-1));
    let t = M::from_i128s(3, 3, &[-7, 100, 3, 0, 11, -5, 0, 0, 13]).unwrap();
    assert_eq!(t.determinant().unwrap().to_i128(), Some(-7 * 11 * 13));
}

/// det ≡ 0 modulo some of the base's primes while det ≠ 0: those channels see a
/// singular matrix, the others do not, and CRT must still give the exact value.
#[test]
fn determinant_divisible_by_moduli() {
    type M = RnsMatrix<Small, 5>;
    let a = M::from_i128s(3, 3, &[3, 0, 0, 0, 5, 0, 1, 2, 7]).unwrap();
    assert_eq!(a.determinant().unwrap().to_i128(), Some(105));
    type W = RnsMatrix<B4, 4>;
    let p = B4::base().moduli()[2] as i128;
    let b = W::from_i128s(2, 2, &[p, 1, 0, -3]).unwrap();
    assert_eq!(b.determinant().unwrap().to_i128(), Some(-3 * p));
}

#[test]
fn determinant_refuses_when_hadamard_exceeds_range() {
    type M = RnsMatrix<B2, 2>;
    let a = M::from_i128s(2, 2, &[1 << 40, 1, 1, 1 << 40]).unwrap();
    assert!(a.determinant().is_none());
    assert!(a.determinant_bits_bound() > Rns::<B2, 2>::max_bits());
}

/// Entries straight from an RnsVec, and element-wise ops through as_vec.
#[test]
fn from_vec_roundtrip() {
    let v = RnsVec::<B4, 4>::from_i128s(&[1, 2, 3, 4]).unwrap();
    let m = RnsMatrix::from_vec(2, 2, v.clone());
    assert_eq!(m.determinant().unwrap().to_i128(), Some(-2));
    assert_eq!(m.into_vec(), v);
}

/// Every residue at its maximum (`-1` is `m_i - 1` in every channel), with an
/// inner dimension of 61: the four-product sums of the SIMD kernels reach
/// their largest values, and the result must still be exact.
#[test]
fn product_of_maximal_residues() {
    rns_base!(B9: largest 9);
    let (m, k, p) = (19, 61, 35);
    let a = RnsMatrix::<B9, 9>::from_i128s(m, k, &vec![-1; m * k]).unwrap();
    let b = RnsMatrix::<B9, 9>::from_i128s(k, p, &vec![-1; k * p]).unwrap();
    let c = &a * &b;
    for r in 0..m {
        for j in 0..p {
            assert_eq!(c.get(r, j).unwrap().to_i128(), Some(k as i128));
        }
    }
}
