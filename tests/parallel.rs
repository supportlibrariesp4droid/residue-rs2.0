//! The parallel code paths, reached on small inputs by lowering the threshold
//! to zero, inside thread pools of several sizes. Oracle: num-bigint.
//!
//! With 7 threads and 4 channels the matrix product splits rows into blocks,
//! and conversions to binary split the elements into ranges, so those paths
//! are covered too.
//!
//! Not built for WebAssembly without threads, where no pool can be created;
//! the other test files cover the `parallel` feature falling back to one
//! thread there.
#![cfg(all(
    feature = "parallel",
    feature = "num-bigint",
    not(all(target_family = "wasm", not(target_feature = "atomics")))
))]

use std::sync::OnceLock;

use num_bigint::{BigInt, BigUint};
use num_traits::{One, Zero};
use proptest::prelude::*;
use residue::{rns_base, Rns, RnsBase, RnsMatrix, RnsVec};

fn pool(threads: usize) -> &'static rayon::ThreadPool {
    static POOLS: OnceLock<Vec<rayon::ThreadPool>> = OnceLock::new();
    let pools = POOLS.get_or_init(|| {
        [2, 3, 7]
            .iter()
            .map(|&t| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(t)
                    .build()
                    .unwrap()
            })
            .collect()
    });
    &pools[[2, 3, 7].iter().position(|&t| t == threads).unwrap()]
}

/// Run `f` in every pool, with the parallel paths forced on.
fn in_pools(f: impl Fn() + Sync) {
    residue::__set_parallel_threshold(0);
    for t in [2, 3, 7] {
        pool(t).install(&f);
    }
}

fn limbs_to_biguint(limbs: &[u64]) -> BigUint {
    let bytes: Vec<u8> = limbs.iter().flat_map(|l| l.to_le_bytes()).collect();
    BigUint::from_bytes_le(&bytes)
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

fn big_of<B: RnsBase<N>, const N: usize>(x: &Rns<B, N>) -> BigInt {
    x.to_string().parse().unwrap()
}

/// Determinant by fraction-free Bareiss elimination.
fn bareiss(mut m: Vec<BigInt>, n: usize) -> BigInt {
    let mut prev = BigInt::one();
    let mut neg = false;
    for k in 0..n.saturating_sub(1) {
        if m[k * n + k].is_zero() {
            match (k + 1..n).find(|&r| !m[r * n + k].is_zero()) {
                Some(r) => {
                    for j in 0..n {
                        m.swap(k * n + j, r * n + j);
                    }
                    neg = !neg;
                }
                None => return BigInt::zero(),
            }
        }
        for i in k + 1..n {
            for j in k + 1..n {
                m[i * n + j] =
                    (&m[k * n + k] * &m[i * n + j] - &m[i * n + k] * &m[k * n + j]) / &prev;
            }
        }
        prev = m[k * n + k].clone();
    }
    let d = if n == 0 {
        BigInt::one()
    } else {
        m[n * n - 1].clone()
    };
    if neg {
        -d
    } else {
        d
    }
}

macro_rules! parallel_tests {
    ($modname:ident, $base:ident, $n:literal) => {
        mod $modname {
            use super::*;
            type Z = Rns<$base, $n>;
            type V = RnsVec<$base, $n>;
            type M = RnsMatrix<$base, $n>;

            proptest! {
                #![proptest_config(ProptestConfig::with_cases(24))]

                #[test]
                fn matrix_product(
                    (m, k, p) in (0usize..9, 0usize..9, 0usize..9),
                    seed in prop::collection::vec(signed(Z::max_bits() / 2 - 6), 162),
                ) {
                    let a: Vec<BigInt> = seed[..m * k].to_vec();
                    let b: Vec<BigInt> = seed[81..81 + k * p].to_vec();
                    let ma = M::from_vec(m, k, V::from_bigints(&a).unwrap());
                    let mb = M::from_vec(k, p, V::from_bigints(&b).unwrap());
                    in_pools(|| {
                        let c = ma.checked_matmul(&mb).unwrap();
                        for i in 0..m {
                            for j in 0..p {
                                let want: BigInt = (0..k).map(|t| &a[i * k + t] * &b[t * p + j]).sum();
                                assert_eq!(big_of(&c.get(i, j).unwrap()), want, "entry ({i}, {j})");
                            }
                        }
                    });
                }

                #[test]
                fn determinant(n in 0usize..7, seed in prop::collection::vec(signed(8), 36)) {
                    let xs: Vec<BigInt> = seed[..n * n].to_vec();
                    let m = M::from_vec(n, n, V::from_bigints(&xs).unwrap());
                    let want = bareiss(xs.clone(), n);
                    in_pools(|| {
                        if let Some(d) = m.determinant() {
                            assert_eq!(big_of(&d), want);
                        }
                    });
                }

                #[test]
                fn conversions(xs in prop::collection::vec(signed(Z::max_bits()), 0..400)) {
                    in_pools(|| {
                        let v = V::from_bigints(&xs).unwrap();
                        assert_eq!(v.to_bigints(), xs);
                        let w = v.twos_complement_width();
                        let mut out = vec![0u64; xs.len() * w];
                        v.to_twos_complement(w, &mut out).unwrap();
                        let back = V::from_twos_complement(w, &out).unwrap();
                        assert_eq!(back.to_bigints(), xs);
                        let as128: Option<Vec<i128>> = xs.iter().map(|x| i128::try_from(x).ok()).collect();
                        assert_eq!(v.to_i128s().ok(), as128.clone());
                        if let Some(small) = as128 {
                            if let Ok(u) = V::from_i128s(&small) {
                                assert_eq!(u.to_bigints(), xs);
                            }
                        }
                    });
                }

                #[test]
                fn dot_and_product(
                    pairs in prop::collection::vec((signed(Z::max_bits() / 2 - 6), signed(Z::max_bits() / 2 - 6)), 0..300),
                ) {
                    let (a, b): (Vec<BigInt>, Vec<BigInt>) = pairs.into_iter().unzip();
                    let va = V::from_bigints(&a).unwrap();
                    let vb = V::from_bigints(&b).unwrap();
                    in_pools(|| {
                        let want: BigInt = a.iter().zip(&b).map(|(x, y)| x * y).sum();
                        assert_eq!(big_of(&va.dot(&vb).unwrap()), want);
                        let mut p = va.clone();
                        p.checked_mul_assign(&vb).unwrap();
                        let want: Vec<BigInt> = a.iter().zip(&b).map(|(x, y)| x * y).collect();
                        assert_eq!(p.to_bigints(), want);
                    });
                }
            }
        }
    };
}

const fn three_and_large() -> [u32; 6] {
    let l = residue::primes::largest_primes::<5>();
    [3, l[0], l[1], l[2], l[3], l[4]]
}

rns_base!(B4: largest 4);
rns_base!(B9: largest 9);
rns_base!(B16: largest 16);
rns_base!(Small: 5 = [3, 5, 7, 11, 13]);
rns_base!(ThreeLarge: 6 = three_and_large());

parallel_tests!(base4, B4, 4);
parallel_tests!(base9, B9, 9);
parallel_tests!(base16, B16, 16);
parallel_tests!(small, Small, 5);
parallel_tests!(three_large, ThreeLarge, 6);

/// Large inputs at the default threshold: the path real programs take.
#[test]
fn large_inputs_default_threshold() {
    rns_base!(B5: largest 5);
    let n = 48;
    let xs: Vec<i128> = (0..n * n)
        .map(|i| (i as i128 * 7919 - 1_000_003) % 1_000_000_007)
        .collect();
    let a = RnsMatrix::<B5, 5>::from_i128s(n, n, &xs).unwrap();
    let seq = {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        pool.install(|| (&a * &a).into_vec().to_bigints())
    };
    let par = pool(7).install(|| (&a * &a).into_vec().to_bigints());
    assert_eq!(seq, par);
}
