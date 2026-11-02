//! Comparison against arbitrary-precision libraries.
//!
//! The competitors are `num-bigint`, the usual pure-Rust choice, and GMP through
//! the `rug` bindings. GMP is given its best case: preallocated values, in-place
//! `assign`, and the fused `addmul` and `submul` operations. Native `i128` is
//! included where the values are small enough for it.
//!
//! Each scenario is measured twice for every library. The plain measurement
//! (`residue`, `gmp`, `num_bigint`) starts from data already in that library's
//! own representation. The `_e2e` measurement starts from native integers and
//! ends with the result in binary, so each library also pays for its own
//! conversions. For GMP that means assigning `i128` values into preallocated
//! integers, which needs no allocation.
//!
//! Every input passes through `black_box`, so nothing can be precomputed or
//! hoisted out of the timed loop. `scripts/plot_benchmarks.py` reads the results
//! and draws the charts in `docs/`.

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use num_bigint::BigInt;
use residue::{rns_base, Rns, RnsBase, RnsMatrix, RnsVec};
use rug::integer::Order;
use rug::{Assign, Integer};

rns_base!(B4: largest 4);
rns_base!(B5: largest 5);
rns_base!(B9: largest 9);
rns_base!(B36: largest 36);
rns_base!(B40: largest 40);
rns_base!(B68: largest 68);

const N: usize = 4096;

/// Deterministic SplitMix64.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Signed integer with magnitude below 2^bits (bits <= 126).
    fn i128(&mut self, bits: u32) -> i128 {
        let mag = (((self.next() as u128) << 64) | self.next() as u128) >> (128 - bits);
        let v = mag as i128;
        if self.next() & 1 == 1 {
            -v
        } else {
            v
        }
    }
}

fn values(seed: u64, n: usize, bits: u32) -> Vec<i128> {
    let mut r = Rng(seed);
    (0..n).map(|_| r.i128(bits)).collect()
}

/// Read every element of an `RnsVec` back to binary, in two's complement.
fn read_back<B: RnsBase<NB>, const NB: usize>(v: &RnsVec<B, NB>) -> u64 {
    let w = v.twos_complement_width();
    let mut out = vec![0u64; v.len() * w];
    v.to_twos_complement(w, &mut out).unwrap();
    out.iter().fold(0, |acc, &x| acc ^ x)
}

fn elementwise_mul(c: &mut Criterion) {
    let (xa, xb) = (values(1, N, 126), values(2, N, 126));
    let mut g = c.benchmark_group("A_elementwise_mul");
    g.throughput(Throughput::Elements(N as u64));

    let (va, vb) = (
        RnsVec::<B9, 9>::from_i128s(&xa).unwrap(),
        RnsVec::<B9, 9>::from_i128s(&xb).unwrap(),
    );
    g.bench_function("residue", |b| {
        b.iter_batched_ref(
            || va.clone(),
            |out| out.mul_assign(black_box(&vb)),
            BatchSize::LargeInput,
        )
    });
    g.bench_function("residue_e2e", |b| {
        b.iter(|| {
            let mut a = RnsVec::<B9, 9>::from_i128s(black_box(&xa)).unwrap();
            let bb = RnsVec::<B9, 9>::from_i128s(black_box(&xb)).unwrap();
            a.mul_assign(&bb);
            read_back(&a)
        })
    });

    let (sa, sb): (Vec<Rns<B9, 9>>, Vec<Rns<B9, 9>>) = (
        xa.iter().map(|&x| Rns::from_i128(x).unwrap()).collect(),
        xb.iter().map(|&x| Rns::from_i128(x).unwrap()).collect(),
    );
    let mut sout = sa.clone();
    g.bench_function("residue_scalar", |b| {
        b.iter(|| {
            for ((o, x), y) in sout.iter_mut().zip(black_box(&sa)).zip(black_box(&sb)) {
                *o = *x * *y;
            }
        })
    });

    let ga: Vec<Integer> = xa.iter().map(|&x| Integer::from(x)).collect();
    let gb: Vec<Integer> = xb.iter().map(|&x| Integer::from(x)).collect();
    let mut gout: Vec<Integer> = (0..N).map(|_| Integer::with_capacity(256)).collect();
    g.bench_function("gmp", |b| {
        b.iter(|| {
            for ((o, x), y) in gout.iter_mut().zip(black_box(&ga)).zip(black_box(&gb)) {
                o.assign(x * y);
            }
        })
    });
    let (mut ta, mut tb) = (Integer::with_capacity(128), Integer::with_capacity(128));
    g.bench_function("gmp_e2e", |b| {
        b.iter(|| {
            for ((o, &x), &y) in gout.iter_mut().zip(black_box(&xa)).zip(black_box(&xb)) {
                ta.assign(x);
                tb.assign(y);
                o.assign(&ta * &tb);
            }
        })
    });

    let na: Vec<BigInt> = xa.iter().map(|&x| BigInt::from(x)).collect();
    let nb: Vec<BigInt> = xb.iter().map(|&x| BigInt::from(x)).collect();
    let mut nout: Vec<BigInt> = vec![BigInt::default(); N];
    g.bench_function("num_bigint", |b| {
        b.iter(|| {
            for ((o, x), y) in nout.iter_mut().zip(black_box(&na)).zip(black_box(&nb)) {
                *o = x * y;
            }
        })
    });
    g.bench_function("num_bigint_e2e", |b| {
        b.iter(|| {
            for ((o, &x), &y) in nout.iter_mut().zip(black_box(&xa)).zip(black_box(&xb)) {
                *o = BigInt::from(x) * BigInt::from(y);
            }
        })
    });
    g.finish();
}

fn dot_product(c: &mut Criterion) {
    let (xa, xb) = (values(3, N, 100), values(4, N, 100));
    let mut g = c.benchmark_group("B_dot_product");
    g.throughput(Throughput::Elements(N as u64));

    let (va, vb) = (
        RnsVec::<B9, 9>::from_i128s(&xa).unwrap(),
        RnsVec::<B9, 9>::from_i128s(&xb).unwrap(),
    );
    g.bench_function("residue", |b| {
        b.iter(|| black_box(&va).dot(black_box(&vb)).unwrap())
    });
    g.bench_function("residue_e2e", |b| {
        b.iter(|| {
            let a = RnsVec::<B9, 9>::from_i128s(black_box(&xa)).unwrap();
            let bb = RnsVec::<B9, 9>::from_i128s(black_box(&xb)).unwrap();
            a.dot(&bb).unwrap().to_sign_magnitude()
        })
    });

    let ga: Vec<Integer> = xa.iter().map(|&x| Integer::from(x)).collect();
    let gb: Vec<Integer> = xb.iter().map(|&x| Integer::from(x)).collect();
    g.bench_function("gmp", |b| {
        b.iter(|| {
            let mut acc = Integer::with_capacity(256);
            for (x, y) in black_box(&ga).iter().zip(black_box(&gb)) {
                acc += x * y;
            }
            acc
        })
    });
    let (mut ta, mut tb) = (Integer::with_capacity(128), Integer::with_capacity(128));
    g.bench_function("gmp_e2e", |b| {
        b.iter(|| {
            let mut acc = Integer::with_capacity(256);
            for (&x, &y) in black_box(&xa).iter().zip(black_box(&xb)) {
                ta.assign(x);
                tb.assign(y);
                acc += &ta * &tb;
            }
            acc
        })
    });

    let na: Vec<BigInt> = xa.iter().map(|&x| BigInt::from(x)).collect();
    let nb: Vec<BigInt> = xb.iter().map(|&x| BigInt::from(x)).collect();
    g.bench_function("num_bigint", |b| {
        b.iter(|| {
            let mut acc = BigInt::default();
            for (x, y) in black_box(&na).iter().zip(black_box(&nb)) {
                acc += x * y;
            }
            acc
        })
    });
    g.bench_function("num_bigint_e2e", |b| {
        b.iter(|| {
            let mut acc = BigInt::default();
            for (&x, &y) in black_box(&xa).iter().zip(black_box(&xb)) {
                acc += BigInt::from(x) * BigInt::from(y);
            }
            acc
        })
    });
    g.finish();
}

fn small_dot(c: &mut Criterion) {
    let (xa, xb) = (values(5, N, 50), values(6, N, 50));
    let mut g = c.benchmark_group("C_small_dot");
    g.throughput(Throughput::Elements(N as u64));

    g.bench_function("native", |b| {
        b.iter(|| {
            black_box(&xa)
                .iter()
                .zip(black_box(&xb))
                .fold(0i128, |acc, (&x, &y)| acc + x * y)
        })
    });
    let (va, vb) = (
        RnsVec::<B4, 4>::from_i128s(&xa).unwrap(),
        RnsVec::<B4, 4>::from_i128s(&xb).unwrap(),
    );
    g.bench_function("residue", |b| {
        b.iter(|| black_box(&va).dot(black_box(&vb)).unwrap())
    });
    g.bench_function("residue_e2e", |b| {
        b.iter(|| {
            let a = RnsVec::<B4, 4>::from_i128s(black_box(&xa)).unwrap();
            let bb = RnsVec::<B4, 4>::from_i128s(black_box(&xb)).unwrap();
            a.dot(&bb).unwrap().to_i128()
        })
    });
    g.finish();
}

fn conversions(c: &mut Criterion) {
    let x100 = values(7, N, 100);
    let mut g = c.benchmark_group("D_conversion");
    g.throughput(Throughput::Elements(N as u64));

    g.bench_function("to_rns_9ch", |b| {
        b.iter(|| RnsVec::<B9, 9>::from_i128s(black_box(&x100)).unwrap())
    });
    let v9 = RnsVec::<B9, 9>::from_i128s(&x100).unwrap();
    let p9 = &v9 * &v9;
    g.bench_function("from_rns_9ch", |b| b.iter(|| read_back(black_box(&p9))));
    g.finish();
}

fn single_large_mul(c: &mut Criterion) {
    let mut r = Rng(9);
    let la: Vec<u64> = (0..16).map(|_| r.next()).collect();
    let lb: Vec<u64> = (0..16).map(|_| r.next()).collect();
    let mut g = c.benchmark_group("E_single_mul_1024");

    let ga = Integer::from_digits(&la, Order::Lsf);
    let gb = Integer::from_digits(&lb, Order::Lsf);
    let mut gout = Integer::with_capacity(2048);
    g.bench_function("gmp", |b| {
        b.iter(|| gout.assign(black_box(&ga) * black_box(&gb)))
    });
    let (mut ta, mut tb) = (Integer::with_capacity(1024), Integer::with_capacity(1024));
    let mut digits = [0u64; 32];
    g.bench_function("gmp_e2e", |b| {
        b.iter(|| {
            ta.assign_digits(black_box(&la), Order::Lsf);
            tb.assign_digits(black_box(&lb), Order::Lsf);
            gout.assign(&ta * &tb);
            gout.write_digits(&mut digits, Order::Lsf);
            digits[0]
        })
    });

    type Z = Rns<B68, 68>;
    let (ra, rb) = (
        Z::from_sign_magnitude(false, &la).unwrap(),
        Z::from_sign_magnitude(false, &lb).unwrap(),
    );
    g.bench_function("residue", |b| b.iter(|| black_box(ra) * black_box(rb)));
    g.bench_function("residue_e2e", |b| {
        b.iter(|| {
            let x = Z::from_sign_magnitude(false, black_box(&la)).unwrap();
            let y = Z::from_sign_magnitude(false, black_box(&lb)).unwrap();
            (x * y).to_sign_magnitude()
        })
    });
    g.finish();
}

fn matmul(c: &mut Criterion) {
    const D: usize = 64;
    let (xa, xb) = (values(11, D * D, 62), values(12, D * D, 62));
    let mut g = c.benchmark_group("F_matmul_64");
    g.throughput(Throughput::Elements((D * D * D) as u64));

    let (ma, mb) = (
        RnsMatrix::<B5, 5>::from_i128s(D, D, &xa).unwrap(),
        RnsMatrix::<B5, 5>::from_i128s(D, D, &xb).unwrap(),
    );
    g.bench_function("residue", |b| {
        b.iter(|| black_box(&ma).checked_matmul(black_box(&mb)).unwrap())
    });
    g.bench_function("residue_e2e", |b| {
        b.iter(|| {
            let a = RnsMatrix::<B5, 5>::from_i128s(D, D, black_box(&xa)).unwrap();
            let bm = RnsMatrix::<B5, 5>::from_i128s(D, D, black_box(&xb)).unwrap();
            read_back(a.checked_matmul(&bm).unwrap().as_vec())
        })
    });

    fn gmp_matmul(a: &[Integer], b: &[Integer], out: &mut [Integer]) {
        for x in out.iter_mut() {
            x.assign(0);
        }
        for i in 0..D {
            for k in 0..D {
                let aik = &a[i * D + k];
                for j in 0..D {
                    out[i * D + j] += aik * &b[k * D + j];
                }
            }
        }
    }
    let ga: Vec<Integer> = xa.iter().map(|&x| Integer::from(x)).collect();
    let gb: Vec<Integer> = xb.iter().map(|&x| Integer::from(x)).collect();
    let mut gc: Vec<Integer> = (0..D * D).map(|_| Integer::with_capacity(192)).collect();
    g.bench_function("gmp", |b| {
        b.iter(|| gmp_matmul(black_box(&ga), black_box(&gb), &mut gc))
    });
    let mut wa = ga.clone();
    let mut wb = gb.clone();
    g.bench_function("gmp_e2e", |b| {
        b.iter(|| {
            for (w, &x) in wa.iter_mut().zip(black_box(&xa)) {
                w.assign(x);
            }
            for (w, &x) in wb.iter_mut().zip(black_box(&xb)) {
                w.assign(x);
            }
            gmp_matmul(&wa, &wb, &mut gc)
        })
    });

    fn num_matmul(a: &[BigInt], b: &[BigInt]) -> Vec<BigInt> {
        let mut out = vec![BigInt::default(); D * D];
        for i in 0..D {
            for k in 0..D {
                let aik = &a[i * D + k];
                for j in 0..D {
                    out[i * D + j] += aik * &b[k * D + j];
                }
            }
        }
        out
    }
    let na: Vec<BigInt> = xa.iter().map(|&x| BigInt::from(x)).collect();
    let nb: Vec<BigInt> = xb.iter().map(|&x| BigInt::from(x)).collect();
    g.bench_function("num_bigint", |b| {
        b.iter(|| num_matmul(black_box(&na), black_box(&nb)))
    });
    g.bench_function("num_bigint_e2e", |b| {
        b.iter(|| {
            let a: Vec<BigInt> = black_box(&xa).iter().map(|&x| BigInt::from(x)).collect();
            let bb: Vec<BigInt> = black_box(&xb).iter().map(|&x| BigInt::from(x)).collect();
            num_matmul(&a, &bb)
        })
    });
    g.finish();
}

fn bareiss_gmp(mut m: Vec<Integer>, n: usize) -> Integer {
    let mut prev = Integer::from(1);
    let mut sign = 1;
    let mut t = Integer::new();
    for k in 0..n - 1 {
        if m[k * n + k] == 0 {
            match (k + 1..n).find(|&r| m[r * n + k] != 0) {
                Some(r) => {
                    for j in 0..n {
                        m.swap(k * n + j, r * n + j);
                    }
                    sign = -sign;
                }
                None => return Integer::new(),
            }
        }
        for i in k + 1..n {
            for j in k + 1..n {
                t.assign(&m[k * n + k] * &m[i * n + j]);
                t -= &m[i * n + k] * &m[k * n + j];
                t.div_exact_mut(&prev);
                std::mem::swap(&mut m[i * n + j], &mut t);
            }
        }
        prev.assign(&m[k * n + k]);
    }
    let d = m.swap_remove(n * n - 1);
    if sign < 0 {
        -d
    } else {
        d
    }
}

fn bareiss_num(mut m: Vec<BigInt>, n: usize) -> BigInt {
    let mut prev = BigInt::from(1);
    let mut neg = false;
    for k in 0..n - 1 {
        if m[k * n + k] == BigInt::default() {
            match (k + 1..n).find(|&r| m[r * n + k] != BigInt::default()) {
                Some(r) => {
                    for j in 0..n {
                        m.swap(k * n + j, r * n + j);
                    }
                    neg = !neg;
                }
                None => return BigInt::default(),
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
    let d = m.swap_remove(n * n - 1);
    if neg {
        -d
    } else {
        d
    }
}

fn determinant<Bs: RnsBase<NB>, const NB: usize>(
    c: &mut Criterion,
    name: &str,
    n: usize,
    bits: u32,
) {
    let xs = values(13 + n as u64, n * n, bits);
    let mut g = c.benchmark_group(name);

    let rm = RnsMatrix::<Bs, NB>::from_i128s(n, n, &xs).unwrap();
    let expect = bareiss_num(xs.iter().map(|&x| BigInt::from(x)).collect(), n);
    assert_eq!(rm.determinant().unwrap().to_string(), expect.to_string());
    let gm: Vec<Integer> = xs.iter().map(|&x| Integer::from(x)).collect();
    assert_eq!(bareiss_gmp(gm.clone(), n).to_string(), expect.to_string());

    g.bench_function("residue", |b| {
        b.iter(|| black_box(&rm).determinant().unwrap())
    });
    g.bench_function("residue_e2e", |b| {
        b.iter(|| {
            let m = RnsMatrix::<Bs, NB>::from_i128s(n, n, black_box(&xs)).unwrap();
            m.determinant().unwrap().to_sign_magnitude()
        })
    });
    g.bench_function("gmp", |b| {
        b.iter_batched(|| gm.clone(), |m| bareiss_gmp(m, n), BatchSize::LargeInput)
    });
    g.bench_function("gmp_e2e", |b| {
        b.iter(|| {
            let m: Vec<Integer> = black_box(&xs).iter().map(|&x| Integer::from(x)).collect();
            bareiss_gmp(m, n)
        })
    });
    let nm: Vec<BigInt> = xs.iter().map(|&x| BigInt::from(x)).collect();
    g.bench_function("num_bigint", |b| {
        b.iter_batched(|| nm.clone(), |m| bareiss_num(m, n), BatchSize::LargeInput)
    });
    g.bench_function("num_bigint_e2e", |b| {
        b.iter(|| {
            let m: Vec<BigInt> = black_box(&xs).iter().map(|&x| BigInt::from(x)).collect();
            bareiss_num(m, n)
        })
    });
    g.finish();
}

fn determinants(c: &mut Criterion) {
    determinant::<B36, 36>(c, "G_det_32", 32, 32);
    determinant::<B40, 40>(c, "H_det_64", 64, 16);
}

fn config() -> Criterion {
    Criterion::default()
        .sample_size(20)
        .warm_up_time(std::time::Duration::from_millis(500))
        .measurement_time(std::time::Duration::from_secs(2))
}

criterion_group! {
    name = benches;
    config = config();
    targets = elementwise_mul, dot_product, small_dot, conversions, single_large_mul, matmul, determinants
}
criterion_main!(benches);
