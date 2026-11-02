//! Phase 0 experiment: would primes wider than 31 bits make residue faster?
//!
//! Two scenarios from the README, timed per channel kernel only (data already
//! in RNS, no reconstruction), exactly as the `residue` column of
//! `benches/compare.rs` measures them:
//!
//! - F: product of two 64×64 matrices with 62-bit entries (130-bit results);
//! - H: determinant of a 64×64 matrix with 16-bit entries (Hadamard: 1216 bits).
//!
//! Each candidate uses the fewest primes of its width that cover the range.
//! Every result is recombined with the CRT and checked against an independent
//! computation before anything is timed.
//!
//! Run with `cargo run --release` (portable build, runtime dispatch in the
//! library only), or with `RUSTFLAGS="-C target-cpu=native"` /
//! `"-C target-cpu=x86-64-v3"` to give every candidate the same instruction set.

mod ch50;
mod ch62;
mod kernels;

use std::hint::black_box;
use std::time::Instant;

use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};
use residue::{rns_base, Channel, RnsMatrix};

use ch50::Ch50;
use ch62::Ch62;

rns_base!(B5: largest 5);
rns_base!(B40: largest 40);

const D: usize = 64;

/// Deterministic SplitMix64, same generator and seeds as `benches/compare.rs`.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
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

/// Median time in nanoseconds of `runs` calls, after a short warm-up.
fn time<R>(runs: usize, mut f: impl FnMut() -> R) -> f64 {
    for _ in 0..3 {
        black_box(f());
    }
    let mut t: Vec<f64> = (0..runs)
        .map(|_| {
            let s = Instant::now();
            black_box(f());
            s.elapsed().as_nanos() as f64
        })
        .collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t[runs / 2]
}

fn reduce(x: i128, p: u64) -> u64 {
    x.rem_euclid(p as i128) as u64
}

/// Symmetric CRT: the integer in (−M/2, M/2) with the given residues.
fn crt(res: &[u64], primes: &[u64]) -> BigInt {
    let m: BigInt = primes.iter().map(|&p| BigInt::from(p)).product();
    let mut x = BigInt::zero();
    for (&r, &p) in res.iter().zip(primes) {
        let pi = BigInt::from(p);
        let mi = &m / &pi;
        let inv = (&mi % &pi).modpow(&(&pi - 2u32), &pi);
        x += BigInt::from(r) * &mi * inv;
    }
    x %= &m;
    if &x * 2u32 > m {
        x -= &m;
    }
    x
}

fn ms(ns: f64) -> String {
    format!("{:7.3} ms", ns / 1e6)
}

fn main() {
    let isa = if cfg!(target_feature = "avx512f") {
        "AVX-512 (compile time)"
    } else if cfg!(target_feature = "avx2") {
        "AVX2 (compile time)"
    } else {
        "baseline (library dispatches at runtime)"
    };
    println!("instruction set for the experiment kernels: {isa}\n");

    let p31_5: Vec<u64> = residue::primes::largest_primes::<5>()
        .iter()
        .map(|&p| p as u64)
        .collect();
    let p31_40: Vec<u64> = residue::primes::largest_primes::<40>()
        .iter()
        .map(|&p| p as u64)
        .collect();
    let p62_3 = ch62::largest_primes_below(1 << 62, 3);
    let p62_20 = ch62::largest_primes_below(1 << 62, 20);
    let p50_3 = ch62::largest_primes_below(1 << 50, 3);
    let p50_25 = ch62::largest_primes_below(1 << 50, 25);

    scenario_f(&p31_5, &p62_3, &p50_3);
    scenario_h(&p31_40, &p62_20, &p50_25);
}

fn scenario_f(p31: &[u64], p62: &[u64], p50: &[u64]) {
    let (xa, xb) = (values(11, D * D, 62), values(12, D * D, 62));

    let ba: Vec<BigInt> = xa.iter().map(|&x| BigInt::from(x)).collect();
    let bb: Vec<BigInt> = xb.iter().map(|&x| BigInt::from(x)).collect();
    let mut exact = vec![BigInt::zero(); D * D];
    for i in 0..D {
        for k in 0..D {
            for j in 0..D {
                exact[i * D + j] += &ba[i * D + k] * &bb[k * D + j];
            }
        }
    }
    let check = |name: &str, res: &[Vec<u64>], primes: &[u64]| {
        for e in 0..D * D {
            let r: Vec<u64> = res.iter().map(|c| c[e]).collect();
            assert_eq!(crt(&r, primes), exact[e], "{name}: entry {e} wrong");
        }
    };

    println!("F. Matrix product 64×64, 62-bit entries (130-bit results)");
    let mut rows: Vec<(String, f64)> = Vec::new();

    let (ma, mb) = (
        RnsMatrix::<B5, 5>::from_i128s(D, D, &xa).unwrap(),
        RnsMatrix::<B5, 5>::from_i128s(D, D, &xb).unwrap(),
    );
    let t = time(41, || {
        black_box(&ma).checked_matmul(black_box(&mb)).unwrap()
    });
    rows.push(("31 bit, 5 ch, library (with mod 2^64 channel)".into(), t));

    let ch31: Vec<Channel> = p31.iter().map(|&p| Channel::new(p as u32)).collect();
    let mont31 = |xs: &[i128], c: &Channel| -> Vec<u32> {
        xs.iter()
            .map(|&x| c.to_mont(reduce(x, c.modulus() as u64) as u32))
            .collect()
    };
    let a31: Vec<Vec<u32>> = ch31.iter().map(|c| mont31(&xa, c)).collect();
    let b31: Vec<Vec<u32>> = ch31.iter().map(|c| mont31(&xb, c)).collect();
    let mut c31 = vec![vec![0u32; D * D]; ch31.len()];
    let to_plain31 = |c: &[Vec<u32>]| -> Vec<Vec<u64>> {
        c.iter()
            .zip(&ch31)
            .map(|(v, ch)| v.iter().map(|&x| ch.from_mont(x) as u64).collect())
            .collect()
    };

    kernels::run_gemm31_naive(&ch31, &a31, &b31, &mut c31, D);
    check("gemm31_naive", &to_plain31(&c31), p31);
    let t = time(41, || {
        kernels::run_gemm31_naive(&ch31, black_box(&a31), black_box(&b31), &mut c31, D)
    });
    rows.push(("31 bit, 5 ch, same loop as library".into(), t));

    {
        let ra: Vec<u64> = xa.iter().map(|&x| x as u64).collect();
        let rb: Vec<u64> = xb.iter().map(|&x| x as u64).collect();
        let mut rc = vec![0u64; D * D];
        kernels::library_body(&ch31, &a31, &b31, &mut c31, &ra, &rb, &mut rc, D);
        check("library_body", &to_plain31(&c31), p31);
        let t = time(41, || {
            kernels::library_body(
                &ch31,
                black_box(&a31),
                black_box(&b31),
                &mut c31,
                &ra,
                &rb,
                &mut rc,
                D,
            )
        });
        rows.push(("31 bit, 5 ch, copy of library body + mod 2^64".into(), t));
    }
    c31.iter_mut().for_each(|c| c.fill(0));
    kernels::run_gemm31_interleaved(&ch31, &a31, &b31, &mut c31, D);
    check("gemm31_interleaved", &to_plain31(&c31), p31);
    let t = time(41, || {
        kernels::run_gemm31_interleaved(&ch31, black_box(&a31), black_box(&b31), &mut c31, D)
    });
    rows.push(("31 bit, 5 ch, same loop, lo/hi interleaved".into(), t));

    macro_rules! tiled {
        ($rb:literal, $cb:literal) => {
            c31.iter_mut().for_each(|c| c.fill(0));
            kernels::run_gemm31_tiled::<$rb, $cb>(&ch31, &a31, &b31, &mut c31);
            check("gemm31_tiled", &to_plain31(&c31), p31);
            let t = time(41, || {
                kernels::run_gemm31_tiled::<$rb, $cb>(
                    &ch31,
                    black_box(&a31),
                    black_box(&b31),
                    &mut c31,
                )
            });
            rows.push((format!("31 bit, 5 ch, register tiling {}×{}", $rb, $cb), t));
        };
    }
    tiled!(1, 16);
    tiled!(2, 16);
    tiled!(4, 8);
    tiled!(4, 16);
    tiled!(8, 8);

    let ra: Vec<u64> = xa.iter().map(|&x| x as u64).collect();
    let rb: Vec<u64> = xb.iter().map(|&x| x as u64).collect();
    let mut rc = vec![0u64; D * D];
    kernels::red_gemm_u64(&ra, &rb, &mut rc, D);
    let expect_red = rc.clone();
    let t = time(41, || {
        kernels::red_gemm_u64(black_box(&ra), black_box(&rb), &mut rc, D)
    });
    rows.push(("  mod 2^64 channel alone, u64 products (library)".into(), t));
    let split = |v: &[u64]| -> (Vec<u32>, Vec<u32>) {
        (
            v.iter().map(|&x| x as u32).collect(),
            v.iter().map(|&x| (x >> 32) as u32).collect(),
        )
    };
    let ((a0, a1), (b0, b1)) = (split(&ra), split(&rb));
    let mut s1 = vec![0u32; D];
    kernels::red_gemm_split(&a0, &a1, &b0, &b1, &mut rc, &mut s1, D);
    assert_eq!(rc, expect_red, "split redundant channel wrong");
    let t = time(41, || {
        kernels::red_gemm_split(
            black_box(&a0),
            black_box(&a1),
            black_box(&b0),
            black_box(&b1),
            &mut rc,
            &mut s1,
            D,
        )
    });
    rows.push(("  mod 2^64 channel alone, 32-bit pieces".into(), t));

    let ch62: Vec<Ch62> = p62.iter().map(|&p| Ch62::new(p)).collect();
    let a62: Vec<Vec<u64>> = ch62
        .iter()
        .map(|c| xa.iter().map(|&x| c.to_mont(reduce(x, c.p))).collect())
        .collect();
    let b62: Vec<Vec<u64>> = ch62
        .iter()
        .map(|c| xb.iter().map(|&x| c.to_mont(reduce(x, c.p))).collect())
        .collect();
    let mut c62 = vec![vec![0u64; D * D]; ch62.len()];
    let mut bt = vec![0u64; D * D];
    kernels::run_gemm62(&ch62, &a62, &b62, &mut c62, &mut bt, D);
    let plain62: Vec<Vec<u64>> = c62
        .iter()
        .zip(&ch62)
        .map(|(v, c)| v.iter().map(|&x| c.from_mont(x)).collect())
        .collect();
    check("gemm62", &plain62, p62);
    let t = time(41, || {
        kernels::run_gemm62(
            &ch62,
            black_box(&a62),
            black_box(&b62),
            &mut c62,
            &mut bt,
            D,
        )
    });
    rows.push(("62 bit, 3 ch, Montgomery u128 (incl. transpose)".into(), t));

    let ch50: Vec<Ch50> = p50.iter().map(|&p| Ch50::new(p)).collect();
    let a50: Vec<Vec<f64>> = ch50
        .iter()
        .map(|c| xa.iter().map(|&x| reduce(x, c.p_u64()) as f64).collect())
        .collect();
    let b50: Vec<Vec<f64>> = ch50
        .iter()
        .map(|c| xb.iter().map(|&x| reduce(x, c.p_u64()) as f64).collect())
        .collect();
    let mut c50 = vec![vec![0f64; D * D]; ch50.len()];
    kernels::run_gemm50(&ch50, &a50, &b50, &mut c50, D);
    let plain50: Vec<Vec<u64>> = c50
        .iter()
        .map(|v| v.iter().map(|&x| x as u64).collect())
        .collect();
    check("gemm50", &plain50, p50);
    let t = time(41, || {
        kernels::run_gemm50(&ch50, black_box(&a50), black_box(&b50), &mut c50, D)
    });
    rows.push(("50 bit, 3 ch, f64 FMA".into(), t));

    print_rows(&rows);
}

fn scenario_h(p31: &[u64], p62: &[u64], p50: &[u64]) {
    let xs = values(13 + D as u64, D * D, 16);
    let m = RnsMatrix::<B40, 40>::from_i128s(D, D, &xs).unwrap();
    let expect: BigInt = m.determinant().unwrap().to_string().parse().unwrap();
    assert!(expect.abs() > BigInt::one(), "degenerate test matrix");

    println!("H. Determinant 64×64, 16-bit entries (Hadamard bound 1216 bits)");
    let mut rows: Vec<(String, f64)> = Vec::new();

    let t = time(21, || black_box(&m).determinant().unwrap());
    rows.push(("31 bit, 40 ch, library (incl. CRT)".into(), t));

    let ch31: Vec<Channel> = p31.iter().map(|&p| Channel::new(p as u32)).collect();
    let a31: Vec<Vec<u32>> = ch31
        .iter()
        .map(|c| {
            xs.iter()
                .map(|&x| c.to_mont(reduce(x, c.modulus() as u64) as u32))
                .collect()
        })
        .collect();
    let mut work31 = vec![0u32; D * D];
    let d = kernels::run_det31(&ch31, &a31, &mut work31, D);
    let plain: Vec<u64> = d
        .iter()
        .zip(&ch31)
        .map(|(&x, c)| c.from_mont(x) as u64)
        .collect();
    assert_eq!(crt(&plain, p31), expect, "det31 wrong");
    let t = time(21, || {
        kernels::run_det31(&ch31, black_box(&a31), &mut work31, D)
    });
    rows.push(("31 bit, 40 ch, same loop as library".into(), t));

    let ch62: Vec<Ch62> = p62.iter().map(|&p| Ch62::new(p)).collect();
    let a62: Vec<Vec<u64>> = ch62
        .iter()
        .map(|c| xs.iter().map(|&x| c.to_mont(reduce(x, c.p))).collect())
        .collect();
    let mut work62 = vec![0u64; D * D];
    let d = kernels::run_det62(&ch62, &a62, &mut work62, D);
    let plain: Vec<u64> = d.iter().zip(&ch62).map(|(&x, c)| c.from_mont(x)).collect();
    assert_eq!(crt(&plain, p62), expect, "det62 wrong");
    let t = time(21, || {
        kernels::run_det62(&ch62, black_box(&a62), &mut work62, D)
    });
    rows.push(("62 bit, 20 ch, Montgomery u128".into(), t));

    let ch50: Vec<Ch50> = p50.iter().map(|&p| Ch50::new(p)).collect();
    let a50: Vec<Vec<f64>> = ch50
        .iter()
        .map(|c| xs.iter().map(|&x| reduce(x, c.p_u64()) as f64).collect())
        .collect();
    let mut work50 = vec![0f64; D * D];
    let d = kernels::run_det50(&ch50, &a50, &mut work50, D);
    let plain: Vec<u64> = d.iter().map(|&x| x as u64).collect();
    assert_eq!(crt(&plain, p50), expect, "det50 wrong");
    let t = time(21, || {
        kernels::run_det50(&ch50, black_box(&a50), &mut work50, D)
    });
    rows.push(("50 bit, 25 ch, f64 FMA".into(), t));

    print_rows(&rows);
}

fn print_rows(rows: &[(String, f64)]) {
    let base = rows[0].1;
    for (name, t) in rows {
        println!("  {name:50} {}   {:5.2}x vs library", ms(*t), base / t);
    }
    println!();
}
