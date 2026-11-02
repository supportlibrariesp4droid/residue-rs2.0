//! Micro-benchmarks of residue's own building blocks, without competitors.
//!
//! `compare.rs` answers "is residue faster than GMP or FLINT?"; this file answers
//! "did this change make residue faster?". It times the pieces that the
//! end-to-end scenarios are made of: conversions in both directions on bases of
//! several sizes, the sign test, and a scalar product. Save a baseline with
//! `cargo bench --bench internal -- --save-baseline <name>` and compare later
//! runs against it with `--baseline <name>`.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use residue::{rns_base, Rns, RnsBase, RnsMatrix, RnsVec};

rns_base!(B9: largest 9);
rns_base!(B36: largest 36);
rns_base!(B68: largest 68);

const N: usize = 1024;

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
}

/// `n` random values of `bits` bits (rounded down to whole limbs), random sign.
fn values<B: RnsBase<NB>, const NB: usize>(seed: u64, n: usize, bits: u32) -> Vec<Rns<B, NB>> {
    let mut r = Rng(seed);
    let limbs = (bits / 64) as usize;
    (0..n)
        .map(|_| {
            let mag: Vec<u64> = (0..limbs).map(|_| r.next()).collect();
            Rns::from_sign_magnitude(r.next() & 1 == 1, &mag).unwrap()
        })
        .collect()
}

fn to_binary<B: RnsBase<NB>, const NB: usize>(c: &mut Criterion, name: &str) {
    let bits = Rns::<B, NB>::max_bits() - 2;
    let xs = values::<B, NB>(1, N, bits);
    let v = RnsVec::<B, NB>::from_slice(&xs);
    let mut g = c.benchmark_group(name);
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function("to_sign_magnitude", |b| {
        b.iter(|| {
            black_box(&xs).iter().fold(0u64, |acc, x| {
                x.to_sign_magnitude().1.iter().fold(acc, |a, &l| a ^ l)
            })
        })
    });
    g.bench_function("vec_iter_read_back", |b| {
        b.iter(|| {
            black_box(&v).iter().fold(0u64, |acc, x| {
                x.to_sign_magnitude().1.iter().fold(acc, |a, &l| a ^ l)
            })
        })
    });
    let w = v.twos_complement_width();
    let mut out = vec![0u64; N * w];
    g.bench_function("vec_to_twos_complement", |b| {
        b.iter(|| {
            black_box(&v).to_twos_complement(w, &mut out).unwrap();
            black_box(&mut out);
        })
    });
    g.bench_function("vec_to_twos_complement_portable", |b| {
        residue::__set_simd_level(2);
        b.iter(|| {
            black_box(&v).to_twos_complement(w, &mut out).unwrap();
            black_box(&mut out);
        });
        residue::__set_simd_level(0);
    });
    g.bench_function("signum", |b| {
        b.iter(|| black_box(&xs).iter().map(|x| x.signum()).sum::<i32>())
    });
    g.finish();
}

fn from_binary<B: RnsBase<NB>, const NB: usize>(c: &mut Criterion, name: &str) {
    let bits = Rns::<B, NB>::max_bits() - 2;
    let limbs = (bits / 64) as usize;
    let mut r = Rng(2);
    let mags: Vec<Vec<u64>> = (0..N)
        .map(|_| (0..limbs).map(|_| r.next()).collect())
        .collect();
    let mut g = c.benchmark_group(name);
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function("from_sign_magnitude", |b| {
        b.iter(|| {
            black_box(&mags)
                .iter()
                .map(|m| Rns::<B, NB>::from_sign_magnitude(false, m).unwrap())
                .fold(0u32, |acc, x| x.residues().iter().fold(acc, |a, &r| a ^ r))
        })
    });
    let input: Vec<u64> = mags
        .iter()
        .flat_map(|m| {
            let mut l = m.clone();
            l.push(0);
            l
        })
        .collect();
    g.bench_function("vec_from_twos_complement", |b| {
        b.iter(|| RnsVec::<B, NB>::from_twos_complement(limbs + 1, black_box(&input)).unwrap())
    });
    g.finish();
}

/// Values that fit `i128` on a wide base: products of 60-bit numbers.
fn small_values_on_wide_base<B: RnsBase<NB>, const NB: usize>(c: &mut Criterion, name: &str) {
    let mut r = Rng(3);
    let xs: Vec<i128> = (0..N).map(|_| (r.next() >> 4) as i64 as i128).collect();
    let v = RnsVec::<B, NB>::from_i128s(&xs).unwrap();
    let p = &v * &v;
    let mut g = c.benchmark_group(name);
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function("scalar_to_i128", |b| {
        b.iter(|| {
            black_box(&p)
                .iter()
                .map(|x| x.to_i128().unwrap())
                .fold(0i128, |a, x| a ^ x)
        })
    });
    g.bench_function("vec_to_i128s", |b| {
        b.iter(|| black_box(&p).to_i128s().unwrap())
    });
    g.bench_function("vec_to_i128s_portable", |b| {
        residue::__set_simd_level(2);
        b.iter(|| black_box(&p).to_i128s().unwrap());
        residue::__set_simd_level(0);
    });
    g.finish();
}

/// The matrix product kernels side by side in one binary: explicit AVX-512,
/// explicit AVX2, and the portable auto-vectorized loop.
fn matmul_kernels(c: &mut Criterion) {
    rns_base!(B5: largest 5);
    rns_base!(B40: largest 40);
    fn run<B: RnsBase<NB>, const NB: usize>(c: &mut Criterion, name: &str, d: usize, bits: u32) {
        let mut r = Rng(11);
        let mut val = || ((r.next() >> (64 - bits)) as i64) as i128;
        let xa: Vec<i128> = (0..d * d).map(|_| val()).collect();
        let xb: Vec<i128> = (0..d * d).map(|_| val()).collect();
        let (a, b) = (
            RnsMatrix::<B, NB>::from_i128s(d, d, &xa).unwrap(),
            RnsMatrix::<B, NB>::from_i128s(d, d, &xb).unwrap(),
        );
        let mut g = c.benchmark_group(name);
        for (level, label) in [(0u8, "best"), (1, "avx2"), (2, "portable")] {
            g.bench_function(label, |bch| {
                residue::__set_simd_level(level);
                bch.iter(|| black_box(&a).checked_matmul(black_box(&b)).unwrap());
                residue::__set_simd_level(0);
            });
        }
        g.finish();
    }
    run::<B5, 5>(c, "matmul_64_5ch", 64, 62);
    run::<B40, 40>(c, "matmul_64_40ch", 64, 62);
    run::<B5, 5>(c, "matmul_256_5ch", 256, 60);
}

fn conversions(c: &mut Criterion) {
    to_binary::<B9, 9>(c, "to_binary_9ch");
    to_binary::<B36, 36>(c, "to_binary_36ch");
    to_binary::<B68, 68>(c, "to_binary_68ch");
    from_binary::<B9, 9>(c, "from_binary_9ch");
    from_binary::<B36, 36>(c, "from_binary_36ch");
    from_binary::<B68, 68>(c, "from_binary_68ch");
    small_values_on_wide_base::<B9, 9>(c, "to_i128_9ch");
    small_values_on_wide_base::<B36, 36>(c, "to_i128_36ch");
}

fn config() -> Criterion {
    Criterion::default()
        .sample_size(30)
        .warm_up_time(std::time::Duration::from_millis(300))
        .measurement_time(std::time::Duration::from_secs(1))
}

criterion_group! {
    name = benches;
    config = config();
    targets = conversions, matmul_kernels
}
criterion_main!(benches);
