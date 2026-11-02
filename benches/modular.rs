//! Arithmetic modulo a large odd number: residue's RNS Montgomery against GMP
//! (through `rug`) and num-bigint.
//!
//! Three scenarios for moduli of 1024 and 2048 bits: one modular product, one
//! modular exponentiation, and a batch of 256 exponentiations with the same
//! 256-bit exponent (many operations with one key). Data is already in each
//! library's own representation. GMP gets preallocated integers, in-place
//! assignment and `pow_mod`; num-bigint gets `modpow`.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use num_bigint::BigUint;
use residue::{rns_base, ModVec, MontRns, RnsBase};
use rug::integer::Order;
use rug::{Assign, Integer};

rns_base!(K1: largest 35);
rns_base!(K2: largest 35 after 35);
rns_base!(T1: largest 70);
rns_base!(T2: largest 70 after 70);

const BATCH: usize = 256;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn limbs(&mut self, n: usize) -> Vec<u64> {
        (0..n).map(|_| self.next()).collect()
    }
}

fn scenario<B1: RnsBase<N1>, const N1: usize, B2: RnsBase<N2>, const N2: usize>(
    c: &mut Criterion,
    bits: usize,
) {
    let limbs = bits / 64;
    let mut r = Rng(bits as u64);
    let mut ml = r.limbs(limbs);
    ml[0] |= 1;
    ml[limbs - 1] |= 1 << 63;
    let a = r.limbs(limbs);
    let b = r.limbs(limbs);
    let e_full = r.limbs(limbs);
    let e256 = r.limbs(4);
    let xs: Vec<Vec<u64>> = (0..BATCH).map(|_| r.limbs(limbs)).collect();

    let ctx = MontRns::<B1, N1, B2, N2>::new(&ml).unwrap();
    let (ra, rb) = (ctx.from_limbs(&a), ctx.from_limbs(&b));
    let gm = Integer::from_digits(&ml, Order::Lsf);
    let (ga, gb) = (
        Integer::from_digits(&a, Order::Lsf) % &gm,
        Integer::from_digits(&b, Order::Lsf) % &gm,
    );
    let to_big = |l: &[u64]| {
        BigUint::from_slice(
            &l.iter()
                .flat_map(|x| [*x as u32, (*x >> 32) as u32])
                .collect::<Vec<u32>>(),
        )
    };
    let nm = to_big(&ml);
    let (na, nb) = (to_big(&a) % &nm, to_big(&b) % &nm);

    let mut g = c.benchmark_group(format!("M_modmul_{bits}"));
    g.bench_function("residue", |bch| bch.iter(|| black_box(ra) * black_box(rb)));
    let mut out = Integer::with_capacity(2 * bits);
    g.bench_function("gmp", |bch| {
        bch.iter(|| {
            out.assign(black_box(&ga) * black_box(&gb));
            out %= &gm;
        })
    });
    g.bench_function("num_bigint", |bch| {
        bch.iter(|| (black_box(&na) * black_box(&nb)) % &nm)
    });
    g.finish();

    let mut g = c.benchmark_group(format!("M_modexp_{bits}"));
    g.sample_size(10);
    g.bench_function("residue", |bch| {
        bch.iter(|| black_box(ra).pow(black_box(&e_full)))
    });
    let ge = Integer::from_digits(&e_full, Order::Lsf);
    g.bench_function("gmp", |bch| {
        bch.iter(|| {
            out.assign(black_box(&ga).pow_mod_ref(&ge, &gm).unwrap());
        })
    });
    let ne = to_big(&e_full);
    g.bench_function("num_bigint", |bch| {
        bch.iter(|| black_box(&na).modpow(&ne, &nm))
    });
    g.finish();

    let mut g = c.benchmark_group(format!("M_batch_modexp_{bits}"));
    g.sample_size(10);
    g.throughput(Throughput::Elements(BATCH as u64));
    let elems: Vec<_> = xs.iter().map(|x| ctx.from_limbs(x)).collect();
    let v = ModVec::from_elems(&ctx, &elems);
    g.bench_function("residue_batch", |bch| bch.iter(|| black_box(&v).pow(&e256)));
    for (level, label) in [
        (0u8, "one_batch_product"),
        (2, "one_batch_product_portable"),
    ] {
        g.bench_function(label, |bch| {
            residue::__set_simd_level(level);
            bch.iter(|| black_box(&v) * black_box(&v));
            residue::__set_simd_level(0);
        });
    }
    g.bench_function("residue_scalar", |bch| {
        bch.iter(|| {
            black_box(&elems)
                .iter()
                .map(|x| x.pow(&e256))
                .collect::<Vec<_>>()
        })
    });
    let gx: Vec<Integer> = xs
        .iter()
        .map(|x| Integer::from_digits(x, Order::Lsf) % &gm)
        .collect();
    let ge256 = Integer::from_digits(&e256, Order::Lsf);
    let mut gout: Vec<Integer> = (0..BATCH).map(|_| Integer::with_capacity(bits)).collect();
    g.bench_function("gmp", |bch| {
        bch.iter(|| {
            for (o, x) in gout.iter_mut().zip(black_box(&gx)) {
                o.assign(x.pow_mod_ref(&ge256, &gm).unwrap());
            }
        })
    });
    let nx: Vec<BigUint> = xs.iter().map(|x| to_big(x) % &nm).collect();
    let ne256 = to_big(&e256);
    g.bench_function("num_bigint", |bch| {
        bch.iter(|| {
            black_box(&nx)
                .iter()
                .map(|x| x.modpow(&ne256, &nm))
                .collect::<Vec<_>>()
        })
    });
    g.finish();

    let check = v.pow(&e256);
    for k in [0, BATCH / 2, BATCH - 1] {
        let want = gx[k].clone().pow_mod(&ge256, &gm).unwrap();
        let got = Integer::from_digits(&check.get(k).unwrap().to_limbs(), Order::Lsf);
        assert_eq!(got, want, "batch element {k}");
    }
    let want = ga.clone().pow_mod(&ge, &gm).unwrap();
    assert_eq!(
        Integer::from_digits(&ra.pow(&e_full).to_limbs(), Order::Lsf),
        want
    );
}

fn modular(c: &mut Criterion) {
    scenario::<K1, 35, K2, 35>(c, 1024);
    scenario::<T1, 70, T2, 70>(c, 2048);
}

fn config() -> Criterion {
    Criterion::default()
        .sample_size(20)
        .warm_up_time(std::time::Duration::from_millis(500))
        .measurement_time(std::time::Duration::from_secs(3))
}

criterion_group! {
    name = benches;
    config = config();
    targets = modular
}
criterion_main!(benches);
