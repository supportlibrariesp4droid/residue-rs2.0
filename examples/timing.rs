//! A small timing program with no dependencies beyond the library, for targets
//! where the criterion benchmarks do not run (WebAssembly, emulators):
//!
//! ```sh
//! cargo run --release --example timing
//! cargo build --release --example timing --target wasm32-wasip1
//! wasmtime target/wasm32-wasip1/release/examples/timing.wasm
//! RUSTFLAGS="-C target-feature=+simd128" cargo build --release --example timing --target wasm32-wasip1
//! ```
//!
//! Each line is the median of several runs. Results are folded into a
//! checksum, printed at the end, so that no work can be optimized away.

use std::hint::black_box;
use std::time::Instant;

use residue::{rns_base, ModVec, MontRns, RnsMatrix, RnsVec};

rns_base!(B5: largest 5);
rns_base!(B9: largest 9);
rns_base!(K1: largest 35);
rns_base!(K2: largest 35 after 35);

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
        ((self.next() as i128) << 64 | self.next() as i128) >> (128 - bits)
    }
}

/// Median time of `runs` calls of `f`, in microseconds; `f` returns a value
/// that is folded into `sum`.
fn time(name: &str, runs: usize, sum: &mut u64, mut f: impl FnMut() -> u64) {
    *sum = sum.wrapping_add(f());
    let mut t: Vec<f64> = (0..runs)
        .map(|_| {
            let s = Instant::now();
            let v = f();
            let e = s.elapsed().as_secs_f64() * 1e6;
            *sum = sum.wrapping_add(v);
            e
        })
        .collect();
    t.sort_by(f64::total_cmp);
    println!("{name:40} {:12.1} µs", t[t.len() / 2]);
}

fn main() {
    let mut r = Rng(7);
    let mut sum = 0u64;

    let xs: Vec<i128> = (0..4096).map(|_| r.i128(120)).collect();
    let ys: Vec<i128> = (0..4096).map(|_| r.i128(120)).collect();
    let a = RnsVec::<B9, 9>::from_i128s(&xs).unwrap();
    let b = RnsVec::<B9, 9>::from_i128s(&ys).unwrap();
    time("vec_mul_4096_9ch", 51, &mut sum, || {
        let mut c = black_box(&a).clone();
        c.mul_assign(black_box(&b));
        c.get(4095).unwrap().residues()[0] as u64
    });
    time("vec_dot_4096_9ch", 51, &mut sum, || {
        black_box(&a).dot(black_box(&b)).unwrap().residues()[0] as u64
    });
    time("vec_from_i128s_4096_9ch", 51, &mut sum, || {
        let v = RnsVec::<B9, 9>::from_i128s(black_box(&xs)).unwrap();
        v.get(4095).unwrap().residues()[8] as u64
    });
    time("vec_to_i128s_4096_9ch", 51, &mut sum, || {
        black_box(&a)
            .to_i128s()
            .unwrap()
            .iter()
            .fold(0u64, |s, &x| s ^ x as u64)
    });

    let m: Vec<i128> = (0..64 * 64).map(|_| r.i128(62)).collect();
    let ma = RnsMatrix::<B5, 5>::from_i128s(64, 64, &m).unwrap();
    time("matmul_64_5ch", 21, &mut sum, || {
        let c = black_box(&ma).checked_matmul(black_box(&ma)).unwrap();
        c.as_vec().get(0).unwrap().residues()[0] as u64
    });
    let d: Vec<i128> = (0..32 * 32).map(|_| r.i128(4)).collect();
    let md = RnsMatrix::<B9, 9>::from_i128s(32, 32, &d).unwrap();
    time("determinant_32_9ch", 21, &mut sum, || {
        black_box(&md).determinant().unwrap().residues()[0] as u64
    });

    let mut ml: Vec<u64> = (0..16).map(|_| r.next()).collect();
    ml[0] |= 1;
    ml[15] |= 1 << 63;
    let ctx = MontRns::<K1, 35, K2, 35>::new(&ml).unwrap();
    let elems: Vec<_> = (0..256)
        .map(|_| ctx.from_limbs(&(0..16).map(|_| r.next()).collect::<Vec<_>>()))
        .collect();
    let v = ModVec::from_elems(&ctx, &elems);
    time("modvec_product_1024bit_256", 21, &mut sum, || {
        (black_box(&v) * black_box(&v)).get(255).unwrap().to_limbs()[0]
    });
    time("modelem_product_1024bit", 201, &mut sum, || {
        (black_box(elems[0]) * black_box(elems[1])).to_limbs()[0]
    });

    println!("checksum {sum:016x}");
}
