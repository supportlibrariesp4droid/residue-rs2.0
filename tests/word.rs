//! Correctness of single-channel arithmetic vs a naive u64 reference.

use residue::primes::largest_primes;
use residue::Channel;

/// Deterministic SplitMix64, to avoid an external RNG dependency.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u32) -> u32 {
        (self.next() % n as u64) as u32
    }
}

/// A spread of moduli: big (near 2^31), medium, and tiny edge cases.
fn test_moduli() -> Vec<u32> {
    let mut m = largest_primes::<4>().to_vec();
    m.extend_from_slice(&[3, 5, 7, 97, 65_537, 1_000_003, 2_000_003]);
    m
}

#[test]
fn add_sub_neg_match_reference() {
    let mut rng = Rng(1);
    for &p in &test_moduli() {
        let ch = Channel::new(p);
        for _ in 0..50_000 {
            let a = rng.below(p);
            let b = rng.below(p);
            assert_eq!(ch.add(a, b), ((a as u64 + b as u64) % p as u64) as u32);
            assert_eq!(
                ch.sub(a, b),
                ((a as i64 - b as i64).rem_euclid(p as i64)) as u32
            );
            assert_eq!(ch.neg(a), ((p as u64 - a as u64) % p as u64) as u32);
        }
    }
}

#[test]
fn montgomery_roundtrip_and_mul() {
    let mut rng = Rng(2);
    for &p in &test_moduli() {
        let ch = Channel::new(p);
        for _ in 0..50_000 {
            let a = rng.below(p);
            let b = rng.below(p);
            assert_eq!(ch.from_mont(ch.to_mont(a)), a);
            let prod = ch.from_mont(ch.mul_mont(ch.to_mont(a), ch.to_mont(b)));
            assert_eq!(prod, ((a as u64 * b as u64) % p as u64) as u32);
        }
    }
}

#[test]
fn shoup_matches_reference() {
    let mut rng = Rng(3);
    for &p in &test_moduli() {
        let ch = Channel::new(p);
        for _ in 0..20_000 {
            let w = rng.below(p);
            let w_shoup = ch.shoup_factor(w);
            for _ in 0..8 {
                let a = rng.below(p);
                assert_eq!(
                    ch.mul_shoup(a, w, w_shoup),
                    ((a as u64 * w as u64) % p as u64) as u32
                );
            }
        }
    }
}

#[test]
fn boundary_values() {
    for &p in &test_moduli() {
        let ch = Channel::new(p);
        let edge = [0u32, 1, 2, p - 1, p - 2, p / 2];
        for &a in &edge {
            for &b in &edge {
                assert_eq!(ch.add(a, b), ((a as u64 + b as u64) % p as u64) as u32);
                assert_eq!(
                    ch.sub(a, b),
                    ((a as i64 - b as i64).rem_euclid(p as i64)) as u32
                );
                let prod = ch.from_mont(ch.mul_mont(ch.to_mont(a), ch.to_mont(b)));
                assert_eq!(prod, ((a as u64 * b as u64) % p as u64) as u32);
            }
        }
    }
}
