//! The fallback paths of delayed reduction.
//!
//! Conversions and base extension sum raw products and reduce once with
//! `Channel::redc_halves`, which needs fewer than `p` products in the sum.
//! With a prime as small as 3 that fails as soon as two limbs or two source
//! channels are summed, and `Channel::reduce_halves` takes a slower branch with
//! divisions. The bases of the other test files are either all large primes or
//! small enough to fit one limb, so they never reach it. Oracle: num-bigint.

use num_bigint::{BigInt, Sign};
use residue::{rns_base, Channel, Extender, Rns, RnsBase, RnsMod};

const fn three_and_large() -> [u32; 6] {
    let l = residue::primes::largest_primes::<5>();
    [3, l[0], l[1], l[2], l[3], l[4]]
}

rns_base!(ThreeLarge: 6 = three_and_large());
rns_base!(B9: largest 9);

type Z = Rns<ThreeLarge, 6>;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// A magnitude of a random bit length up to `max_bits`, and a sign.
    fn value(&mut self, max_bits: u32) -> (bool, Vec<u64>) {
        let bits = (self.next() % (max_bits as u64 + 1)) as u32;
        let mut limbs: Vec<u64> = (0..bits.div_ceil(64)).map(|_| self.next()).collect();
        if !bits.is_multiple_of(64) {
            *limbs.last_mut().unwrap() &= (1u64 << (bits % 64)) - 1;
        }
        (self.next() & 1 == 1, limbs)
    }
}

fn big(neg: bool, limbs: &[u64]) -> BigInt {
    let bytes: Vec<u8> = limbs.iter().flat_map(|l| l.to_le_bytes()).collect();
    BigInt::from_bytes_le(if neg { Sign::Minus } else { Sign::Plus }, &bytes)
}

#[test]
fn reduce_halves_matches_division() {
    for p in [3u32, 5, 7, 65_537, 2_147_483_647] {
        let ch = Channel::new(p);
        let mut r = Rng(p as u64);
        for _ in 0..10_000 {
            let (lo, hi) = (r.next(), r.next() >> (r.next() % 64));
            let t = ((hi as u128) << 32) + lo as u128;
            let rinv = BigInt::from(1u64 << 32).modpow(&BigInt::from(p - 2), &BigInt::from(p));
            let want = (BigInt::from(t) * rinv) % BigInt::from(p);
            assert_eq!(
                BigInt::from(ch.reduce_halves(lo, hi)),
                want,
                "p={p} lo={lo} hi={hi}"
            );
        }
    }
}

#[test]
fn conversion_round_trip_with_prime_three() {
    let mut r = Rng(1);
    for _ in 0..5_000 {
        let (neg, mag) = r.value(Z::max_bits());
        let x = Z::from_sign_magnitude(neg, &mag).unwrap();
        assert_eq!(x.to_string(), big(neg, &mag).to_string());
    }
}

#[test]
fn rns_mod_from_long_limbs_with_prime_three() {
    let m = BigInt::from_bytes_le(
        Sign::Plus,
        &<ThreeLarge as RnsBase<6>>::base()
            .modulus_limbs()
            .iter()
            .flat_map(|l| l.to_le_bytes())
            .collect::<Vec<u8>>(),
    );
    let mut r = Rng(2);
    for len in [1usize, 2, 3, 4, 7, 12] {
        for _ in 0..500 {
            let limbs: Vec<u64> = (0..len).map(|_| r.next()).collect();
            let x = RnsMod::<ThreeLarge, 6>::from_limbs(&limbs);
            let want = big(false, &limbs) % &m;
            assert_eq!(big(false, &x.to_limbs()), want, "len={len}");
        }
    }
}

#[test]
fn extension_into_and_out_of_prime_three() {
    let narrow = Extender::<B9, 9, ThreeLarge, 6>::new();
    let wide = Extender::<ThreeLarge, 6, B9, 9>::new();
    let mut r = Rng(3);
    for _ in 0..5_000 {
        let (neg, mag) = r.value(Z::max_bits());
        let x = Rns::<B9, 9>::from_sign_magnitude(neg, &mag).unwrap();
        let y = narrow.extend(&x).unwrap();
        assert_eq!(y.to_string(), x.to_string());
        let z = wide.extend(&y).unwrap();
        assert_eq!(z.to_string(), x.to_string());
    }
}
