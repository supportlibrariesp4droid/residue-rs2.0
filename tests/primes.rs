//! Correctness of compile-time primality testing and prime tables.

use residue::primes::{is_prime, largest_primes, prime_below};

/// Reference: trial division.
fn naive_is_prime(n: u32) -> bool {
    if n < 2 {
        return false;
    }
    let mut d = 2u64;
    while d * d <= n as u64 {
        if (n as u64).is_multiple_of(d) {
            return false;
        }
        d += 1;
    }
    true
}

#[test]
fn is_prime_matches_trial_division_small() {
    for n in 0..200_000u32 {
        assert_eq!(is_prime(n), naive_is_prime(n), "mismatch at n={n}");
    }
}

#[test]
fn is_prime_known_large() {
    assert!(is_prime(2_147_483_647));
    assert!(!is_prime(2_147_483_646));
    assert!(!is_prime(2_147_483_645));
    assert!(is_prime(1_000_000_007));
    assert!(!is_prime(1_000_000_006));
    assert!(!is_prime(62_745));
    assert!(!is_prime(162_401));
}

#[test]
fn largest_primes_are_prime_and_descending() {
    let ps = largest_primes::<64>();
    assert_eq!(ps[0], 2_147_483_647);
    for i in 0..ps.len() {
        assert!(is_prime(ps[i]));
        assert!(ps[i] < (1 << 31));
        if i > 0 {
            assert!(ps[i] < ps[i - 1], "not strictly descending at {i}");
        }
    }
}

#[test]
fn prime_below_works() {
    assert_eq!(prime_below(100), 97);
    assert_eq!(prime_below(97), 97);
    assert_eq!(prime_below(96), 89);
    assert_eq!(prime_below((1 << 31) - 1), 2_147_483_647);
}
