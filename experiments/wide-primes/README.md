# Experiment: primes wider than 31 bits

Phase 0 of the roadmap asked one question before any rewrite: would moduli of
about 62 bits, as FLINT uses, close the gap with FLINT on matrices? Fewer
channels cover the same range (3 instead of 5 for a 130-bit product, 20 instead
of 40 for a 1216-bit determinant), but each channel loses the cheap 32-bit SIMD
products. This crate measures that trade-off. It is not part of the published
package.

Decision: keep 31-bit primes. No wider candidate met the criterion set in
advance (at least 1.3× faster than 31 bits on both the matrix product and the
large determinant), on either instruction set. Details and the other findings
are below; DESIGN.md records the decision.

## Candidates

| Candidate | Product | SIMD | Channels, F / H |
|---|---|---|---|
| 31 bit (current) | `u32×u32→u64`, Montgomery | yes | 5 / 40 |
| 62 bit | `u64×u64→u128`, Montgomery `R = 2^64` | no, scalar | 3 / 20 |
| 50 bit in `f64` | exact product via FMA, quotient from `1/p` | yes | 3 / 25 |

The 62-bit matrix product sums products in a `u128` plus a carry counter and
reduces each output once (three Montgomery reductions). The `f64` product must
reduce every product, because an unreduced sum does not fit a double. The
determinant runs the library's Gaussian elimination, generic over the three
arithmetics. Every result is recombined with the CRT and checked against an
independent computation (num-bigint for the product, the library for the
determinant) before it is timed.

## Results

Scenarios F and H of the README: product of two 64×64 matrices with 62-bit
entries, and determinant of a 64×64 matrix with 16-bit entries. Channel work
only, data already in RNS. Median of 3 processes, each the median of 21 to 41
timings. Intel Xeon at 2.8 GHz, 2 vCPU cloud VM with AVX-512, rustc 1.97.

The "library" rows call residue itself, which always picks its kernels at run
time (AVX-512 on this machine) whatever the build flags; every other row is
compiled for the instruction set in the heading.

### AVX-512 (`-C target-cpu=native`)

| F. Matrix product | Median (ms) | Range |
|---|---:|---:|
| 31 bit, library | 0.735 | 0.720 to 0.794 |
| 31 bit, same loop as library, one function per channel | **0.462** | 0.455 to 0.473 |
| 31 bit, exact copy of the library body (all channels in one function) | 0.590 | 0.585 to 0.728 |
| 31 bit, accumulators interleaved in one buffer | 0.673 | 0.663 to 0.690 |
| 31 bit, register tiling 8×8 (best tile) | 1.035 | 1.020 to 1.053 |
| mod 2^64 redundant channel alone | 0.061 | 0.058 to 0.073 |
| 62 bit, Montgomery `u128` | 0.679 | 0.677 to 0.954 |
| 50 bit, `f64` FMA | 0.781 | 0.780 to 0.804 |

| H. Determinant | Median (ms) | Range |
|---|---:|---:|
| 31 bit, library (includes the CRT) | 4.311 | 4.146 to 4.411 |
| 31 bit, same loop as library | **4.086** | 4.062 to 4.164 |
| 62 bit, Montgomery `u128` | 5.346 | 5.310 to 5.459 |
| 50 bit, `f64` FMA | 4.105 | 4.007 to 4.189 |

### AVX2 (`-C target-cpu=x86-64-v3`)

| F. Matrix product | Median (ms) | Range |
|---|---:|---:|
| 31 bit, same loop as library | **0.617** | 0.589 to 0.787 |
| 31 bit, register tiling 8×8 (best tile) | 0.671 | 0.659 to 1.262 |
| 62 bit, Montgomery `u128` | 1.029 | 1.018 to 1.055 |
| 50 bit, `f64` FMA | 1.023 | 0.986 to 1.342 |

| H. Determinant | Median (ms) | Range |
|---|---:|---:|
| 31 bit, same loop as library | **4.943** | 4.183 to 5.333 |
| 62 bit, Montgomery `u128` | 7.050 | 6.405 to 7.645 |
| 50 bit, `f64` FMA | 5.248 | 5.108 to 5.456 |

## What the numbers say

1. Wider primes lose. Against the best 31-bit loop, 62-bit primes run at
   0.68× (F) and 0.76× (H) with AVX-512 and at 0.60× and 0.70× with AVX2.
   Halving the channels does not pay for losing the 8 or 16 SIMD lanes. The
   `f64` variant ties on the determinant with AVX-512 and loses everywhere
   else.
2. The gap is in the loop, not in the width. The same 31-bit arithmetic
   runs at 0.46 ms when each channel is its own function and at 0.59 ms when
   all channels share one function, as in the library; the library itself
   takes 0.74 ms. That is a factor of 1.3 to 1.6 with identical arithmetic,
   larger than anything a wider prime offered. Part of it is how the loop is
   split into functions (0.46 against 0.59); the rest, between the copy and the
   library, is not isolated yet. Candidates are the allocation of the output,
   the runtime dispatch, and the clock speed of 512-bit instructions, which
   `pulp` uses where a native build for this CPU prefers 256-bit vectors.
3. The redundant channel is cheap. Its `u64` products cost 0.06 ms, about
   8% of the library's product. Splitting them into 32-bit pieces is slower.
4. Register tiling needs explicit SIMD. Written as plain loops, LLVM
   vectorized the tiles along the wrong dimension (gathers and scatters with
   AVX-512, removed only partially by restructuring) and the best tile stayed
   behind the simple loop. A tiled kernel has to be written with explicit
   vector types (pulp's safe `Simd` API keeps the crate free of `unsafe`).

## Running it

```sh
cd experiments/wide-primes
RUSTFLAGS="-C target-cpu=native" cargo run --release
CARGO_TARGET_DIR=target-v3 RUSTFLAGS="-C target-cpu=x86-64-v3" cargo run --release
```

The numbers vary by 10 to 20 percent between processes on a shared VM, so run
each build several times and compare medians.
