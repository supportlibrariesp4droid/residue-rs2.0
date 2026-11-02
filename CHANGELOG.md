# Changelog

## 2.0.0

Phases 0 to 4 and 6 of [ROADMAP.md](ROADMAP.md). Code written for 0.1.0 keeps
compiling and working: the new major version marks the size of the additions
(arithmetic modulo a large odd number, multithreading, WebAssembly kernels,
Python bindings), not a break.

The benchmark charts in `docs/` were measured again for this release, with
FLINT 3.6.0, and three new charts were added: arithmetic modulo a large odd
number, WebAssembly against native code, and the Python bindings.

Speed-ups over earlier code below are measured against 0.1.0 on the same machine, running
the two versions alternately, three rounds each. An earlier draft of this
entry, measured differently, overstated several of them (one by a factor of
8); see "Measuring" in DESIGN.md.

### Phase 6: Python bindings

- New crate `python/` (`residue-py`, not published on crates.io), built with
  PyO3 and maturin into the Python package `residue-rs` (imported as
  `residue_rs`): one abi3 wheel per platform for Python 3.11 and later. The
  repository becomes a Cargo workspace; commands in the root still act on
  the library alone.
- `Base(bits=…)` chooses among 14 compiled bases (2 to 64 primes, 60 to 1982
  bits); `Vector` and `Matrix` with exact arithmetic, dot product, matrix
  product and determinant, from and to Python ints and NumPy int64 arrays;
  `Modulus(m)` for odd moduli up to 4096 bits with `pow_batch` and
  `mul_batch`. Long operations release the GIL.
- Python ints are read in the narrowest form that holds them (`i64`, `i128`,
  then two's complement through one `int.to_bytes` call each): five times
  faster than PyO3's `BigInt` conversion for 100-bit values.
- Tests with pytest and hypothesis against Python's integers; `bench/bench.py`
  against Python ints, gmpy2 and python-flint (results in python/README.md):
  on data already converted, 7.5 times faster than python-flint on a dot
  product and 2.3 times on a matrix product; GMP stays 1.5 to 2 times faster
  on batches of modular exponentiations.
- CI: tests on Linux (x86-64 and arm64), macOS and Windows with Python 3.11
  and 3.13; wheels for Linux, macOS and Windows on x86-64 and arm64 (Windows
  x86-64 only); publication to PyPI on a `py-v*` tag through trusted
  publishing, which must be configured on PyPI first.

### Phase 4: aarch64 and WebAssembly

- Fixed: with `alloc`, the crate did not compile for 32-bit targets (a
  length check used `1usize << 32`).
- Fixed: with `parallel` on WebAssembly without threads, the first parallel
  operation panicked inside rayon. Parallel work now runs only where threads
  can be created; elsewhere everything runs on the calling thread.
- WebAssembly `simd128` kernels for the dot product, the matrix product row
  and the element-wise Montgomery product, using `i64x2.extmul_*_i32x4_u`
  where LLVM had emitted full 64-bit multiplies. In wasmtime: element-wise
  product 2.5 times faster than before (with simd128 it had been slower than
  the scalar build), dot product, matrix product and `ModVec` product 11 to
  14 percent faster. simd128 is now worth 1.2 to 1.8 times over scalar
  WebAssembly, except for the conversion from `i128`.
- Verified: the whole test suite on aarch64 (under qemu) and on
  `wasm32-wasip1` with and without simd128 (under wasmtime); the aarch64
  assembly of the main loops uses NEON `umull`/`umlal`. No ARM hardware was
  available, so there are no ARM timings yet.
- `examples/timing.rs`: timings without criterion, for WebAssembly and other
  targets where the benchmarks do not run.
- CI: WebAssembly tests with every feature and with `alloc` only, both with
  and without simd128; a `timing` job on x86-64, ARM Linux, Apple Silicon and
  WebAssembly. proptest no longer uses its `fork` and `timeout` features,
  which do not build for WebAssembly.

### Phase 3: arithmetic modulo a large odd number

- `MontRns<B1, N1, B2, N2>`: RNS Montgomery multiplication modulo an odd
  modulus chosen at run time, given as 64-bit limbs, over two disjoint bases
  (Bajard, Didier and Kornerup; Kawamura et al.; the proofs of the bounds are in
  DESIGN.md). `MontRnsError` reports an even modulus or one below 3, a modulus
  sharing a prime with a base, overlapping bases, and bases too small (with
  the number of bits needed).
- `ModElem`: one element; `*`, `+`, `-`, unary `-`, `square`, `pow` (4-bit
  window), `inv_prime` (Fermat), `to_limbs`, `to_u128`, canonical equality.
  Elements carry a factor bound instead of being reduced after every
  operation; sums normalize when needed, so no operation can fail.
- `ModVec`: many elements, stored channel by channel. The two base
  extensions of a product become matrix products on the register-tiled
  kernels; the element-wise steps are vectorized loops; with `parallel`,
  batches are split into blocks across threads.
- **Not constant time.** Branches and memory accesses depend on the data; do
  not use these types with secret values.
- Measured on one thread against GMP (`rug`), medians of three sessions: one
  product 3.2 times slower at 1024 bits and 5.2 at 2048; one exponentiation
  4.8 and 7.6 times slower. 256 exponentiations with one 256-bit exponent in
  a `ModVec`: 41.8 ms against GMP's 29.1 ms at 1024 bits (1.4 times slower)
  and 125 ms against 94.6 ms at 2048 bits (1.3 times slower). num-bigint takes
  37.9 and 158 ms, so it is faster at 1024 bits and slower at 2048. The batch
  is 3.4 to 5.9 times faster than the same work one `ModElem` at a time.
- Matrix product tiles reduce with vector instructions instead of one scalar
  reduction per output: 1.3 times faster on 64×64, and 1.4 to 1.6 times on
  the 1024-bit batch. Tile loops no longer check bounds on every step: a
  further 5 to 10 percent on batches.
- Tests: property tests against num-bigint on three pairs of bases up to 600
  bits, at every kernel level and with the parallel path forced; fixed tests
  at 2048 and 4096 bits; inverses modulo known primes; factor growth over 200
  doublings. Debug builds check the bound of every product.
- `largest_primes_after` and `rns_base!(Name: largest N after K)` to declare
  the second, disjoint base.

### Phase 2: explicit kernels and multithreading

Measured against the end of Phase 1, alternating, three rounds, on a 2-vCPU
x86-64 VM with AVX-512.

- Matrix product with explicit AVX2 / AVX-512 register tiles
  (`src/batch/x86.rs`, through `pulp`'s safe intrinsics): 1.5 times faster on
  one thread (64×64, 5 channels); in one binary against the portable kernel,
  1.5 to 1.9 times with AVX-512 and 1.1 to 1.2 with AVX2.
- Conversion to binary up to 6 limbs with an explicit AVX-512 kernel: 1.25 to
  1.3 times faster for 9 channels at full width.
- New `parallel` feature (rayon, implies `std`): matrix products,
  determinants, conversions in both directions, dot products and element-wise
  products split across threads above about 150 to 300 µs of work. With two
  threads against one: determinant 64×64 1.8 to 1.9 times faster, 32×32 1.6 to
  1.8, matrix product 1.3 to 1.6, conversions of 1024 values on 36 to 68
  channels 1.5 to 1.8. Against Phase 1, the 64×64 matrix product is 2.0 to 2.2
  times faster with two threads.
- Scenarios B and C starting from binary measured 5 to 10 percent slower than
  in Phase 1, with rounds that disagree; noted, not explained.

### Added
- Batch conversions of `RnsVec`, vectorized across elements:
  `from_twos_complement`, `to_twos_complement`, `twos_complement_width`,
  `to_i128s`, `to_i64s`, and `from_bigints` / `to_bigints` with the
  `num-bigint` feature. From binary: 3.0 times faster than converting one value
  at a time on 9 channels, 5.1 times on 68. To `i128`: 2.5 times on 9
  channels, 17 times on 36. Full width to binary: 1.1 to 1.3 times.
- `Channel::redc_halves` and `Channel::reduce_halves`: Montgomery reduction of
  a sum kept in two 64-bit halves, the last step of delayed reduction.
- `core::error::Error` for `OutOfRange`, `ParseExactError` and
  `ParseRnsError`.
- `#[must_use]` on pure functions and methods.
- `benches/internal.rs`: conversions, sign test, batch against scalar.
- `experiments/wide-primes` (not published): the measurement that rejected
  primes of about 62 bits.
- Continuous integration on GitHub Actions.

### Performance
- Scalar conversion from binary reduces once per channel instead of twice per
  limb: 1.3 to 1.9 times faster on 9 channels, 3.2 on 36, 3.4 on 68.
- Reconstruction computes only the limbs the certified bound needs, so
  `Rns::to_i128` computes two limbs whatever the base: 1.4 times faster on 9
  channels, 4 to 4.7 times on 36.
- Reconstruction scans products by column: 1.2 to 1.3 times faster on 36 and
  68 channels. On 9 channels the scalar conversion to binary is about 15
  percent slower than in 0.1.0, for a reason not yet found; the batch API is
  1.3 times faster than 0.1.0 there.
- A 1024-bit product starting and ending in binary (scenario E) is 2.3 times
  faster; scenario A starting from binary 1.3 times; the matrix product
  starting from binary 1.3 to 1.4 times.
- The matrix product dispatches each channel separately. Without that, the
  changes above had made it 28 percent slower; it is now as fast as in 0.1.0.
- `Extender` and long-input reduction no longer use 128-bit divisions (not
  measured separately).

### Fixed
- `cargo test --no-default-features` did not compile: test files that need an
  allocator now declare it.

### Changed
- `rug` and `criterion` are no longer dev-dependencies on WebAssembly and
  Windows targets, where GMP cannot be built; the tests need neither.

## 0.1.0

First release.
