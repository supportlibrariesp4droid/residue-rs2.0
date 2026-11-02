# residue-rs for Python

Python bindings for [residue-rs](https://github.com/supportlibrariesp4droid/residue-rs2.0),
exact integer arithmetic in a Residue Number System, written in Rust.
Vectors and matrices of big integers with exact dot products, matrix products
and determinants, and modular exponentiation in batches.

```sh
pip install residue-rs
```

The package is imported as `residue_rs`.

Requires Python 3.11 or later. NumPy is optional: it is needed only by
`to_numpy()`, and int64 arrays are accepted as input when it is installed.

## Example

`Base(bits=200)` holds signed values of up to 200 bits. Vectors and matrices
are built from Python ints or from NumPy int64 arrays, and `Modulus` works
modulo any odd number of up to 4096 bits.

```python
from residue_rs import Base, Modulus

b = Base(bits=200)
v = b.vector([1, -2, 3, 10**50])
w = b.vector([4, 5, -6, 2])
assert (v * w).tolist() == [4, -10, -18, 2 * 10**50]
assert v.dot(w) == 4 - 10 - 18 + 2 * 10**50

m = Base(bits=300).matrix([[2, 1], [7, 4]])
assert m.det() == 1
assert (m @ m).tolist() == [[11, 6], [42, 23]]

M = Modulus(2**127 - 1)
assert M.pow_batch([3, 5, 7], 65537) == [pow(x, 65537, M.modulus) for x in [3, 5, 7]]
```

Results are exact. An operation whose result could leave the range of the base
raises `OverflowError`, never wraps: choose the base for the largest value a
computation can produce (for a dot product of `n` values of `b` bits, `2b +
log2(n)` bits).

## API

- `Base(bits=…)` picks the smallest base holding signed values of that many
  bits; `Base(channels=…)` asks for a number of primes. The bindings carry a
  fixed menu of bases, from 2 primes (60 bits) to 64 primes (1982 bits):
  `Base.available()`.
- `Base.vector(data)` and `Base.matrix(data)` build from Python ints (nested
  lists for a matrix) or from NumPy int64 arrays.
- `Vector`: `+`, `-`, `*` (element-wise, or by an int), unary `-`, `dot`,
  `sum`, indexing, iteration, `tolist`, `to_numpy`.
- `Matrix`: `@`, `+`, `-`, unary `-`, `.T`, `det`, `m[i, j]`, `shape`,
  `tolist`, `to_numpy`.
- `Modulus(m)`: an odd modulus of up to 4096 bits; `pow_batch(xs, e)` and
  `mul_batch(xs, ys)`, which return lists of Python ints.

Long operations release the GIL, so other Python threads run meanwhile, and
large ones use several cores.

`Modulus` is **not constant time**: do not use it with secret data.

## When it is worth it

![Running times of Python ints, gmpy2, python-flint and residue_rs](https://raw.githubusercontent.com/supportlibrariesp4droid/residue-rs2.0/main/docs/python.png)

Measured with `bench/bench.py` on a 2-vCPU x86-64 VM with AVX-512, Python
3.13, gmpy2 2.3.2, python-flint 0.9.0; milliseconds, fastest of three runs of
the medians. "+ conv" includes converting from Python ints and back.

| | Python ints | gmpy2 | python-flint | python-flint + conv | residue | residue + conv |
|---|---|---|---|---|---|---|
| dot product, 10 000 × 100 bits | 0.716 | 0.660 | 0.166 | 3.67 | 0.022 | 1.19 |
| matrix product 64×64, 60 bits | 19.5 | n/a | 0.282 | 0.907 | 0.124 | 0.970 |
| determinant 64×64, 10 bits | n/a | n/a | 1.05 | 1.23 | 0.877 | 1.19 |
| 256 × `pow`, 1024-bit modulus, 256-bit exponent | 220 | 21.8 | 22.5 | | | 44.1 |
| 256 × `pow`, 2048-bit modulus, 256-bit exponent | 789 | 95.2 | 72.3 | | | 108 |

- On data kept in residue's vectors and matrices it is the fastest of these:
  7.5 times python-flint on the dot product, 2.3 times on the matrix product,
  1.2 times on the determinant.
- Converting Python ints costs more than the arithmetic in the first two rows,
  so the gain comes from doing several operations per conversion. Conversion
  included, it is 3.1 times faster than python-flint on the dot product and
  level with it on the matrix product and the determinant; for a single dot
  product on Python ints, plain Python or gmpy2 is faster still.
- For modular exponentiation GMP (through gmpy2 or python-flint) stays 1.5 to
  2 times faster, even when residue uses both cores; residue is 5 to 7 times
  faster than Python's own `pow`.

## Building from source

`maturin develop` installs into the active virtual environment; `maturin build
--release` writes a wheel instead.

```sh
cd python
pip install maturin
maturin develop --release
pytest tests
```

The extension is large (about 14 MB, a 5 MB wheel) because each base of the
menu is a full instantiation of the Rust library.
