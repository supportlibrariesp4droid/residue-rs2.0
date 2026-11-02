"""residue_rs against Python ints, gmpy2 and python-flint.

Run with `python bench/bench.py`, or `python bench/bench.py --json FILE` to
also write the medians to FILE for scripts/plot_benchmarks.py. Each figure is
the median of several runs, in milliseconds. "residue" times the operation on data already converted;
"residue + conv" includes building the vectors or matrices from Python ints
and reading the result back, which is what a one-off call costs.
"""

import json
import random
import statistics
import sys
import time

import flint
import gmpy2
from residue_rs import Base, Modulus

R = random.Random(1)


def timed(f, runs):
    f()
    t = []
    for _ in range(runs):
        s = time.perf_counter()
        f()
        t.append(time.perf_counter() - s)
    return statistics.median(t) * 1e3


RESULTS = {}


def table(title, rows):
    print(f"\n{title}")
    for name, ms in rows:
        print(f"  {name:28}{ms:10.3f} ms")
    RESULTS[title] = dict(rows)


def rand(bits, n):
    return [R.randrange(-(2**bits), 2**bits) for _ in range(n)]


def dot():
    n, bits = 10_000, 100
    a, b = rand(bits, n), rand(bits, n)
    want = sum(x * y for x, y in zip(a, b))
    base = Base(bits=2 * bits + 16)
    va, vb = base.vector(a), base.vector(b)
    ga, gb = [gmpy2.mpz(x) for x in a], [gmpy2.mpz(x) for x in b]
    fa, fb = flint.fmpz_mat(1, n, a), flint.fmpz_mat(n, 1, b)
    assert va.dot(vb) == want and int((fa * fb)[0, 0]) == want
    table(
        f"dot product, {n} values of {bits} bits",
        [
            ("Python ints", timed(lambda: sum(x * y for x, y in zip(a, b)), 21)),
            ("gmpy2", timed(lambda: sum(x * y for x, y in zip(ga, gb)), 21)),
            ("python-flint fmpz_mat", timed(lambda: fa * fb, 21)),
            ("python-flint + conv", timed(lambda: int((flint.fmpz_mat(1, n, a) * flint.fmpz_mat(n, 1, b))[0, 0]), 21)),
            ("residue", timed(lambda: va.dot(vb), 51)),
            ("residue + conv", timed(lambda: base.vector(a).dot(base.vector(b)), 21)),
        ],
    )


def matmul():
    n, bits = 64, 60
    a = [rand(bits, n) for _ in range(n)]
    b = [rand(bits, n) for _ in range(n)]
    base = Base(bits=2 * bits + 8)
    ma, mb = base.matrix(a), base.matrix(b)
    fa, fb = flint.fmpz_mat(a), flint.fmpz_mat(b)
    assert (ma @ mb).tolist() == [[int(x) for x in row] for row in (fa * fb).tolist()]

    def python():
        bt = list(zip(*b))
        return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]

    table(
        f"matrix product {n}×{n}, {bits}-bit entries",
        [
            ("Python ints", timed(python, 5)),
            ("python-flint fmpz_mat", timed(lambda: fa * fb, 21)),
            ("python-flint + conv", timed(lambda: (flint.fmpz_mat(a) * flint.fmpz_mat(b)).tolist(), 11)),
            ("residue", timed(lambda: ma @ mb, 21)),
            ("residue + conv", timed(lambda: (base.matrix(a) @ base.matrix(b)).tolist(), 11)),
        ],
    )


def det():
    n, bits = 64, 10
    a = [rand(bits, n) for _ in range(n)]
    base = Base(bits=n * (bits + 3) + 16)
    m, f = base.matrix(a), flint.fmpz_mat(a)
    assert m.det() == int(f.det())
    table(
        f"determinant {n}×{n}, {bits}-bit entries (base of {base.channels} primes)",
        [
            ("python-flint fmpz_mat", timed(f.det, 21)),
            ("python-flint + conv", timed(lambda: int(flint.fmpz_mat(a).det()), 21)),
            ("residue", timed(m.det, 21)),
            ("residue + conv", timed(lambda: base.matrix(a).det(), 21)),
        ],
    )


def modexp(bits):
    n = 256
    m = R.getrandbits(bits) | 1 | (1 << (bits - 1))
    e = R.getrandbits(256)
    xs = [R.randrange(m) for _ in range(n)]
    M = Modulus(m)
    gm, ge = gmpy2.mpz(m), gmpy2.mpz(e)
    gx = [gmpy2.mpz(x) for x in xs]
    fm, fe = flint.fmpz(m), flint.fmpz(e)
    fx = [flint.fmpz(x) for x in xs]
    want = [pow(x, e, m) for x in xs]
    assert M.pow_batch(xs, e) == want
    table(
        f"{n} modular exponentiations, {bits}-bit modulus, 256-bit exponent",
        [
            ("Python pow", timed(lambda: [pow(x, e, m) for x in xs], 5)),
            ("gmpy2 powmod", timed(lambda: [gmpy2.powmod(x, ge, gm) for x in gx], 11)),
            ("gmpy2 powmod_base_list", timed(lambda: gmpy2.powmod_base_list(gx, ge, gm), 11)),
            ("python-flint fmpz", timed(lambda: [pow(x, fe, fm) for x in fx], 11)),
            ("residue pow_batch", timed(lambda: M.pow_batch(xs, e), 11)),
        ],
    )


if __name__ == "__main__":
    print(f"gmpy2 {gmpy2.version()}, python-flint {flint.__version__}")
    dot()
    matmul()
    det()
    modexp(1024)
    modexp(2048)
    if len(sys.argv) > 2 and sys.argv[1] == "--json":
        with open(sys.argv[2], "w") as f:
            json.dump(RESULTS, f, indent=1)
