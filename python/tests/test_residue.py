"""The Python bindings against Python's own integers, which are exact."""

import threading
from fractions import Fraction

import pytest
from hypothesis import given, settings, strategies as st

from residue_rs import Base, Matrix, Modulus, Vector

np = pytest.importorskip("numpy")


def ints(bits, min_size=0, max_size=40):
    """Lists of signed ints of at most `bits` bits, with random bit lengths."""
    one = st.integers(0, bits).flatmap(lambda b: st.integers(-(2**b) + 1, 2**b - 1))
    return st.lists(one, min_size=min_size, max_size=max_size)


def det_exact(rows):
    """Determinant by Gaussian elimination over the rationals."""
    a = [[Fraction(x) for x in r] for r in rows]
    n, d = len(a), Fraction(1)
    for c in range(n):
        p = next((r for r in range(c, n) if a[r][c] != 0), None)
        if p is None:
            return 0
        if p != c:
            a[c], a[p], d = a[p], a[c], -d
        d *= a[c][c]
        for r in range(c + 1, n):
            f = a[r][c] / a[c][c]
            a[r] = [x - f * y for x, y in zip(a[r], a[c])]
    return int(d)


def test_base_selection():
    for bits in [1, 60, 61, 62, 100, 122, 123, 500, 1000, 1980]:
        b = Base(bits=bits)
        assert b.max_bits >= bits
        smaller = [c for c in Base.available() if c < b.channels]
        if smaller:
            assert Base(channels=max(smaller)).max_bits < bits
    assert Base(channels=10).channels == 10
    assert len(Base(channels=4).primes) == 4
    assert Base(bits=100) == Base(channels=Base(bits=100).channels)
    with pytest.raises(ValueError):
        Base(bits=10**6)
    with pytest.raises(ValueError):
        Base(channels=7)
    with pytest.raises(TypeError):
        Base()
    with pytest.raises(TypeError):
        Base(bits=10, channels=4)


@settings(max_examples=60)
@given(st.sampled_from([2, 5, 8, 16, 32, 64]), st.data())
def test_vector_arithmetic(channels, data):
    b = Base(channels=channels)
    half = b.max_bits // 2 - 1
    n = data.draw(st.integers(0, 40))
    xs = data.draw(ints(half, n, n))
    ys = data.draw(ints(half, n, n))
    k = data.draw(ints(half - 8, 1, 1))[0]
    v, w = b.vector(xs), b.vector(ys)
    assert len(v) == n
    assert v.tolist() == xs
    assert list(v) == xs
    assert (v + w).tolist() == [x + y for x, y in zip(xs, ys)]
    assert (v - w).tolist() == [x - y for x, y in zip(xs, ys)]
    assert (v * w).tolist() == [x * y for x, y in zip(xs, ys)]
    assert (-v).tolist() == [-x for x in xs]
    assert (v * k).tolist() == [x * k for x in xs]
    assert (k * v).tolist() == [x * k for x in xs]
    if n:
        assert v[0] == xs[0] and v[-1] == xs[-1]
    if half * 2 + 8 <= b.max_bits and n:
        assert v.dot(w) == sum(x * y for x, y in zip(xs, ys))
    assert v.sum() == sum(xs)


def test_vector_errors():
    b = Base(bits=100)
    with pytest.raises(OverflowError):
        b.vector([2**200])
    big = b.vector([2**90])
    with pytest.raises(OverflowError):
        big * big
    with pytest.raises(ValueError):
        b.vector([1, 2]) + b.vector([1])
    with pytest.raises(TypeError):
        b.vector([1]) + Base(bits=500).vector([1])
    with pytest.raises(TypeError):
        b.vector([1]) + "x"
    with pytest.raises(TypeError):
        b.vector([1.5])
    with pytest.raises(IndexError):
        b.vector([1])[1]


def test_vector_numpy():
    b = Base(bits=130)
    a = np.array([1, -2, 2**62, -(2**63)], dtype=np.int64)
    v = b.vector(a)
    assert v.tolist() == [int(x) for x in a]
    out = v.to_numpy()
    assert out.dtype == np.int64 and (out == a).all()
    assert (v * v).tolist() == [int(x) ** 2 for x in a]
    with pytest.raises(OverflowError):
        (v * v).to_numpy()
    assert b.vector(np.array([3, -4], dtype=np.int32)).tolist() == [3, -4]
    with pytest.raises(ValueError):
        b.vector(np.zeros((2, 2), dtype=np.int64))


def test_vector_long_dot():
    with pytest.raises(OverflowError):
        Base(channels=4).vector([2**60 - 1] * 3000).dot(Base(channels=4).vector([2**60 - 1] * 3000))
    b = Base(channels=5)
    x = 2**60 - 1
    v = b.vector([x] * 3000)
    assert v.dot(v) == 3000 * x * x


@settings(max_examples=40)
@given(st.integers(1, 9), st.integers(1, 9), st.integers(1, 9), st.data())
def test_matrix_product(r, k, c, data):
    b = Base(bits=300)
    a = [data.draw(ints(120, k, k)) for _ in range(r)]
    m = [data.draw(ints(120, c, c)) for _ in range(k)]
    want = [[sum(a[i][t] * m[t][j] for t in range(k)) for j in range(c)] for i in range(r)]
    ma, mm = b.matrix(a), b.matrix(m)
    assert ma.shape == (r, k)
    assert ma.tolist() == a
    assert (ma @ mm).tolist() == want
    assert ma.T.tolist() == [list(col) for col in zip(*a)]
    assert (ma + ma).tolist() == [[2 * x for x in row] for row in a]
    assert (ma - ma).tolist() == [[0] * k for _ in range(r)]
    assert (-ma)[0, 0] == -a[0][0]


@settings(max_examples=40)
@given(st.integers(1, 12), st.data())
def test_determinant(n, data):
    rows = [data.draw(ints(20, n, n)) for _ in range(n)]
    assert Base(bits=500).matrix(rows).det() == det_exact(rows)


def test_matrix_errors_and_numpy():
    b = Base(bits=200)
    with pytest.raises(ValueError):
        b.matrix([[1, 2], [3]])
    with pytest.raises(ValueError):
        b.matrix([[1, 2]]) @ b.matrix([[1, 2]])
    with pytest.raises(ValueError):
        b.matrix([[1, 2]]).det()
    with pytest.raises(OverflowError):
        Base(channels=2).matrix([[2**50] * 30] * 30).det()
    a = np.arange(12, dtype=np.int64).reshape(3, 4) - 5
    m = b.matrix(a)
    assert m.shape == (3, 4)
    assert (m.to_numpy() == a).all()
    assert (m @ m.T).tolist() == (a @ a.T).tolist()


def test_identity_det_large():
    n = 40
    rows = [[int(i == j) * 3 for j in range(n)] for i in range(n)]
    assert Base(bits=200).matrix(rows).det() == 3**n


@settings(max_examples=60, deadline=None)
@given(st.integers(2, 4096), st.data())
def test_pow_batch(bits, data):
    m = data.draw(st.integers(2 ** (bits - 1), 2**bits - 1)) | 1
    if m < 3:
        m = 3
    M = Modulus(m)
    assert M.modulus == m
    xs = data.draw(st.lists(st.integers(-(2 ** (bits + 10)), 2 ** (bits + 10)), max_size=20))
    e = data.draw(st.integers(0, 2**300))
    try:
        got = M.pow_batch(xs, e)
    except ValueError:
        return
    assert got == [pow(x, e, m) for x in xs]
    ys = data.draw(st.lists(st.integers(-(2**bits), 2**bits), min_size=len(xs), max_size=len(xs)))
    assert M.mul_batch(xs, ys) == [x * y % m for x, y in zip(xs, ys)]


def test_modulus_sizes_and_errors():
    for bits in [128, 256, 512, 1024, 1536, 2048, 3072, 4096]:
        m = 2**bits - 1 if bits % 2 else (2 ** (bits - 1)) + 2 ** (bits // 3) + 1
        M = Modulus(m)
        xs = [2, 3, m - 1, 0]
        assert M.pow_batch(xs, 65537) == [pow(x, 65537, m) for x in xs]
    with pytest.raises(ValueError):
        Modulus(2**64)
    with pytest.raises(ValueError):
        Modulus(-7)
    with pytest.raises(ValueError):
        Modulus(2**5000 + 1)
    with pytest.raises(ValueError):
        Modulus(2147483647 * 3)
    with pytest.raises(ValueError):
        Modulus(11).pow_batch([1], -1)
    with pytest.raises(ValueError):
        Modulus(11).mul_batch([1, 2], [1])
    assert Modulus(11).pow_batch([], 5) == []


def test_threads():
    b = Base(bits=300)
    v = b.vector(list(range(-5000, 5000)))
    want = sum(x * x for x in range(-5000, 5000))
    M = Modulus(2**521 - 1)
    xs = list(range(2, 200))
    pw = [pow(x, 3, 2**521 - 1) for x in xs]
    errors = []

    def work():
        for _ in range(20):
            if v.dot(v) != want or M.pow_batch(xs, 3) != pw:
                errors.append(1)

    ts = [threading.Thread(target=work) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors


def test_types_and_repr():
    b = Base(bits=64)
    v = b.vector([1, 2])
    assert isinstance(v, Vector) and isinstance(b.matrix([[1]]), Matrix)
    assert v.base == b
    assert "Vector([1, 2]" in repr(v)
    assert "Modulus" in repr(Modulus(7))
    assert v.bits_bound >= 2
