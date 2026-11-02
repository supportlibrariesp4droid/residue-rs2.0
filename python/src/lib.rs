//! Python bindings for residue-rs.
//!
//! Python has no const generics, so the number of primes in a base cannot be
//! chosen at run time from Rust's point of view. The bindings instantiate a
//! fixed menu of bases instead (the roadmap's option (b)): `Base(bits=…)`
//! picks the smallest one that holds the requested number of bits, and
//! `Modulus(m)` picks the smallest pair of bases that fits the modulus. Every
//! menu entry is a full instantiation of the library, which is what makes the
//! extension a few megabytes.
//!
//! Heavy work runs with the GIL released, so other Python threads keep going.
#![deny(unsafe_code)]
#![allow(long_running_const_eval)]

use num_bigint::{BigInt, BigUint, Sign};
use num_integer::Integer;
use num_traits::Signed;
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyIndexError, PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyInt, PyList, PyTuple};
use residue::{ModVec, MontRns, MontRnsError, OutOfRange, Rns, RnsBase, RnsMatrix, RnsVec};

/// Integers read from Python: machine-sized when they all fit, else as
/// two's complement limbs, `width` per value.
enum Ints {
    Small(Vec<i64>),
    Mid(Vec<i128>),
    Twos { width: usize, limbs: Vec<u64> },
}

/// Read Python ints, in the narrowest form that holds them all: `i64`, then
/// `i128` (two cheap C calls per value under the stable ABI), then two's
/// complement limbs written by `int.to_bytes`, `width` limbs per value. A
/// value too wide for `width` limbs is out of range of the base.
fn ints_from_items(items: &[Bound<'_, PyAny>], width: usize) -> PyResult<Ints> {
    let mut small = Vec::with_capacity(items.len());
    for it in items {
        match it.extract::<i64>() {
            Ok(v) => small.push(v),
            Err(_) => return mid_from_items(items, width),
        }
    }
    Ok(Ints::Small(small))
}

fn mid_from_items(items: &[Bound<'_, PyAny>], width: usize) -> PyResult<Ints> {
    let mut mid = Vec::with_capacity(items.len());
    for it in items {
        match it.extract::<i128>() {
            Ok(v) => mid.push(v),
            Err(_) => return twos_from_items(items, width),
        }
    }
    Ok(Ints::Mid(mid))
}

fn twos_from_items(items: &[Bound<'_, PyAny>], width: usize) -> PyResult<Ints> {
    let Some(first) = items.first() else {
        return Ok(Ints::Small(Vec::new()));
    };
    let py = first.py();
    let kwargs = PyDict::new(py);
    kwargs.set_item("signed", true)?;
    let args = PyTuple::new(
        py,
        [
            (8 * width).into_pyobject(py)?.into_any(),
            "little".into_pyobject(py)?.into_any(),
        ],
    )?;
    let name = pyo3::intern!(py, "to_bytes");
    let mut limbs = Vec::with_capacity(items.len() * width);
    for it in items {
        if !it.is_instance_of::<PyInt>() {
            it.extract::<BigInt>()?;
        }
        let b = it.call_method(name, &args, Some(&kwargs)).map_err(|e| {
            if e.is_instance_of::<PyOverflowError>(py) {
                overflow(OutOfRange)
            } else {
                e
            }
        })?;
        let b = b.cast::<PyBytes>()?.as_bytes();
        limbs.extend(
            b.chunks_exact(8)
                .map(|c| u64::from_le_bytes(c.try_into().unwrap())),
        );
    }
    Ok(Ints::Twos { width, limbs })
}

fn items<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    obj.try_iter()?.collect()
}

/// A flat list of integers and its shape: from an int64 buffer (a NumPy
/// array), or from nested sequences of Python ints (`dims` levels deep).
fn read_ints<'py>(
    obj: &Bound<'py, PyAny>,
    dims: usize,
    width: usize,
) -> PyResult<(Ints, Vec<usize>)> {
    if let Ok(buf) = PyBuffer::<i64>::get(obj) {
        if buf.dimensions() != dims {
            return Err(PyValueError::new_err(format!(
                "expected a {dims}-dimensional array, got {} dimensions",
                buf.dimensions()
            )));
        }
        let shape = buf.shape().to_vec();
        return Ok((Ints::Small(buf.to_vec(obj.py())?), shape));
    }
    if dims == 1 {
        let it = items(obj)?;
        let n = it.len();
        return Ok((ints_from_items(&it, width)?, vec![n]));
    }
    let rows = items(obj)?;
    let mut flat = Vec::new();
    let mut cols = None;
    for row in &rows {
        let r = items(row)?;
        if *cols.get_or_insert(r.len()) != r.len() {
            return Err(PyValueError::new_err("rows of different lengths"));
        }
        flat.extend(r);
    }
    Ok((
        ints_from_items(&flat, width)?,
        vec![rows.len(), cols.unwrap_or(0)],
    ))
}

fn overflow(_: OutOfRange) -> PyErr {
    PyOverflowError::new_err("value outside the range of the base; use a Base with more bits")
}

fn too_big() -> PyErr {
    PyOverflowError::new_err("result may exceed the range of the base; use a Base with more bits")
}

fn to_rnsvec<B: RnsBase<N>, const N: usize>(d: &Ints) -> PyResult<RnsVec<B, N>> {
    match d {
        Ints::Small(v) => {
            let w: Vec<i128> = v.iter().map(|&x| x as i128).collect();
            RnsVec::from_i128s(&w).map_err(overflow)
        }
        Ints::Mid(v) => RnsVec::from_i128s(v).map_err(overflow),
        Ints::Twos { width, limbs } => {
            RnsVec::from_twos_complement(*width, limbs).map_err(overflow)
        }
    }
}

/// The values of a vector, in the cheapest exact form.
enum Out {
    I64(Vec<i64>),
    I128(Vec<i128>),
    Big(Vec<BigInt>),
}

fn vec_out<B: RnsBase<N>, const N: usize>(v: &RnsVec<B, N>) -> Out {
    if v.bits_bound() < 64 {
        if let Ok(x) = v.to_i64s() {
            return Out::I64(x);
        }
    }
    if v.bits_bound() < 128 {
        if let Ok(x) = v.to_i128s() {
            return Out::I128(x);
        }
    }
    Out::Big(v.to_bigints())
}

fn out_list<'py>(py: Python<'py>, o: Out) -> PyResult<Bound<'py, PyList>> {
    match o {
        Out::I64(x) => PyList::new(py, x),
        Out::I128(x) => PyList::new(py, x),
        Out::Big(x) => PyList::new(py, x),
    }
}

fn numpy_i64<'py>(py: Python<'py>, data: &[i64], shape: &[usize]) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_ne_bytes()).collect();
    let arr = np.call_method1(
        "frombuffer",
        (PyByteArray::new(py, &bytes), np.getattr("int64")?),
    )?;
    arr.call_method1("reshape", (PyTuple::new(py, shape)?,))
}

#[derive(Clone, Copy)]
enum Op {
    Add,
    Sub,
    Mul,
}

fn vec_binop<B: RnsBase<N>, const N: usize>(
    a: &RnsVec<B, N>,
    b: &RnsVec<B, N>,
    op: Op,
) -> PyResult<RnsVec<B, N>> {
    if a.len() != b.len() {
        return Err(PyValueError::new_err(format!(
            "length mismatch: {} and {}",
            a.len(),
            b.len()
        )));
    }
    let mut c = a.clone();
    match op {
        Op::Add => c.checked_add_assign(b),
        Op::Sub => c.checked_sub_assign(b),
        Op::Mul => c.checked_mul_assign(b),
    }
    .map_err(overflow)?;
    Ok(c)
}

fn vec_scale<B: RnsBase<N>, const N: usize>(
    a: &RnsVec<B, N>,
    k: &BigInt,
) -> PyResult<RnsVec<B, N>> {
    let k = Rns::<B, N>::try_from(k).map_err(overflow)?;
    let mut c = a.clone();
    c.checked_scale_assign(k).map_err(overflow)?;
    Ok(c)
}

fn vec_neg<B: RnsBase<N>, const N: usize>(a: &RnsVec<B, N>) -> RnsVec<B, N> {
    let mut c = a.clone();
    c.neg_assign();
    c
}

macro_rules! int_menu {
    ($($v:ident $n:literal),* $(,)?) => {
        /// Base markers of the integer menu, `I<N>` with the `N` largest primes.
        mod ib {
            $( residue::rns_base!(pub $v: largest $n); )*
        }

        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        enum IntKind { $($v),* }

        impl IntKind {
            const ALL: &'static [IntKind] = &[$(IntKind::$v),*];
            fn channels(self) -> usize { match self { $(IntKind::$v => $n),* } }
            fn max_bits(self) -> u32 { match self { $(IntKind::$v => Rns::<ib::$v, $n>::max_bits()),* } }
            fn primes(self) -> Vec<u32> {
                match self {
                    $(IntKind::$v => <ib::$v as RnsBase<$n>>::base().channels().iter().map(|c| c.modulus()).collect()),*
                }
            }
            fn vector(self, d: &Ints) -> PyResult<AnyVec> {
                match self { $(IntKind::$v => Ok(AnyVec::$v(to_rnsvec(d)?))),* }
            }
            fn matrix(self, rows: usize, cols: usize, d: &Ints) -> PyResult<AnyMat> {
                match self { $(IntKind::$v => Ok(AnyMat::$v(RnsMatrix::from_vec(rows, cols, to_rnsvec(d)?)))),* }
            }
        }

        #[allow(clippy::large_enum_variant)]
        enum AnyVec { $($v(RnsVec<ib::$v, $n>)),* }

        impl AnyVec {
            fn kind(&self) -> IntKind { match self { $(AnyVec::$v(_) => IntKind::$v),* } }
            fn len(&self) -> usize { match self { $(AnyVec::$v(a) => a.len()),* } }
            fn bits_bound(&self) -> u32 { match self { $(AnyVec::$v(a) => a.bits_bound()),* } }
            fn binop(&self, o: &AnyVec, op: Op) -> PyResult<AnyVec> {
                match (self, o) {
                    $((AnyVec::$v(a), AnyVec::$v(b)) => Ok(AnyVec::$v(vec_binop(a, b, op)?)),)*
                    #[allow(unreachable_patterns)]
                    _ => Err(mismatch()),
                }
            }
            fn scale(&self, k: &BigInt) -> PyResult<AnyVec> {
                match self { $(AnyVec::$v(a) => Ok(AnyVec::$v(vec_scale(a, k)?))),* }
            }
            fn neg(&self) -> AnyVec { match self { $(AnyVec::$v(a) => AnyVec::$v(vec_neg(a))),* } }
            fn dot(&self, o: &AnyVec) -> PyResult<BigInt> {
                match (self, o) {
                    $((AnyVec::$v(a), AnyVec::$v(b)) => {
                        if a.len() != b.len() {
                            return Err(PyValueError::new_err("length mismatch"));
                        }
                        a.dot(b).map(|r| r.to_bigint()).ok_or_else(too_big)
                    })*
                    #[allow(unreachable_patterns)]
                    _ => Err(mismatch()),
                }
            }
            fn sum(&self) -> PyResult<BigInt> {
                match self { $(AnyVec::$v(a) => a.sum().map(|r| r.to_bigint()).ok_or_else(too_big)),* }
            }
            fn get(&self, k: usize) -> Option<BigInt> {
                match self { $(AnyVec::$v(a) => a.get(k).map(|r| r.to_bigint())),* }
            }
            fn out(&self) -> Out { match self { $(AnyVec::$v(a) => vec_out(a)),* } }
            fn i64s(&self) -> PyResult<Vec<i64>> {
                match self { $(AnyVec::$v(a) => a.to_i64s().map_err(|_| PyOverflowError::new_err("a value does not fit int64"))),* }
            }
        }

        #[allow(clippy::large_enum_variant)]
        enum AnyMat { $($v(RnsMatrix<ib::$v, $n>)),* }

        impl AnyMat {
            fn kind(&self) -> IntKind { match self { $(AnyMat::$v(_) => IntKind::$v),* } }
            fn shape(&self) -> (usize, usize) { match self { $(AnyMat::$v(a) => (a.rows(), a.cols())),* } }
            fn vec(&self) -> AnyVec { match self { $(AnyMat::$v(a) => AnyVec::$v(a.as_vec().clone())),* } }
            fn from_vec(rows: usize, cols: usize, v: AnyVec) -> AnyMat {
                match v { $(AnyVec::$v(a) => AnyMat::$v(RnsMatrix::from_vec(rows, cols, a))),* }
            }
            fn matmul(&self, o: &AnyMat) -> PyResult<AnyMat> {
                let ((r, k), (k2, c)) = (self.shape(), o.shape());
                if k != k2 {
                    return Err(PyValueError::new_err(format!(
                        "matrix product of {r}×{k} and {k2}×{c}"
                    )));
                }
                match (self, o) {
                    $((AnyMat::$v(a), AnyMat::$v(b)) => a.checked_matmul(b).map(AnyMat::$v).ok_or_else(too_big),)*
                    #[allow(unreachable_patterns)]
                    _ => Err(mismatch()),
                }
            }
            fn det(&self) -> PyResult<BigInt> {
                let (r, c) = self.shape();
                if r != c {
                    return Err(PyValueError::new_err(format!("determinant of a {r}×{c} matrix")));
                }
                match self { $(AnyMat::$v(a) => a.determinant().map(|d| d.to_bigint()).ok_or_else(too_big)),* }
            }
            fn transpose(&self) -> AnyMat { match self { $(AnyMat::$v(a) => AnyMat::$v(a.transpose())),* } }
        }
    };
}

int_menu!(
    I2 2, I3 3, I4 4, I5 5, I6 6, I8 8, I10 10, I12 12, I16 16, I20 20, I24 24, I32 32, I48 48,
    I64 64,
);

fn mismatch() -> PyErr {
    PyTypeError::new_err("operands belong to different bases")
}

/// A choice of RNS base: the primes every value of a vector or matrix is kept
/// modulo, and so the largest absolute value they can hold.
///
/// `Base(bits=200)` picks the smallest base that holds signed integers of 200
/// bits; `Base(channels=9)` asks for a number of primes directly.
#[pyclass(frozen, module = "residue_rs")]
struct Base {
    kind: IntKind,
}

#[pymethods]
impl Base {
    #[new]
    #[pyo3(signature = (bits = None, *, channels = None))]
    fn new(bits: Option<u32>, channels: Option<usize>) -> PyResult<Self> {
        let kind = match (bits, channels) {
            (Some(b), None) => IntKind::ALL
                .iter()
                .copied()
                .find(|k| k.max_bits() >= b)
                .ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "no base holds {b} bits; the largest holds {}",
                        IntKind::ALL.last().unwrap().max_bits()
                    ))
                })?,
            (None, Some(c)) => IntKind::ALL
                .iter()
                .copied()
                .find(|k| k.channels() == c)
                .ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "no base of {c} primes; available: {:?}",
                        IntKind::ALL
                            .iter()
                            .map(|k| k.channels())
                            .collect::<Vec<_>>()
                    ))
                })?,
            _ => {
                return Err(PyTypeError::new_err(
                    "give exactly one of bits= or channels=",
                ))
            }
        };
        Ok(Base { kind })
    }

    /// Number of primes.
    #[getter]
    fn channels(&self) -> usize {
        self.kind.channels()
    }

    /// Largest bit length of a value (in absolute value) the base holds.
    #[getter]
    fn max_bits(&self) -> u32 {
        self.kind.max_bits()
    }

    /// The primes of the base.
    #[getter]
    fn primes(&self) -> Vec<u32> {
        self.kind.primes()
    }

    /// The numbers of primes available, smallest first.
    #[staticmethod]
    fn available() -> Vec<usize> {
        IntKind::ALL.iter().map(|k| k.channels()).collect()
    }

    /// A vector from Python ints, or from a one-dimensional int64 array.
    fn vector(&self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<Vector> {
        let (d, _) = read_ints(data, 1, self.width())?;
        let kind = self.kind;
        let v = py.detach(|| kind.vector(&d))?;
        Ok(Vector { v })
    }

    /// A matrix from a list of rows of Python ints, or from a two-dimensional
    /// int64 array.
    fn matrix(&self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<Matrix> {
        let (d, shape) = read_ints(data, 2, self.width())?;
        let kind = self.kind;
        let m = py.detach(|| kind.matrix(shape[0], shape[1], &d))?;
        Ok(Matrix { m })
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .cast::<Base>()
            .is_ok_and(|o| o.get().kind == self.kind)
    }

    fn __hash__(&self) -> u64 {
        self.kind.channels() as u64
    }

    fn __repr__(&self) -> String {
        format!(
            "Base(channels={}, max_bits={})",
            self.kind.channels(),
            self.kind.max_bits()
        )
    }
}

impl Base {
    /// Limbs of the two's complement form of any value the base holds.
    fn width(&self) -> usize {
        (self.kind.max_bits() as usize + 1).div_ceil(64)
    }
}

/// A vector of exact signed integers, all kept modulo the primes of one base.
///
/// Arithmetic is exact: an operation whose result could leave the range of
/// the base raises `OverflowError` instead of wrapping.
#[pyclass(frozen, module = "residue_rs", sequence)]
struct Vector {
    v: AnyVec,
}

fn other_vector<'a>(o: &'a Bound<'_, PyAny>) -> Option<&'a Bound<'a, Vector>> {
    o.cast::<Vector>().ok()
}

#[pymethods]
impl Vector {
    fn __len__(&self) -> usize {
        self.v.len()
    }

    fn __getitem__(&self, i: isize) -> PyResult<BigInt> {
        let n = self.v.len() as isize;
        let k = if i < 0 { i + n } else { i };
        if k < 0 || k >= n {
            return Err(PyIndexError::new_err("index out of range"));
        }
        Ok(self.v.get(k as usize).unwrap())
    }

    fn __add__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binary(py, o, Op::Add)
    }

    fn __sub__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binary(py, o, Op::Sub)
    }

    /// Element-wise product with another vector, or product by an int.
    fn __mul__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if other_vector(o).is_none() {
            if let Ok(k) = o.extract::<BigInt>() {
                let v = py.detach(|| self.v.scale(&k))?;
                return Ok(Py::new(py, Vector { v })?.into_any());
            }
        }
        self.binary(py, o, Op::Mul)
    }

    fn __rmul__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.__mul__(py, o)
    }

    fn __neg__(&self, py: Python<'_>) -> Vector {
        Vector {
            v: py.detach(|| self.v.neg()),
        }
    }

    /// Exact dot product, as a Python int.
    fn dot(&self, py: Python<'_>, other: &Bound<'_, Vector>) -> PyResult<BigInt> {
        let o = &other.get().v;
        py.detach(|| self.v.dot(o))
    }

    /// Exact sum, as a Python int.
    fn sum(&self, py: Python<'_>) -> PyResult<BigInt> {
        py.detach(|| self.v.sum())
    }

    /// The values as a list of Python ints.
    fn tolist<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let o = py.detach(|| self.v.out());
        out_list(py, o)
    }

    /// The values as a NumPy int64 array; `OverflowError` if one does not fit.
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let x = py.detach(|| self.v.i64s())?;
        numpy_i64(py, &x, &[x.len()])
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(self.tolist(py)?.try_iter()?.into_any())
    }

    /// The base of the vector.
    #[getter]
    fn base(&self) -> Base {
        Base {
            kind: self.v.kind(),
        }
    }

    /// A certified bound on the bit length of every value.
    #[getter]
    fn bits_bound(&self) -> u32 {
        self.v.bits_bound()
    }

    fn __repr__(&self) -> String {
        let n = self.v.len();
        let shown: Vec<String> = (0..n.min(6))
            .map(|k| self.v.get(k).unwrap().to_string())
            .collect();
        let more = if n > 6 { ", ..." } else { "" };
        format!(
            "Vector([{}{more}], channels={})",
            shown.join(", "),
            self.v.kind().channels()
        )
    }
}

impl Vector {
    fn binary(&self, py: Python<'_>, o: &Bound<'_, PyAny>, op: Op) -> PyResult<Py<PyAny>> {
        let Some(o) = other_vector(o) else {
            return Ok(py.NotImplemented());
        };
        let o = &o.get().v;
        let v = py.detach(|| self.v.binop(o, op))?;
        Ok(Py::new(py, Vector { v })?.into_any())
    }
}

/// A matrix of exact signed integers, kept modulo the primes of one base.
#[pyclass(frozen, module = "residue_rs")]
struct Matrix {
    m: AnyMat,
}

#[pymethods]
impl Matrix {
    /// `(rows, cols)`.
    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.m.shape()
    }

    /// The base of the matrix.
    #[getter]
    fn base(&self) -> Base {
        Base {
            kind: self.m.kind(),
        }
    }

    fn __matmul__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let Ok(o) = o.cast::<Matrix>() else {
            return Ok(py.NotImplemented());
        };
        let o = &o.get().m;
        let m = py.detach(|| self.m.matmul(o))?;
        Ok(Py::new(py, Matrix { m })?.into_any())
    }

    fn __add__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.elementwise(py, o, Op::Add)
    }

    fn __sub__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.elementwise(py, o, Op::Sub)
    }

    fn __neg__(&self, py: Python<'_>) -> Matrix {
        let (r, c) = self.m.shape();
        Matrix {
            m: AnyMat::from_vec(r, c, py.detach(|| self.m.vec().neg())),
        }
    }

    /// The transpose.
    #[getter(T)]
    fn transpose(&self, py: Python<'_>) -> Matrix {
        Matrix {
            m: py.detach(|| self.m.transpose()),
        }
    }

    /// The exact determinant, as a Python int (`OverflowError` if Hadamard's
    /// bound exceeds the base).
    fn det(&self, py: Python<'_>) -> PyResult<BigInt> {
        py.detach(|| self.m.det())
    }

    fn __getitem__(&self, idx: (usize, usize)) -> PyResult<BigInt> {
        let (r, c) = self.m.shape();
        if idx.0 >= r || idx.1 >= c {
            return Err(PyIndexError::new_err("index out of range"));
        }
        Ok(self.m.vec().get(idx.0 * c + idx.1).unwrap())
    }

    /// The entries as a list of rows of Python ints.
    fn tolist<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let (r, c) = self.m.shape();
        let flat = out_list(py, py.detach(|| self.m.vec().out()))?;
        let rows = (0..r)
            .map(|i| flat.get_slice(i * c, (i + 1) * c))
            .collect::<Vec<_>>();
        PyList::new(py, rows)
    }

    /// The entries as a two-dimensional NumPy int64 array.
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let (r, c) = self.m.shape();
        let x = py.detach(|| self.m.vec().i64s())?;
        numpy_i64(py, &x, &[r, c])
    }

    fn __repr__(&self) -> String {
        let (r, c) = self.m.shape();
        format!(
            "Matrix(shape=({r}, {c}), channels={})",
            self.m.kind().channels()
        )
    }
}

impl Matrix {
    fn elementwise(&self, py: Python<'_>, o: &Bound<'_, PyAny>, op: Op) -> PyResult<Py<PyAny>> {
        let Ok(o) = o.cast::<Matrix>() else {
            return Ok(py.NotImplemented());
        };
        let o = &o.get().m;
        if o.shape() != self.m.shape() {
            return Err(PyValueError::new_err("shape mismatch"));
        }
        let (r, c) = self.m.shape();
        let v = py.detach(|| self.m.vec().binop(&o.vec(), op))?;
        Ok(Py::new(
            py,
            Matrix {
                m: AnyMat::from_vec(r, c, v),
            },
        )?
        .into_any())
    }
}

/// `x` limbs, least significant first, of a non-negative integer.
fn limbs(x: &BigUint, width: usize) -> Vec<u64> {
    let mut d = x.to_u64_digits();
    d.resize(width, 0);
    d
}

fn from_limbs(l: &[u64]) -> BigUint {
    BigUint::new(
        l.iter()
            .flat_map(|&x| [x as u32, (x >> 32) as u32])
            .collect(),
    )
}

macro_rules! mont_menu {
    ($($v:ident $n:literal),* $(,)?) => {
        /// Pairs of disjoint bases for modular arithmetic, `$n + $n` primes.
        mod mb {
            $(
                #[allow(non_snake_case)]
                pub mod $v {
                    residue::rns_base!(pub A: largest $n);
                    residue::rns_base!(pub B: largest $n after $n);
                }
            )*
        }

        #[allow(clippy::large_enum_variant)]
        enum AnyMont { $($v(MontRns<mb::$v::A, $n, mb::$v::B, $n>)),* }

        impl AnyMont {
            /// The smallest pair of bases that fits `m`.
            fn new(m: &[u64]) -> PyResult<AnyMont> {
                $(
                    match MontRns::<mb::$v::A, $n, mb::$v::B, $n>::new(m) {
                        Ok(c) => return Ok(AnyMont::$v(c)),
                        Err(MontRnsError::BaseTooSmall { .. }) => {}
                        Err(e) => return Err(PyValueError::new_err(e.to_string())),
                    }
                )*
                Err(PyValueError::new_err("modulus too large: at most 4096 bits are supported"))
            }

            fn channels(&self) -> usize { match self { $(AnyMont::$v(_) => $n),* } }

            /// `x^e` for every `x` (canonical limbs, `w` per element).
            fn pow(&self, xs: &[u64], w: usize, e: &[u64]) -> Vec<u64> {
                match self { $(AnyMont::$v(c) => ModVec::from_limbs(c, w, xs).pow(e).to_limbs()),* }
            }

            /// `x·y` element by element.
            fn mul(&self, xs: &[u64], ys: &[u64], w: usize) -> Vec<u64> {
                match self {
                    $(AnyMont::$v(c) => (&ModVec::from_limbs(c, w, xs) * &ModVec::from_limbs(c, w, ys)).to_limbs()),*
                }
            }
        }
    };
}

mont_menu!(M5 5, M9 9, M18 18, M34 34, M51 51, M67 67, M100 100, M133 133);

/// Arithmetic modulo an odd number of up to 4096 bits, many elements at a
/// time (RNS Montgomery multiplication).
///
/// Worth it for batches: one element at a time, Python's `pow(x, e, m)` (or
/// gmpy2) is faster. **Not constant time**: do not use with secret data.
#[pyclass(frozen, module = "residue_rs")]
struct Modulus {
    ctx: AnyMont,
    m: BigUint,
    width: usize,
}

impl Modulus {
    fn reduce(&self, obj: &Bound<'_, PyAny>) -> PyResult<Vec<u64>> {
        let m = BigInt::from_biguint(Sign::Plus, self.m.clone());
        let mut flat = Vec::new();
        for x in obj.try_iter()? {
            let x: BigInt = x?.extract()?;
            let r = x.mod_floor(&m);
            flat.extend(limbs(r.magnitude(), self.width));
        }
        Ok(flat)
    }

    fn results<'py>(&self, py: Python<'py>, flat: &[u64]) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, flat.chunks(self.width).map(from_limbs))
    }
}

#[pymethods]
impl Modulus {
    #[new]
    fn new(py: Python<'_>, m: BigInt) -> PyResult<Self> {
        if !m.is_positive() || m.is_even() {
            return Err(PyValueError::new_err(
                "the modulus must be a positive odd integer",
            ));
        }
        let m = m.into_parts().1;
        let width = m.to_u64_digits().len();
        let l = m.to_u64_digits();
        let ctx = py.detach(|| AnyMont::new(&l))?;
        Ok(Modulus { ctx, m, width })
    }

    /// The modulus.
    #[getter]
    fn modulus(&self) -> BigUint {
        self.m.clone()
    }

    /// Number of primes in each of the two bases.
    #[getter]
    fn channels(&self) -> usize {
        self.ctx.channels()
    }

    /// `[pow(x, e, m) for x in xs]`, computed together.
    fn pow_batch<'py>(
        &self,
        py: Python<'py>,
        xs: &Bound<'py, PyAny>,
        e: BigInt,
    ) -> PyResult<Bound<'py, PyList>> {
        if e.is_negative() {
            return Err(PyValueError::new_err("negative exponent"));
        }
        let flat = self.reduce(xs)?;
        let e = e.magnitude().to_u64_digits();
        let out = py.detach(|| self.ctx.pow(&flat, self.width, &e));
        self.results(py, &out)
    }

    /// `[x * y % m for x, y in zip(xs, ys)]`, computed together.
    fn mul_batch<'py>(
        &self,
        py: Python<'py>,
        xs: &Bound<'py, PyAny>,
        ys: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyList>> {
        let (a, b) = (self.reduce(xs)?, self.reduce(ys)?);
        if a.len() != b.len() {
            return Err(PyValueError::new_err("length mismatch"));
        }
        let out = py.detach(|| self.ctx.mul(&a, &b, self.width));
        self.results(py, &out)
    }

    fn __repr__(&self) -> String {
        format!(
            "Modulus(<{} bits>, channels={})",
            self.m.bits(),
            self.ctx.channels()
        )
    }
}

/// Exact integer arithmetic in a Residue Number System: vectors and matrices
/// of big integers, and batches of modular exponentiations.
#[pymodule]
fn residue_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Base>()?;
    m.add_class::<Vector>()?;
    m.add_class::<Matrix>()?;
    m.add_class::<Modulus>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
