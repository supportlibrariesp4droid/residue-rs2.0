//! Optional parallelism across channels and across blocks of elements.
//!
//! With the `parallel` feature, operations whose work is large enough split it
//! across the threads of rayon's global pool; without it, or below the
//! threshold, the same code runs sequentially on the calling thread. Every
//! task still enters its SIMD kernel through [`super::kernels::dispatch`], so
//! each thread runs the AVX2 or AVX-512 code.
//!
//! The channels of an RNS number are independent, which is what makes this
//! simple: a matrix product or a determinant is `N` unrelated computations,
//! one per prime, and the only shared step is the final reconstruction.

use alloc::vec::Vec;

#[cfg(feature = "parallel")]
extern crate std;

/// Work, counted in word multiplications, below which an operation stays on
/// one thread. `2^19` of them take roughly 150 to 300 µs; with less, waking
/// the pool's threads cost more than it saved on the 2-vCPU x86-64 VM where
/// this was measured (a threshold of `2^17` made conversions of 4096 values
/// 25 percent slower with two threads). Call sites weight operations that cost
/// more per multiplication, such as elimination.
#[cfg(feature = "parallel")]
pub(crate) const MIN_PAR_WORK: usize = 1 << 19;

/// The threshold in force; tests lower it to reach the parallel code with
/// small inputs.
#[cfg(feature = "parallel")]
static THRESHOLD: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(MIN_PAR_WORK);

/// Test hook: set the parallel threshold (`MIN_PAR_WORK` by default). Not part
/// of the API.
#[cfg(feature = "parallel")]
#[doc(hidden)]
pub fn __set_parallel_threshold(work: usize) {
    THRESHOLD.store(work, core::sync::atomic::Ordering::Relaxed);
}

/// Whether an operation of this much work should run in parallel.
#[inline]
pub(crate) fn enabled(work: usize) -> bool {
    #[cfg(feature = "parallel")]
    {
        work >= THRESHOLD.load(core::sync::atomic::Ordering::Relaxed)
            && usable()
            && rayon::current_num_threads() > 1
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = work;
        false
    }
}

/// Whether rayon can run work on more than one thread here.
///
/// Inside a rayon pool it can. Otherwise the global pool is used, and rayon
/// panics where threads cannot be created: on WebAssembly without threads,
/// `std` reports the failure as `WouldBlock`, which rayon's fallback to a
/// single thread does not recognize. So the global pool is touched only on
/// targets that can have threads, and only when the system reports more than
/// one hardware thread (with one, there is nothing to gain).
#[cfg(feature = "parallel")]
fn usable() -> bool {
    if rayon::current_thread_index().is_some() {
        return true;
    }
    static GLOBAL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GLOBAL.get_or_init(|| {
        cfg!(not(all(
            target_family = "wasm",
            not(target_feature = "atomics")
        ))) && std::thread::available_parallelism().is_ok_and(|n| n.get() > 1)
    })
}

/// Number of threads that parallel work is spread over (1 without the
/// feature, or where threads are not usable).
#[inline]
pub(crate) fn threads() -> usize {
    #[cfg(feature = "parallel")]
    {
        if usable() {
            rayon::current_num_threads()
        } else {
            1
        }
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}

/// `f(i, &mut items[i])` for every item, in parallel when `par` is set.
#[inline]
pub(crate) fn for_each_mut<T: Send>(
    items: &mut [T],
    par: bool,
    f: impl Fn(usize, &mut T) + Sync + Send,
) {
    #[cfg(feature = "parallel")]
    if par {
        use rayon::prelude::*;
        items.par_iter_mut().enumerate().for_each(|(i, t)| f(i, t));
        return;
    }
    let _ = par;
    for (i, t) in items.iter_mut().enumerate() {
        f(i, t);
    }
}

/// `[f(0), …, f(n−1)]`, computed in parallel when `par` is set.
#[inline]
pub(crate) fn map<R: Send>(n: usize, par: bool, f: impl Fn(usize) -> R + Sync + Send) -> Vec<R> {
    #[cfg(feature = "parallel")]
    if par {
        use rayon::prelude::*;
        return (0..n).into_par_iter().map(f).collect();
    }
    let _ = par;
    (0..n).map(f).collect()
}

/// `(a(), b())`, side by side when `par` is set.
#[inline]
pub(crate) fn join<A: Send, B: Send>(
    par: bool,
    a: impl FnOnce() -> A + Send,
    b: impl FnOnce() -> B + Send,
) -> (A, B) {
    #[cfg(feature = "parallel")]
    if par {
        return rayon::join(a, b);
    }
    let _ = par;
    (a(), b())
}
