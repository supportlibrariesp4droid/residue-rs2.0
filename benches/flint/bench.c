#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <gmp.h>
#include <flint/flint.h>
#include <flint/fmpz.h>
#include <flint/fmpz_vec.h>
#include <flint/fmpz_mat.h>

typedef unsigned __int128 u128;
typedef __int128 i128;

static uint64_t state;
static uint64_t next(void) {
    state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static i128 random_i128(unsigned bits) {
    uint64_t hi = next(), lo = next();
    u128 mag = (((u128)hi << 64) | lo) >> (128 - bits);
    i128 v = (i128)mag;
    return (next() & 1) ? -v : v;
}

static i128 *values(uint64_t seed, long n, unsigned bits) {
    i128 *v = malloc(sizeof(i128) * n);
    state = seed;
    for (long i = 0; i < n; i++) v[i] = random_i128(bits);
    return v;
}

static void set_i128(fmpz_t f, i128 x) {
    u128 u = (u128)x;
    fmpz_set_signed_uiui(f, (ulong)(u >> 64), (ulong)u);
}

static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}

static int compare_doubles(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

#define SAMPLES 20
static double measure(void (*f)(void)) {
    double start = now_ns();
    long calls = 0;
    while (now_ns() - start < 5e8) {
        f();
        calls++;
    }
    long per_sample = calls / 5 + 1;
    double s[SAMPLES];
    for (int i = 0; i < SAMPLES; i++) {
        double a = now_ns();
        for (long j = 0; j < per_sample; j++) f();
        s[i] = (now_ns() - a) / per_sample;
    }
    qsort(s, SAMPLES, sizeof(double), compare_doubles);
    return (s[SAMPLES / 2 - 1] + s[SAMPLES / 2]) / 2;
}

static const char *out_dir;
static void save(const char *group, const char *bench, double ns) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", out_dir, group);
    mkdir(path, 0755);
    snprintf(path, sizeof path, "%s/%s/%s", out_dir, group, bench);
    mkdir(path, 0755);
    snprintf(path, sizeof path, "%s/%s/%s/new", out_dir, group, bench);
    mkdir(path, 0755);
    snprintf(path, sizeof path, "%s/%s/%s/new/estimates.json", out_dir, group, bench);
    FILE *f = fopen(path, "w");
    if (!f) {
        perror(path);
        exit(1);
    }
    fprintf(f, "{\"median\":{\"point_estimate\":%.3f}}\n", ns);
    fclose(f);
}

static void report(const char *group, double plain, double e2e) {
    save(group, "flint", plain);
    save(group, "flint_e2e", e2e);
    printf("%-18s %12.1f ns %12.1f ns from i128   check ", group, plain, e2e);
}

static ulong low_xor(const fmpz *v, long n) {
    fmpz_t t;
    fmpz_init(t);
    ulong h = 0;
    for (long i = 0; i < n; i++) {
        fmpz_abs(t, v + i);
        fmpz_fdiv_r_2exp(t, t, 64);
        h ^= fmpz_get_ui(t);
    }
    fmpz_clear(t);
    return h;
}

#define N 4096
static i128 *xa, *xb;
static fmpz *fa, *fb, *fo;
static fmpz_t acc;

static void a_plain(void) {
    for (long i = 0; i < N; i++) fmpz_mul(fo + i, fa + i, fb + i);
}
static void a_e2e(void) {
    for (long i = 0; i < N; i++) {
        set_i128(fa + i, xa[i]);
        set_i128(fb + i, xb[i]);
        fmpz_mul(fo + i, fa + i, fb + i);
    }
}

static void b_plain(void) { _fmpz_vec_dot(acc, fa, fb, N); }
static void b_e2e(void) {
    for (long i = 0; i < N; i++) {
        set_i128(fa + i, xa[i]);
        set_i128(fb + i, xb[i]);
    }
    _fmpz_vec_dot(acc, fa, fb, N);
}

static ulong la[16], lb[16], digits[32];
static fmpz_t ea, eb, eo;
static void e_plain(void) { fmpz_mul(eo, ea, eb); }
static void e_e2e(void) {
    fmpz_set_ui_array(ea, la, 16);
    fmpz_set_ui_array(eb, lb, 16);
    fmpz_mul(eo, ea, eb);
    fmpz_get_ui_array(digits, 32, eo);
}

static long dim;
static i128 *ma, *mb;
static fmpz_mat_t A, B, C;
static void load(fmpz_mat_t m, const i128 *x) {
    for (long i = 0; i < dim; i++)
        for (long j = 0; j < dim; j++) set_i128(fmpz_mat_entry(m, i, j), x[i * dim + j]);
}
static void f_plain(void) { fmpz_mat_mul(C, A, B); }
static void f_e2e(void) {
    load(A, ma);
    load(B, mb);
    fmpz_mat_mul(C, A, B);
}
static void det_plain(void) { fmpz_mat_det(acc, A); }
static void det_e2e(void) {
    load(A, ma);
    fmpz_mat_det(acc, A);
}

static void elementwise_mul(void) {
    xa = values(1, N, 126);
    xb = values(2, N, 126);
    fa = _fmpz_vec_init(N);
    fb = _fmpz_vec_init(N);
    fo = _fmpz_vec_init(N);
    for (long i = 0; i < N; i++) {
        set_i128(fa + i, xa[i]);
        set_i128(fb + i, xb[i]);
    }
    double p = measure(a_plain), e = measure(a_e2e);
    report("A_elementwise_mul", p, e);
    printf("%016lx\n", low_xor(fo, N));
    _fmpz_vec_clear(fa, N);
    _fmpz_vec_clear(fb, N);
    _fmpz_vec_clear(fo, N);
}

static void dot_product(void) {
    xa = values(3, N, 100);
    xb = values(4, N, 100);
    fa = _fmpz_vec_init(N);
    fb = _fmpz_vec_init(N);
    for (long i = 0; i < N; i++) {
        set_i128(fa + i, xa[i]);
        set_i128(fb + i, xb[i]);
    }
    double p = measure(b_plain), e = measure(b_e2e);
    report("B_dot_product", p, e);
    fmpz_print(acc);
    printf("\n");
    _fmpz_vec_clear(fa, N);
    _fmpz_vec_clear(fb, N);
}

static void single_large_mul(void) {
    state = 9;
    for (int i = 0; i < 16; i++) la[i] = next();
    for (int i = 0; i < 16; i++) lb[i] = next();
    fmpz_init(ea);
    fmpz_init(eb);
    fmpz_init(eo);
    fmpz_set_ui_array(ea, la, 16);
    fmpz_set_ui_array(eb, lb, 16);
    double p = measure(e_plain), e = measure(e_e2e);
    report("E_single_mul_1024", p, e);
    printf("%016lx\n", (unsigned long)digits[0]);
}

static void matmul(void) {
    dim = 64;
    ma = values(11, dim * dim, 62);
    mb = values(12, dim * dim, 62);
    fmpz_mat_init(A, dim, dim);
    fmpz_mat_init(B, dim, dim);
    fmpz_mat_init(C, dim, dim);
    load(A, ma);
    load(B, mb);
    double p = measure(f_plain), e = measure(f_e2e);
    ulong h = 0;
    for (long i = 0; i < dim; i++) h ^= low_xor(fmpz_mat_row(C, i), dim);
    report("F_matmul_64", p, e);
    printf("%016lx\n", h);
    fmpz_mat_clear(A);
    fmpz_mat_clear(B);
    fmpz_mat_clear(C);
}

static void determinant(const char *group, long n, unsigned bits) {
    dim = n;
    ma = values(13 + n, n * n, bits);
    fmpz_mat_init(A, dim, dim);
    load(A, ma);
    double p = measure(det_plain), e = measure(det_e2e);
    report(group, p, e);
    fmpz_print(acc);
    printf("\n");
    fmpz_mat_clear(A);
}

int main(int argc, char **argv) {
    out_dir = argc > 1 ? argv[1] : "target/criterion";
    mkdir(out_dir, 0755);
    fmpz_init(acc);
    elementwise_mul();
    dot_product();
    single_large_mul();
    matmul();
    determinant("G_det_32", 32, 32);
    determinant("H_det_64", 64, 16);
    printf("FLINT %s, GMP %s\n", FLINT_VERSION, gmp_version);
    return 0;
}
