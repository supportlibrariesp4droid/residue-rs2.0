#!/usr/bin/env python3
"""Draw the benchmark charts in docs/ from measured results.

Usage:
    cargo bench --bench compare
    cc -O3 -march=native benches/flint/bench.c -lflint -lgmp -o target/flint-bench
    target/flint-bench target/criterion
    cargo bench --bench modular
    python3 scripts/plot_benchmarks.py [CRITERION_DIR] [OUT_DIR]

CRITERION_DIR defaults to target/criterion and OUT_DIR to docs. Every number in
the charts is read from measurement output (the median of each benchmark in
criterion's estimates.json files); nothing is typed in by hand. The FLINT
program writes its medians in the same layout, and FLINT is left out of the
charts if it has not been run.

Two more charts are drawn when their data files exist. OUT_DIR/data/wasm.json
holds medians of examples/timing.rs run natively and in wasmtime, written by

    python3 scripts/plot_benchmarks.py timing-json docs/data/wasm.json \\
        native=a.txt,b.txt wasm=c.txt,d.txt simd128=e.txt,f.txt

and OUT_DIR/data/python.json is written by `python bench/bench.py --json`
in python/. Each chart is written twice, for light and dark page backgrounds.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

N = 4096

MACHINE = (
    "Intel Xeon 2.8 GHz cloud VM, 2 vCPUs, AVX-512, rustc 1.97, single thread. "
    "GMP 6.3.0 through rug 1.30, FLINT 3.6.0 built with -march=native, num-bigint 0.4.8. Medians of three sessions."
)

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "residue": "#2a78d6",
        "gmp": "#eb6834",
        "num": "#1baf7a",
        "flint": "#eda100",
        "native": "#898781",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "residue": "#3987e5",
        "gmp": "#d95926",
        "num": "#199e70",
        "flint": "#c98500",
        "native": "#898781",
    },
}

LIBRARIES = [
    ("residue", "residue", "residue"),
    ("gmp", "GMP", "gmp"),
    ("flint", "FLINT", "flint"),
    ("num_bigint", "num-bigint", "num"),
    ("native", "native i128", "native"),
]

SCENARIOS = [
    ("A_elementwise_mul", "A. Element-wise product, 126-bit values", "ns per element", N),
    ("B_dot_product", "B. Exact dot product, 100-bit values", "ns per element", N),
    ("C_small_dot", "C. Dot product, 50-bit values (fits in i128)", "ns per element", N),
    ("E_single_mul_1024", "E. One 1024 by 1024-bit product", "ns", 1),
    ("F_matmul_64", "F. Matrix product 64×64, 62-bit entries", "ms", 1e6),
    ("G_det_32", "G. Determinant 32×32, 32-bit entries", "ms", 1e6),
    ("H_det_64", "H. Determinant 64×64, 16-bit entries", "ms", 1e6),
]


def load(root):
    data = {}
    for group in os.listdir(root):
        gdir = os.path.join(root, group)
        if not os.path.isdir(gdir):
            continue
        for bench in os.listdir(gdir):
            est = os.path.join(gdir, bench, "new", "estimates.json")
            if os.path.exists(est):
                with open(est) as f:
                    data.setdefault(group, {})[bench] = json.load(f)["median"]["point_estimate"]
    return data


def pick_font():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Inter", "DejaVu Sans"):
        if name in names:
            return name
    return "sans-serif"


def style(theme):
    t = THEMES[theme]
    plt.rcParams.update(
        {
            "font.family": pick_font(),
            "font.size": 10,
            "figure.facecolor": t["surface"],
            "axes.facecolor": t["surface"],
            "savefig.facecolor": t["surface"],
            "text.color": t["ink"],
            "axes.labelcolor": t["ink2"],
            "xtick.color": t["muted"],
            "ytick.color": t["ink2"],
            "axes.edgecolor": t["axis"],
        }
    )
    return t


def tint(color, alpha=0.28):
    """A light fill of the same hue, for the 'starting from i128' variant."""
    from matplotlib.colors import to_rgba

    r, g, b, _ = to_rgba(color)
    return (r, g, b, alpha)


def ratio_text(r, other):
    if 1 / 1.05 <= r <= 1.05:
        return f"on par with {other}"
    if r >= 1:
        text = f"{r:.1f}× faster" if r < 10 else f"{r:.0f}× faster"
    else:
        s = 1 / r
        text = f"{s:.1f}× slower" if s < 10 else f"{s:.0f}× slower"
    return f"{text} than {other}"


NAMES = {bench: name for bench, name, _ in LIBRARIES}
RIVALS = ("gmp", "flint", "num_bigint")


def fastest(g, suffix):
    """The fastest library other than residue in one variant of a scenario."""
    times = {lib: g[lib + suffix] for lib in RIVALS if lib + suffix in g}
    lib = min(times, key=times.get)
    return NAMES[lib], times[lib]


def speedup_chart(data, theme, path):
    """Running time of the fastest other library divided by residue's, per scenario."""
    t = style(theme)
    rows = []
    for key, title, _, _ in SCENARIOS:
        g = data[key]
        label = title.split(". ", 1)[1].replace(" (fits in i128)", "")
        if key == "C_small_dot":
            label += "\n(against native i128)"
            plain = e2e = ("i128", g["native"])
        else:
            plain, e2e = fastest(g, ""), fastest(g, "_e2e")
        rows.append(
            (
                title.split(".")[0] + ". " + label,
                (plain[1] / g["residue"], plain[0]),
                (e2e[1] / g["residue_e2e"], e2e[0]),
            )
        )

    fig, ax = plt.subplots(figsize=(10, 6.4), dpi=200)
    fig.subplots_adjust(left=0.30, right=0.95, top=0.80, bottom=0.14)
    h = 0.34
    for i, (label, a, b) in enumerate(rows):
        y = len(rows) - 1 - i
        for dy, (v, other), solid in ((h / 2 + 0.02, a, True), (-h / 2 - 0.02, b, False)):
            color = t["residue"]
            ax.barh(
                y + dy,
                v - 1 if v >= 1 else -(1 - v),
                left=1,
                height=h,
                color=color if solid else tint(color),
                edgecolor=color,
                linewidth=0 if solid else 1.2,
            )
            inside = v < 1 / 20
            x_text = v * 1.12 if v >= 1 or inside else v / 1.12
            ax.text(
                x_text,
                y + dy,
                ratio_text(v, other),
                va="center",
                ha="left" if v >= 1 or inside else "right",
                fontsize=9,
                color=t["ink"],
            )
    ax.set_xscale("log")
    ax.set_xlim(0.005, 60)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=9.5)
    ax.tick_params(axis="y", length=0)
    ax.axvline(1, color=t["ink2"], linewidth=1)
    ax.set_xticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50])
    ax.set_xticklabels(["1/100", "1/50", "1/20", "1/10", "1/5", "1/2", "1", "2", "5", "10", "20", "50"])
    ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.set_xlabel("speed of residue relative to the fastest other library (log scale)", fontsize=9.5)
    ax.text(0.9, len(rows) - 0.35, "the other library is faster", ha="right", fontsize=9, color=t["ink2"])
    ax.text(1.1, len(rows) - 0.35, "residue is faster", ha="left", fontsize=9, color=t["ink2"])
    ax.set_ylim(-0.6, len(rows) - 0.1)

    fig.text(0.03, 0.94, "residue compared with the fastest alternative", fontsize=15, weight="semibold")
    fig.text(
        0.03,
        0.885,
        "Ratio of running times against whichever of GMP, FLINT and num-bigint is fastest in each case. "
        "Row C compares\nwith native i128, which is enough for 50-bit values.",
        fontsize=9.5,
        color=t["ink2"],
    )
    handles = [
        Patch(facecolor=t["residue"], label="data already in each library's format"),
        Patch(
            facecolor=tint(t["residue"]),
            edgecolor=t["residue"],
            label="starting from i128 values, result back in binary",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.295, 0.865),
        ncol=2,
        frameon=False,
        fontsize=9,
        handlelength=1.4,
    )
    fig.text(0.03, 0.035, MACHINE, fontsize=7.5, color=t["muted"])
    fig.savefig(path)
    plt.close(fig)


def fmt_value(v, unit):
    if unit == "ms":
        return f"{v:.2f} ms" if v < 10 else f"{v:.1f} ms"
    if v >= 1000:
        return f"{v:,.0f} {unit.split()[0]}"
    return f"{v:.1f} {unit.split()[0]}"


def absolute_chart(data, theme, path):
    """Small multiples: absolute times per scenario, every library, both variants."""
    t = style(theme)
    fig, axes = plt.subplots(4, 2, figsize=(11, 16), dpi=200)
    fig.subplots_adjust(left=0.13, right=0.96, top=0.905, bottom=0.07, hspace=0.62, wspace=0.42)
    flat = axes.flatten()
    for ax, (key, title, unit, scale) in zip(flat, SCENARIOS):
        g = data[key]
        bars = []
        for bench, name, ckey in LIBRARIES:
            if bench in g:
                bars.append((name, g[bench] / scale, t[ckey], True))
            if bench + "_e2e" in g:
                bars.append((name + ", from i128", g[bench + "_e2e"] / scale, t[ckey], False))
        n = len(bars)
        vmax = max(b[1] for b in bars)
        slots = 8
        for i, (label, v, color, solid) in enumerate(bars):
            y = slots - 1 - i
            ax.barh(
                y,
                v,
                height=0.72,
                color=color if solid else tint(color),
                edgecolor=color,
                linewidth=0 if solid else 1.1,
            )
            ax.text(v + vmax * 0.02, y, fmt_value(v, unit), va="center", fontsize=8.5, color=t["ink"])
        ax.set_yticks([slots - 1 - i for i in range(n)])
        ax.set_yticklabels([b[0] for b in bars], fontsize=8.5)
        ax.set_ylim(-0.6, slots - 0.4)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, vmax * 1.32)
        ax.grid(axis="x", color=t["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(t["axis"])
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlabel(unit + ", lower is better", fontsize=8.5)
        ax.set_title(title, loc="left", fontsize=10.5, weight="semibold", pad=8)

    legend_ax = flat[-1]
    legend_ax.axis("off")
    handles = [
        Patch(facecolor=t["residue"], label="residue"),
        Patch(facecolor=t["gmp"], label="GMP"),
        Patch(facecolor=t["flint"], label="FLINT"),
        Patch(facecolor=t["num"], label="num-bigint"),
        Patch(facecolor=t["native"], label="native i128"),
    ]
    legend_ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9.5, handlelength=1.4)
    legend_ax.text(
        0.0,
        0.38,
        "Solid bars start from data already in each\n"
        "library's own format. Light bars, marked\n"
        "\"from i128\", start from i128 values and\n"
        "convert the result back to binary, so each\n"
        "library also pays for its conversions.",
        transform=legend_ax.transAxes,
        fontsize=9,
        color=t["ink2"],
        va="top",
        linespacing=1.5,
    )

    fig.text(0.04, 0.955, "Running times, residue against GMP, FLINT and num-bigint", fontsize=15, weight="semibold")
    fig.text(
        0.04,
        0.935,
        "Each panel has its own scale and unit. GMP gets its best case: preallocated values, "
        "in-place assignment, fused addmul and\nsubmul. FLINT runs its own default algorithms, "
        "fmpz_mat_mul and fmpz_mat_det for the matrices.",
        fontsize=9.5,
        color=t["ink2"],
    )
    fig.text(0.04, 0.025, MACHINE, fontsize=7.5, color=t["muted"])
    fig.savefig(path)
    plt.close(fig)


CROSSOVER_PANELS = [
    (
        "A_elementwise_mul",
        "Element-wise products, 126-bit values",
        "Two vectors converted in, k products with them,\none vector converted back.",
        N,
        1e3,
        "million products per second, higher is faster",
        "{:.0f}",
    ),
    (
        "B_dot_product",
        "Dot products, 100-bit values",
        "Two vectors converted in, k dot products\ncomputed from them.",
        N,
        1e3,
        "million multiply-adds per second, higher is faster",
        "{:.0f}",
    ),
    (
        "F_matmul_64",
        "Matrix products 64×64, 62-bit entries",
        "Two matrices converted in, k matrix products\ncomputed from them, results converted back.",
        1,
        1e6,
        "thousand matrix products per second, higher is faster",
        "{:.1f}",
    ),
]

CROSSOVER_LABELS = {
    ("A_elementwise_mul", "num_bigint"): (150, 62),
    ("A_elementwise_mul", "flint"): (24, 100),
    ("A_elementwise_mul", "gmp"): (60, 215),
    ("B_dot_product", "gmp"): (1.12, 140),
    ("B_dot_product", "flint"): (30, 160),
}


def costs(g, per):
    """Per unit: (conversion cost, cost of one operation) for each library present."""
    out = {}
    for lib in ("residue",) + RIVALS:
        if lib in g and lib + "_e2e" in g:
            op = g[lib] / per
            out[lib] = (max(0.0, g[lib + "_e2e"] / per - op), op)
    return out


def crossing(a, b):
    """The k >= 1 where conv_a + k·op_a = conv_b + k·op_b, if there is one."""
    (ca, oa), (cb, ob) = a, b
    if oa == ob:
        return None
    k = (cb - ca) / (oa - ob)
    return k if k >= 1 else None


def crossover_chart(data, theme, path):
    """Speed as a function of k, the number of operations per conversion."""
    t = style(theme)
    colors = {"residue": t["residue"], "gmp": t["gmp"], "flint": t["flint"], "num_bigint": t["num"]}
    k = np.logspace(0, 3, 400)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4), dpi=200)
    fig.subplots_adjust(left=0.05, right=0.93, top=0.72, bottom=0.17, wspace=0.42)

    for ax, (group, title, note, per, scale, ylabel, fmt) in zip(axes, CROSSOVER_PANELS):
        c = costs(data[group], per)
        ymax = 0
        ends = []
        for lib, (conv, op) in c.items():
            speed = scale * k / (conv + k * op)
            ymax = max(ymax, speed.max())
            ax.plot(k, speed, color=colors[lib], linewidth=2, solid_capstyle="round")
            ax.plot([1], [scale / (conv + op)], "o", color=colors[lib], markersize=7,
                    markeredgecolor=t["surface"], markeredgewidth=2, zorder=5)
            ends.append([speed[-1], f"{NAMES[lib]}  " + fmt.format(speed[-1])])
        gap = ymax * 1.12 * 0.07
        ends.sort()
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
        for y, text in ends:
            ax.text(1200 * 1.06, y, text, va="center", fontsize=9, color=t["ink"])

        conv, op = c["residue"]
        for lib in RIVALS:
            if lib not in c:
                continue
            kx = crossing(c["residue"], c[lib])
            if kx and kx <= 1000:
                y = scale * kx / (conv + kx * op)
                ax.plot([kx], [y], "o", markersize=9, markerfacecolor="none",
                        markeredgecolor=t["ink"], markeredgewidth=1.4, zorder=6)
                where = CROSSOVER_LABELS.get((group, lib))
                ax.annotate(
                    f"overtakes {NAMES[lib]}\nat k ≈ " + (f"{kx:.0f}" if kx >= 3 else f"{kx:.1f}"),
                    (kx, y),
                    xytext=where if where else (16, -38),
                    textcoords="data" if where else "offset points",
                    fontsize=8.5,
                    color=t["ink"],
                    arrowprops=dict(arrowstyle="-", color=t["muted"], lw=0.8),
                )
        for lib in RIVALS:
            if lib in c and c[lib][0] <= conv and c[lib][1] < op:
                ax.text(30, ymax * 0.62,
                        f"{NAMES[lib]} stays ahead for every k:\nits operation is "
                        f"{op / c[lib][1]:.1f} times faster\nand its conversions cost less.",
                        fontsize=8.5, color=t["ink"], ha="center")

        ax.set_xscale("log")
        ax.set_xlim(0.85, 1200)
        ax.set_ylim(0, ymax * 1.12)
        ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
        ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100", "200", "500", "1000"])
        ax.grid(color=t["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(t["axis"])
        ax.tick_params(axis="y", colors=t["muted"])
        ax.set_xlabel("k, operations per conversion (log scale)", fontsize=9.5)
        ax.set_ylabel(ylabel, fontsize=9.5)
        ax.set_title(title, loc="left", fontsize=11.5, weight="semibold", pad=26)
        ax.text(0, 1.02, note, transform=ax.transAxes, fontsize=8.5, color=t["ink2"], va="bottom")

    fig.text(0.05, 0.95, "When residue overtakes the other libraries, and when it does not",
             fontsize=16, weight="semibold")
    fig.text(
        0.05,
        0.905,
        "Lines follow time = conversions + k × operation, with both costs measured for each library. "
        "Dots at k = 1 are direct end-to-end measurements;\nfor large k each line tends to the "
        "measurement with data already in the library's format. Operations are assumed to keep "
        "the same size.",
        fontsize=9,
        color=t["ink2"],
        va="top",
    )
    present = [lib for lib in ("residue",) + RIVALS if lib in data["A_elementwise_mul"]]
    handles = [Line2D([0], [0], color=colors[lib], lw=2, label=NAMES[lib]) for lib in present]
    handles.append(Line2D([0], [0], marker="o", color=t["muted"], lw=0, markersize=7,
                          label="measured, k = 1"))
    handles.append(Line2D([0], [0], marker="o", markerfacecolor="none", markeredgecolor=t["ink"],
                          lw=0, markersize=9, label="crossing point"))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.045, 0.04), ncol=6,
               frameon=False, fontsize=9)
    fig.text(0.05, 0.015, MACHINE, fontsize=7.5, color=t["muted"])
    fig.savefig(path)
    plt.close(fig)


def grouped_chart(theme, path, title, subtitle, rows, series, xlabel, ticks, note, ref=None):
    """Horizontal bars, one group per row and one bar per series, log scale.

    ref, if given, is (x, label): a vertical reference line, such as GMP's
    time in a chart of ratios to GMP.
    """
    t = style(theme)
    n = len(series)
    legend_rows = (n + 2) // 3
    head = 1.25 + 0.28 * legend_rows
    body = 0.17 * n * len(rows) + 0.25 * len(rows)
    foot = 0.95
    fig_h = head + body + foot
    fig, ax = plt.subplots(figsize=(10, fig_h), dpi=200)
    fig.subplots_adjust(left=0.30, right=0.93, top=1 - head / fig_h, bottom=foot / fig_h)
    h = 0.8 / n
    values = [v for _, vals in rows for v in vals if v is not None]
    for i, (label, vals) in enumerate(rows):
        y = len(rows) - 1 - i
        for k, ((name, ckey, solid), v) in enumerate(zip(series, vals)):
            if v is None:
                continue
            yy = y + 0.4 - h * (k + 0.5)
            color = t[ckey]
            ax.barh(
                yy,
                v,
                height=h * 0.86,
                color=color if solid else tint(color),
                edgecolor=color,
                linewidth=0 if solid else 1.0,
            )
            text = f"{v:.2f}" if v < 10 else f"{v:.1f}" if v < 100 else f"{v:,.0f}"
            ax.text(v * 1.06, yy, text, va="center", ha="left", fontsize=8, color=t["ink2"])
    lo = min(values + ([ref[0]] if ref else [])) / 1.6
    ax.set_xscale("log")
    ax.set_xlim(lo, max(values) * 2.2)
    if ref:
        ax.axvline(ref[0], color=t["ink2"], linewidth=1)
        ax.text(ref[0] * 0.97, len(rows) - 0.58, ref[1], ha="right", va="bottom", fontsize=8.5, color=t["ink2"])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=9.5)
    ax.tick_params(axis="y", length=0)
    if ticks:
        ticks = [x for x in ticks if lo <= x <= max(values) * 2.2]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{x:g}" for x in ticks])
    ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.set_xlabel(xlabel, fontsize=9.5)
    ax.set_ylim(-0.6, len(rows) - 0.5)
    fig.text(0.03, 1 - 0.42 / fig_h, title, fontsize=15, weight="semibold")
    fig.text(0.03, 1 - 0.56 / fig_h, subtitle, fontsize=9.5, color=t["ink2"], va="top")
    handles = [
        Patch(facecolor=t[c] if solid else tint(t[c]), edgecolor=t[c], label=name) for name, c, solid in series
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.295, 1 - 1.12 / fig_h),
        ncol=min(n, 3),
        frameon=False,
        fontsize=8.5,
        handlelength=1.2,
    )
    fig.text(0.03, 0.3 / fig_h, note, fontsize=7.5, color=t["muted"])
    fig.savefig(path)
    plt.close(fig)


MODULAR = [
    ("M_modmul_{}", "one product", "residue"),
    ("M_modexp_{}", "one exponentiation", "residue"),
    ("M_batch_modexp_{}", "256 exponentiations", "residue_batch"),
]


def modular_chart(data, theme, path):
    """RNS Montgomery against GMP and num-bigint, time relative to GMP."""
    rows = []
    for bits in (1024, 2048):
        for key, label, res in MODULAR:
            g = data[key.format(bits)]
            rows.append((f"{label}, {bits} bits", [g[res] / g["gmp"], g["num_bigint"] / g["gmp"]]))
    grouped_chart(
        theme,
        path,
        "Arithmetic modulo a large odd number",
        "Running time relative to GMP (1 = as fast as GMP, higher is slower). The 256 exponentiations share one\n"
        "256-bit exponent and run as one ModVec batch; the single exponentiation has a full-size exponent.",
        rows,
        [("residue (RNS Montgomery)", "residue", True), ("num-bigint", "num", True)],
        "time relative to GMP (log scale)",
        [1, 1.5, 2, 3, 5, 7],
        MACHINE,
        ref=(1, "GMP"),
    )


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


WASM_LABELS = {
    "vec_mul_4096_9ch": "element-wise product, 4096 × 9 primes",
    "vec_dot_4096_9ch": "dot product, 4096 × 9 primes",
    "vec_from_i128s_4096_9ch": "from i128, 4096 × 9 primes",
    "vec_to_i128s_4096_9ch": "to i128, 4096 × 9 primes",
    "matmul_64_5ch": "matrix product 64×64, 5 primes",
    "determinant_32_9ch": "determinant 32×32, 9 primes",
    "modvec_product_1024bit_256": "ModVec product, 1024 bits × 256",
    "modelem_product_1024bit": "ModElem product, 1024 bits",
}


def wasm_chart(timing, theme, path):
    """WebAssembly in wasmtime, scalar and simd128, against native code."""
    rows = []
    for key, label in WASM_LABELS.items():
        native = timing["native"][key]
        rows.append((label, [timing["wasm"][key] / native, timing["simd128"][key] / native]))
    grouped_chart(
        theme,
        path,
        "WebAssembly against native code",
        "Running time in wasmtime relative to the native x86-64 build with AVX-512 (1 = as fast as native).\n"
        "examples/timing.rs, fastest of nine alternating runs.",
        rows,
        [("wasm32-wasip1", "gmp", False), ("wasm32-wasip1 + simd128", "gmp", True)],
        "time relative to native code (log scale)",
        [1, 2, 3, 5, 10],
        timing.get("machine", MACHINE.split(" GMP")[0] + " wasmtime with Cranelift."),
        ref=(1, "native"),
    )


PY_ROWS = [
    ("dot product", "dot product, 10 000 values of 100 bits"),
    ("matrix product", "matrix product 64×64, 60-bit entries"),
    ("determinant", "determinant 64×64, 10-bit entries"),
    ("256 modular exponentiations, 1024-bit", "256 × pow, 1024-bit modulus"),
    ("256 modular exponentiations, 2048-bit", "256 × pow, 2048-bit modulus"),
]

PY_SERIES = [
    (("Python ints", "Python pow"), "Python ints", "native", True),
    (("gmpy2", "gmpy2 powmod_base_list"), "gmpy2", "gmp", True),
    (("python-flint fmpz_mat", "python-flint fmpz"), "python-flint", "flint", True),
    (("python-flint + conv",), "python-flint, from Python ints", "flint", False),
    (("residue",), "residue_rs", "residue", True),
    (("residue + conv", "residue pow_batch"), "residue_rs, from Python ints", "residue", False),
]


def python_chart(res, theme, path):
    """The Python bindings against Python ints, gmpy2 and python-flint, in ms."""
    rows = []
    for prefix, label in PY_ROWS:
        title = next(k for k in res if k.startswith(prefix))
        vals = []
        for names, *_ in PY_SERIES:
            vals.append(next((res[title][n] for n in names if n in res[title]), None))
        rows.append((label, vals))
    grouped_chart(
        theme,
        path,
        "residue_rs from Python",
        "Milliseconds, log scale, lower is better. Solid bars: data already in each library's types; light bars:\n"
        "converting from Python ints and back included. python/bench/bench.py.",
        rows,
        [(name, c, solid) for _, name, c, solid in PY_SERIES],
        "milliseconds (log scale)",
        [0.01, 0.1, 1, 10, 100, 1000],
        res.get("machine", MACHINE),
    )


def timing_json(out, groups):
    """Fastest of several examples/timing.rs outputs per benchmark.

    groups maps a name to output files. Each output is already the median of
    many calls; across runs the fastest is kept, because on a shared virtual
    machine whole runs were sometimes twice as slow, native code included.
    """
    result = {}
    for name, files in groups.items():
        runs = {}
        for path in files:
            with open(path) as f:
                for line in f:
                    p = line.split()
                    if len(p) == 3 and p[2] == "µs":
                        runs.setdefault(p[0], []).append(float(p[1]))
        result[name] = {k: min(v) for k, v in runs.items()}
    with open(out, "w") as f:
        json.dump(result, f, indent=1)


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "timing-json":
        groups = dict(arg.split("=", 1) for arg in sys.argv[3:])
        timing_json(sys.argv[2], {k: v.split(",") for k, v in groups.items()})
        return
    root = sys.argv[1] if len(sys.argv) > 1 else "target/criterion"
    out = sys.argv[2] if len(sys.argv) > 2 else "docs"
    os.makedirs(out, exist_ok=True)
    data = load(root)
    missing = [k for k, *_ in SCENARIOS if k not in data]
    if missing:
        sys.exit(f"missing benchmark groups in {root}: {missing}")
    for theme in THEMES:
        suffix = "" if theme == "light" else "-dark"
        speedup_chart(data, theme, os.path.join(out, f"speedup{suffix}.png"))
        absolute_chart(data, theme, os.path.join(out, f"benchmarks{suffix}.png"))
        crossover_chart(data, theme, os.path.join(out, f"crossover{suffix}.png"))
        if all(k.format(b) in data for k, *_ in MODULAR for b in (1024, 2048)):
            modular_chart(data, theme, os.path.join(out, f"modular{suffix}.png"))
        timing = load_json(os.path.join(out, "data", "wasm.json"))
        if timing:
            wasm_chart(timing, theme, os.path.join(out, f"wasm{suffix}.png"))
        py = load_json(os.path.join(out, "data", "python.json"))
        if py:
            python_chart(py, theme, os.path.join(out, f"python{suffix}.png"))
    print(f"charts written to {out}/")


if __name__ == "__main__":
    main()
