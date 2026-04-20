"""
Plot paper-quality success rate curves from cluster log files.

Usage:
    python scripts/plot_results.py --logdir ~/JaxGCRL/logs --env ant_ball --out plots/
    python scripts/plot_results.py --logdir ~/JaxGCRL/logs --env reacher --out plots/

Reads all log files matching <method>_<env>_seed*.err and plots
mean ± standard error across seeds.
"""
import argparse
import re
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import defaultdict

PATTERN = re.compile(r"step:\s*(\d+),\s*eval/episode_success_any:\s*([\d.]+)")

STYLES = {
    "crl":          {"color": "#1f77b4", "ls": "-",  "lw": 2.5, "label": "CRL (vanilla)"},
    "crl_learntemp":{"color": "#e377c2", "ls": "--", "lw": 2.5, "label": "CRL + Learned Temp (ours)"},
}

def parse_log(path):
    """Parse all (step, value) pairs from a log file.
    Returns a list of (steps_array, values_array) — one per seed run.
    Seeds are separated when the step counter resets back toward 0.
    """
    runs = []
    steps, values = [], []
    last_step = -1
    with open(path) as f:
        for line in f:
            m = PATTERN.search(line)
            if m:
                s = int(m.group(1))
                v = float(m.group(2))
                # Detect seed boundary: step resets significantly
                if steps and s < last_step * 0.5:
                    if len(steps) > 1:
                        runs.append((np.array(steps, dtype=float), np.array(values, dtype=float)))
                    steps, values = [], []
                steps.append(s)
                values.append(v)
                last_step = s
    if len(steps) > 1:
        runs.append((np.array(steps, dtype=float), np.array(values, dtype=float)))
    return runs


def interpolate_to_grid(steps, values, grid):
    """Linearly interpolate values onto a common step grid."""
    if len(steps) < 2:
        return None
    return np.interp(grid, steps, values)


def load_method(logdir, method, env, max_steps=None):
    # Match actual filenames: crl_ant_ball_3s_27094274.err
    pattern = os.path.join(logdir, f"{method}_{env}_*.err")
    files = sorted(glob.glob(pattern))
    if not files:
        pattern = os.path.join(logdir, f"*{method}*{env}*.err")
        files = sorted(glob.glob(pattern))
    if not files:
        print(f"[WARN] No files found for {method} on {env} in {logdir}")
        return None, None, None

    all_steps, all_values = [], []
    for f in files:
        runs = parse_log(f)
        print(f"  {os.path.basename(f)}: {len(runs)} seed run(s) detected")
        for i, (s, v) in enumerate(runs):
            if len(s) > 1:
                all_steps.append(s)
                all_values.append(v)
                print(f"    seed {i}: {len(s)} evals, final={v[-1]:.3f}")

    if not all_steps:
        return None, None, None

    # Build common grid to longest run (shorter runs extrapolate flat via np.interp)
    max_step = max(s[-1] for s in all_steps)
    if max_steps:
        max_step = min(max_step, max_steps)
    grid = np.linspace(0, max_step, 300)

    interpolated = []
    for s, v in zip(all_steps, all_values):
        interp = interpolate_to_grid(s, v, grid)
        if interp is not None:
            interpolated.append(interp)

    if not interpolated:
        return None, None, None

    interpolated = np.array(interpolated)
    mean = interpolated.mean(axis=0)
    se = interpolated.std(axis=0) / np.sqrt(len(interpolated))
    return grid, mean, se


def plot_env(logdir, env, out_dir, max_steps=None, legend_loc="upper left", suffix=""):
    # Pre-load all methods so multi-plot calls reuse the same data
    data = {}
    for method, style in STYLES.items():
        print(f"\nLoading {method} on {env}...")
        grid, mean, se = load_method(logdir, method, env, max_steps)
        if grid is not None:
            data[method] = (grid, mean, se)

    if not data:
        print(f"[WARN] No data found for env={env}, skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    for method, style in STYLES.items():
        if method not in data:
            continue
        grid, mean, se = data[method]
        ax.plot(grid, mean,
                color=style["color"],
                linestyle=style["ls"],
                linewidth=style["lw"],
                label=style["label"])
        ax.fill_between(grid, mean - se, mean + se,
                        color=style["color"], alpha=0.15)

    ax.set_xlabel("Training Steps", fontsize=13)
    ax.set_ylabel("Success Rate (eval/episode_success_any)", fontsize=12)
    ax.set_title(f"{env.replace('_', ' ').title()}", fontsize=14)
    ax.legend(loc=legend_loc, fontsize=11, framealpha=0.9)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M" if x >= 1e6 else f"{int(x/1e3)}k")
    )

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{env}_success_rate{suffix}.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    out_png = out_path.replace(".pdf", ".png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved → {out_path}")
    print(f"Saved → {out_png}")


def print_table_stats(logdir, env, n_last=20):
    """Print mean ± std of last n_last evaluations per method, ready for LaTeX table."""
    print(f"\n{'='*55}")
    print(f"Table stats for {env}  (last {n_last} evals per seed)")
    print(f"{'='*55}")
    for method, style in STYLES.items():
        pattern = os.path.join(logdir, f"{method}_{env}_*.err")
        files = sorted(glob.glob(pattern))
        if not files:
            pattern = os.path.join(logdir, f"*{method}*{env}*.err")
            files = sorted(glob.glob(pattern))
        if not files:
            print(f"  {style['label']:40s}  --")
            continue

        tail_means = []
        for f in files:
            runs = parse_log(f)
            for _, v in runs:
                if len(v) >= 1:
                    tail = v[-min(n_last, len(v)):]
                    tail_means.append(tail.mean())

        if not tail_means:
            print(f"  {style['label']:40s}  --")
            continue

        arr = np.array(tail_means) * 100  # convert to %
        mean, std = arr.mean(), arr.std()
        print(f"  {style['label']:40s}  {mean:.1f} +/- {std:.1f}%")
        print(f"    LaTeX: ${mean:.1f} \\pm {std:.1f}$")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True, help="Path to cluster logs directory")
    parser.add_argument("--env", required=True, help="Environment name e.g. ant_ball")
    parser.add_argument("--out", default="plots", help="Output directory")
    parser.add_argument("--max_steps", type=float, default=None, help="Truncate at this many steps")
    parser.add_argument("--legend_loc", default="upper left", help="Legend position e.g. 'lower left'")
    parser.add_argument("--zoom_steps", type=float, default=None,
                        help="Also save a second zoomed plot up to this many steps")
    parser.add_argument("--stats", action="store_true",
                        help="Print last-20-eval mean±std for each method (for table)")
    parser.add_argument("--n_last", type=int, default=20,
                        help="Number of final evals to average for --stats")
    args = parser.parse_args()

    if args.stats:
        print_table_stats(args.logdir, args.env, args.n_last)
    else:
        plot_env(args.logdir, args.env, args.out, args.max_steps, args.legend_loc)
        if args.zoom_steps:
            plot_env(args.logdir, args.env, args.out, args.zoom_steps, args.legend_loc,
                     suffix=f"_zoom{int(args.zoom_steps/1e6)}M")
