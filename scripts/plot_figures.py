#!/usr/bin/env python
"""
Figure generation utilities for Paper 1 experiments.

Currently implemented:
  - E1 comparison bar chart (TTM, Raw Return, Independence Gain)
  - Learning curve plotting helper (unused by default)
  - OPE comparison helper (legacy support)

Manuscript figures are emitted under output/paper1/figures/ using the
same filenames as paper/figures/, and are also synced to paper/figures/.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

try:
    from scipy import stats
except Exception:  # pragma: no cover - SciPy optional
    stats = None  # type: ignore


STYLE = {
    "figure.figsize": (12, 4),  # default for multi-panel layouts
    "dpi": 150,
    "fontsize": 10,
    "labelsize": 11,
    "titlesize": 12,
    "linewidth": 1.6,
    "capsize": 4,
    "alpha_grid": 0.25,
}

SINGLE_FIGSIZE = (6.0, 4.0)
MULTI_FIGSIZE = (12, 4)

ORIGIN_COLORS = {
    "ttm": "#1f4e79",
    "reward": "#e37b40",
    "indep": "#4f9da6",
    "horizon_ttm": "#1f4e79",
    "horizon_feas": "#e37b40",
    "safety_window": "#1f4e79",
    "safety_peak": "#e37b40",
}

AGENT_COLORS = {
    "pianompc": ORIGIN_COLORS["ttm"],       # match E1 primary color
    "ccb_df": ORIGIN_COLORS["reward"],      # match E1 secondary color
    "bayesianmab": ORIGIN_COLORS["indep"],  # match E1 tertiary color
    "thompson": "#2ca02c",                  # tab10[2]
    "linucb": "#9467bd",                    # tab10[4]
    "dqn": "#8c564b",                       # tab10[5]
    "safe_ac": "#17becf",                   # tab10[9]
    "autocurriculum": "#e377c2",            # tab10[6]
}

AGENT_LABELS = {
    "pianompc": "PianoMPC",
    "ccb_df": "CCB-DF",
    "bayesianmab": "BayesianMAB",
    "thompson": "Thompson",
    "linucb": "LinUCB",
    "dqn": "DQN",
    "safe_ac": "Safe-AC",
    "autocurriculum": "AutoCurriculum",
}


def _agent_key(agent_name: str) -> str:
    return agent_name.strip().lower().replace("-", "_")


def _agent_color(agent_name: str) -> str:
    return AGENT_COLORS.get(_agent_key(agent_name), "#7f7f7f")


def _agent_label(agent_name: str) -> str:
    return AGENT_LABELS.get(_agent_key(agent_name), agent_name)

LINE_WIDTH = 1.5
MARKER_SIZE = 4.0

FIGURE_ROOT = Path("output/paper1/figures")
FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
MANUSCRIPT_FIGURE_ROOT = Path("paper/figures")
MANUSCRIPT_FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

MANUSCRIPT_FIGURE_NAMES = {
    "e1_comparison.png",
    "e2_learning_curves.png",
    "e3_pianoMPC_horizon.png",
    "e4_safety_tradeoff.png",
    "e5_safety_consistency.png",
    "e6_dynamics_mismatch.png",
    "e7_threshold_window.png",
}


def _save_figure(fig: plt.Figure, save_path: Path, *, sync_manuscript: bool = True) -> None:
    """Save a figure and, for manuscript figures, mirror PNG/PDF/SVG into paper/figures."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=STYLE["dpi"], bbox_inches="tight")
    if sync_manuscript and save_path.parent == FIGURE_ROOT and save_path.name in MANUSCRIPT_FIGURE_NAMES:
        pdf_path = save_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"  ✓ Saved PDF: {pdf_path}")
        svg_path = save_path.with_suffix(".svg")
        fig.savefig(svg_path, bbox_inches="tight")
        print(f"  ✓ Saved SVG: {svg_path}")

        manuscript_path = MANUSCRIPT_FIGURE_ROOT / save_path.name
        manuscript_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(manuscript_path, dpi=STYLE["dpi"], bbox_inches="tight")
        print(f"  ✓ Synced: {manuscript_path}")
        manuscript_pdf_path = manuscript_path.with_suffix(".pdf")
        fig.savefig(manuscript_pdf_path, bbox_inches="tight")
        print(f"  ✓ Synced PDF: {manuscript_pdf_path}")
        manuscript_svg_path = manuscript_path.with_suffix(".svg")
        fig.savefig(manuscript_svg_path, bbox_inches="tight")
        print(f"  ✓ Synced SVG: {manuscript_svg_path}")


def _load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _t_critical(df: int) -> float:
    if df <= 0:
        return 0.0
    if stats is None:
        return 1.96
    try:
        return float(stats.t.ppf(0.975, df))
    except Exception:
        return 1.96


def _apply_origin_axes_style(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.set_facecolor("white")
    if grid_axis == "both":
        ax.grid(alpha=STYLE["alpha_grid"], linewidth=0.7)
    elif grid_axis == "y":
        ax.grid(axis="y", alpha=STYLE["alpha_grid"], linewidth=0.7)
    elif grid_axis == "x":
        ax.grid(axis="x", alpha=STYLE["alpha_grid"], linewidth=0.7)
    else:
        ax.grid(False)

    ax.tick_params(colors="#2b2b2b", direction="out", length=5, width=0.8, labelsize=STYLE["fontsize"])
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="out", length=3, width=0.6)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#2b2b2b")
    ax.set_axisbelow(True)


def plot_comparison(results: Dict[str, Mapping[str, float]] | None = None,
                    save_path: Path | None = None) -> Path | None:
    """Plot TTM / Raw Reward / Independence Gain comparison for E1."""
    if results is None:
        data_path = Path("output/data/comparison_data.json")
        if not data_path.exists():
            print("⚠️  Skipping plot_comparison: comparison_data.json not found.")
            return None
        results = _load_json(data_path)

    if not results:
        print("⚠️  Skipping plot_comparison: empty results.")
        return None

    if save_path is None:
        save_path = FIGURE_ROOT / "e1_comparison.png"
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("default")
    rc_updates = {k: v for k, v in STYLE.items() if k.startswith("figure.")}
    rc_updates.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    })
    plt.rcParams.update(rc_updates)

    fig, axes = plt.subplots(1, 3, figsize=MULTI_FIGSIZE, facecolor="white")

    # Sort agents for consistent plotting (TTM ascending, with PianoMPC forced to the front)
    agents = list(results.keys())
    agents.sort(key=lambda name: results[name].get("time_to_mastery_mean", float("inf")))
    if "PianoMPC" in agents:
        agents.remove("PianoMPC")
        agents.insert(0, "PianoMPC")
    ttm_means = [results[a].get("time_to_mastery_mean", 0.0) for a in agents]
    ttm_stds = [results[a].get("time_to_mastery_std", 0.0) for a in agents]

    reward_means = [results[a].get("total_reward_raw_mean", results[a].get("total_reward_mean", 0.0)) for a in agents]
    reward_stds = [results[a].get("total_reward_raw_std", results[a].get("total_reward_std", 0.0)) for a in agents]

    indep_means = [results[a].get("independence_gain_mean", 0.0) for a in agents]
    indep_stds = [results[a].get("independence_gain_std", 0.0) for a in agents]

    bars_ttm = axes[0].bar(
        agents,
        ttm_means,
        yerr=ttm_stds,
        capsize=STYLE["capsize"],
        alpha=0.85,
        color=ORIGIN_COLORS["ttm"],
        edgecolor="#1b3c5d",
        linewidth=0.8,
        zorder=3,
    )
    axes[0].set_ylabel("Time-to-Mastery (steps)", fontsize=STYLE["labelsize"])
    axes[0].set_title("(a) Time-to-Mastery", fontsize=STYLE["titlesize"])

    bars_reward = axes[1].bar(
        agents,
        reward_means,
        yerr=reward_stds,
        capsize=STYLE["capsize"],
        alpha=0.85,
        color=ORIGIN_COLORS["reward"],
        edgecolor="#b3541d",
        linewidth=0.8,
        zorder=3,
    )
    axes[1].set_ylabel("Total Raw Reward", fontsize=STYLE["labelsize"])
    axes[1].set_title("(b) Total Reward", fontsize=STYLE["titlesize"])

    bars_indep = axes[2].bar(
        agents,
        indep_means,
        yerr=indep_stds,
        capsize=STYLE["capsize"],
        alpha=0.85,
        color=ORIGIN_COLORS["indep"],
        edgecolor="#2f716f",
        linewidth=0.8,
        zorder=3,
    )
    axes[2].set_ylabel("Independence Gain", fontsize=STYLE["labelsize"])
    axes[2].set_title("(c) Coordination Improvement", fontsize=STYLE["titlesize"])

    for ax in axes:
        _apply_origin_axes_style(ax, grid_axis="y")
        ax.tick_params(axis="x", rotation=30, labelsize=max(STYLE["fontsize"] - 1, 6))

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    _save_figure(fig, save_path)
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")
    return save_path


def plot_learning_curves(histories: Dict[str, Sequence[Mapping]] | None = None,
                         save_path: Path | None = None) -> Path | None:
    """Plot smoothed learning curves from trajectory histories."""
    if histories is None:
        data_path = Path("output/data/learning_curves_data.json")
        if not data_path.exists():
            print("⚠️  Skipping plot_learning_curves: learning_curves_data.json not found.")
            return None
        histories = _load_json(data_path)

    if save_path is None:
        save_path = FIGURE_ROOT / "e2_learning_curves.png"
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("default")
    rc_updates = {k: v for k, v in STYLE.items() if k.startswith("figure.")}
    rc_updates.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    })
    plt.rcParams.update(rc_updates)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    labelsize = STYLE["labelsize"] + 1
    titlesize = STYLE["titlesize"] + 1
    ticksize = STYLE["fontsize"] + 1
    legend_size = STYLE["fontsize"] + 1
    def _initial_reward(agent: str) -> float:
        trajs = histories.get(agent, [])
        if not trajs:
            return float("inf")
        rewards = trajs[0].get("raw_rewards", trajs[0].get("rewards", []))
        if not rewards:
            return float("inf")
        return float(rewards[0])

    sorted_agents = sorted(
        histories.keys(),
        key=lambda name: (_initial_reward(name), name)
    )
    if "PianoMPC" in sorted_agents:
        sorted_agents.remove("PianoMPC")
        sorted_agents.insert(0, "PianoMPC")

    cycle_colors = iter(["#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#17becf", "#ff7f0e", "#bcbd22", "#7f7f7f"])

    for agent_name in sorted_agents:
        trajs = histories.get(agent_name, [])
        series: List[np.ndarray] = []
        for traj in trajs:
            rewards = np.asarray(traj.get("raw_rewards", traj.get("rewards", [])), dtype=float)
            if rewards.size == 0:
                continue
            mastery_step = int(traj.get("mastery_step", 0))
            if mastery_step > 0:
                rewards = rewards[: min(mastery_step, rewards.size)]
            window = min(5, rewards.size)
            if window <= 1:
                smoothed = rewards.astype(float)
            elif rewards.size >= window:
                kernel = np.ones(window, dtype=float) / window
                smoothed = np.convolve(rewards, kernel, mode="valid")
            else:
                smoothed = rewards.astype(float)
            if smoothed.size:
                series.append(smoothed)
        if not series:
            continue
        min_len = min(len(s) for s in series if len(s) > 0)
        if min_len == 0:
            continue
        aligned = np.stack([s[:min_len] for s in series], axis=0)
        mean_curve = np.nanmean(aligned, axis=0)
        std_curve = np.nanstd(aligned, axis=0)
        x = np.arange(min_len)
        line_color = "#c00000" if agent_name == "PianoMPC" else next(cycle_colors)
        ax.plot(
            x,
            mean_curve,
            label=agent_name,
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=MARKER_SIZE,
            color=line_color,
        )
        ax.fill_between(
            x,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=line_color,
            alpha=0.2,
        )

    ax.set_xlabel("Steps", fontsize=labelsize)
    ax.set_ylabel("Sliding Avg Raw Reward", fontsize=labelsize)
    ax.legend(fontsize=legend_size, frameon=False)
    _apply_origin_axes_style(ax, grid_axis="both")
    ax.tick_params(labelsize=ticksize + 1)

    fig.tight_layout()
    _save_figure(fig, save_path)
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")
    return save_path


def plot_pianoMPC_horizon(data: Dict | None = None,
                     save_path: Path | None = None) -> Path | None:
    """Plot PianoMPC horizon sweep (TTM vs feasible rate)."""
    if data is None:
        data_path = Path("output/data/pianoMPC_horizon.json")
        if not data_path.exists():
            print("⚠️  Skipping plot_pianoMPC_horizon: pianoMPC_horizon.json not found.")
            return None
        data = _load_json(data_path)

    meta = data.get("meta", {})
    num_runs = int(meta.get("num_runs", 0))
    horizons = sorted(int(h) for h in meta.get("horizons", []))
    profile_metrics = data.get("metrics", {}).get("balanced")
    if not profile_metrics:
        print("⚠️  Skipping plot_pianoMPC_horizon: balanced profile metrics missing.")
        return None

    if save_path is None:
        save_path = FIGURE_ROOT / "e3_pianoMPC_horizon.png"
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("default")
    rc_updates = {k: v for k, v in STYLE.items() if k.startswith("figure.")}
    rc_updates.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    })
    plt.rcParams.update(rc_updates)

    plot_horizons: List[int] = []
    ttm_means = []
    ttm_ci = []
    feas_means = []
    feas_ci = []

    t_value = _t_critical(num_runs - 1)
    for horizon in horizons:
        metrics = profile_metrics.get(str(horizon))
        if not metrics:
            continue
        plot_horizons.append(horizon)
        ttm_means.append(metrics.get("time_to_mastery_mean", np.nan))
        feas_mean = metrics.get("feasible_rate_mean", np.nan) * 100.0
        feas_means.append(feas_mean)

        std_ttm = metrics.get("time_to_mastery_std", 0.0)
        std_feas = metrics.get("feasible_rate_std", 0.0) * 100.0
        denom = np.sqrt(num_runs) if num_runs > 0 else 1.0
        ttm_ci.append(float(std_ttm * t_value / denom))
        feas_ci.append(float(std_feas * t_value / denom))

    x = np.array(plot_horizons, dtype=float)
    if x.size == 0:
        print("⚠️  Skipping plot_pianoMPC_horizon: no horizon metrics available.")
        return None

    feas_means_arr = np.clip(np.asarray(feas_means, dtype=float), 0.0, 100.0)
    feas_ci_arr = np.asarray(feas_ci, dtype=float)
    feas_lower = np.maximum(0.0, feas_means_arr - feas_ci_arr)
    feas_upper = np.minimum(100.0, feas_means_arr + feas_ci_arr)
    feas_yerr = np.vstack([feas_means_arr - feas_lower, feas_upper - feas_means_arr])

    fig, ax1 = plt.subplots(figsize=SINGLE_FIGSIZE, facecolor="white")
    labelsize = STYLE["labelsize"] + 1
    titlesize = STYLE["titlesize"] + 1
    ticksize = STYLE["fontsize"] + 1
    legend_size = STYLE["fontsize"] + 1
    line1 = ax1.errorbar(
        x,
        ttm_means,
        yerr=ttm_ci,
        fmt="o-",
        color=ORIGIN_COLORS["horizon_ttm"],
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label="TTM (steps)",
        capsize=STYLE["capsize"],
    )
    ax1.set_xlabel("Planning horizon $H$", fontsize=labelsize)
    ax1.set_ylabel("TTM (steps)", fontsize=labelsize)
    ax1.set_ylim(16.0, 34.5)
    ax1.set_yticks(np.arange(16, 35, 2))
    ax1.set_xticks(x)
    _apply_origin_axes_style(ax1, grid_axis="both")
    ax1.tick_params(labelsize=ticksize)

    ax2 = ax1.twinx()
    line2 = ax2.errorbar(
        x,
        feas_means_arr,
        yerr=feas_yerr,
        fmt="s-",
        color=ORIGIN_COLORS["horizon_feas"],
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label="Feasible rate (%)",
        capsize=STYLE["capsize"],
    )
    ax2.set_ylabel("Feasible rate (%)", fontsize=labelsize)
    ax2.set_ylim(70.0, 102.0)
    ax2.set_yticks([70, 80, 90, 100])
    ax2.axhline(100.0, color="#b0b0b0", alpha=STYLE["alpha_grid"], linewidth=0.7, zorder=0)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.tick_params(colors="#2b2b2b", direction="out", length=5, width=0.8, labelsize=ticksize)

    ax2.grid(False)
    handles = [line1.lines[0], line2.lines[0]]
    labels = ["TTM (steps)", "Feasible rate (%)"]
    ax1.legend(handles, labels, loc="best", frameon=False, fontsize=legend_size)

    fig.tight_layout()
    _save_figure(fig, save_path)
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")
    return save_path


def plot_safety_tradeoff(data: Dict | None = None,
                         save_path: Path | None = None) -> Path | None:
    if data is None:
        data_path = Path("output/data/safety_ablation.json")
        if not data_path.exists():
            print("⚠️  Skipping plot_safety_tradeoff: safety_ablation.json not found.")
            return None
        data = _load_json(data_path)

    macro = data.get("macro", {})
    configs = {cfg["name"]: cfg.get("label", cfg["name"]) for cfg in data.get("meta", {}).get("configurations", [])}
    if not macro:
        print("⚠️  Skipping plot_safety_tradeoff: macro metrics missing.")
        return None

    if save_path is None:
        save_path = FIGURE_ROOT / "e4_safety_tradeoff.png"
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("default")
    rc_updates = {k: v for k, v in STYLE.items() if k.startswith("figure.")}
    rc_updates.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    })
    plt.rcParams.update(rc_updates)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), facecolor="white", sharey=True)
    labelsize = STYLE["labelsize"] + 1
    ticksize = STYLE["fontsize"] + 1
    titlesize = STYLE["titlesize"] + 1
    value_label_size = STYLE["fontsize"] + 1

    family_colors = {
        "LinUCB": "#1f77b4",
        "PianoMPC": "#ff7f0e",
        "Safe-AC": "#2ca02c",
    }
    selected_configs = [
        "PianoMPC-default",
        "PianoMPC-no-soft",
        "Safe-AC",
        "LinUCB-both",
        "LinUCB-guard",
        "LinUCB-soft",
        "LinUCB-none",
    ]

    rows = []
    for name in selected_configs:
        metrics = macro.get(name)
        if not metrics:
            continue
        label = configs.get(name, name)
        family = "LinUCB" if name.startswith("LinUCB") else (
            "PianoMPC" if name.startswith("PianoMPC") else "Safe-AC"
        )
        rows.append(
            {
                "name": name,
                "label": label,
                "family": family,
                "ttm": metrics.get("time_to_mastery_mean", np.nan),
                "violation": (1.0 - metrics.get("feasible_rate_mean", 0.0)) * 100.0,
            }
        )

    if not rows:
        print("⚠️  Skipping plot_safety_tradeoff: no selected configurations available.")
        return None

    y_pos = np.arange(len(rows))
    labels = [row["label"] for row in rows]
    colors = [family_colors.get(row["family"], "#7f7f7f") for row in rows]
    ttm_vals = np.asarray([row["ttm"] for row in rows], dtype=float)
    violation_vals = np.asarray([row["violation"] for row in rows], dtype=float)
    profile_metrics = data.get("metrics", {})

    panel_specs = [
        (axes[0], ttm_vals, "ttm", "(a) Time-to-mastery", "TTM (steps)", "{:.1f}"),
        (axes[1], violation_vals, "violation", "(b) Realized overload rate", "Overload rate (%)", "{:.1f}"),
    ]
    profile_offsets = np.linspace(-0.20, 0.20, max(len(profile_metrics), 1))
    profile_markers = ["o", "s", "^", "D", "v"]
    for ax, values, metric_key, title, xlabel, value_fmt in panel_specs:
        ax.barh(
            y_pos,
            values,
            color=colors,
            edgecolor="#2b2b2b",
            linewidth=0.6,
            alpha=0.88,
            zorder=3,
        )
        for profile_idx, (profile_name, profile_block) in enumerate(profile_metrics.items()):
            point_values = []
            point_y = []
            for row_idx, row in enumerate(rows):
                profile_row = profile_block.get(row["name"], {})
                if metric_key == "ttm":
                    value = profile_row.get("time_to_mastery_mean", float("nan"))
                else:
                    value = (1.0 - profile_row.get("feasible_rate_mean", float("nan"))) * 100.0
                if math.isfinite(float(value)):
                    point_values.append(float(value))
                    point_y.append(row_idx + profile_offsets[profile_idx])
            if point_values:
                ax.scatter(
                    point_values,
                    point_y,
                    marker=profile_markers[profile_idx % len(profile_markers)],
                    s=28,
                    facecolor="white",
                    edgecolor="#2b2b2b",
                    linewidth=0.8,
                    zorder=5,
                    label=profile_name if ax is axes[0] else None,
                )
        finite_vals = values[np.isfinite(values)]
        max_val = float(np.nanmax(finite_vals)) if finite_vals.size else 1.0
        point_max_vals: List[float] = []
        for profile_block in profile_metrics.values():
            for row in rows:
                profile_row = profile_block.get(row["name"], {})
                if metric_key == "ttm":
                    point_value = profile_row.get("time_to_mastery_mean", float("nan"))
                else:
                    point_value = (1.0 - profile_row.get("feasible_rate_mean", float("nan"))) * 100.0
                if math.isfinite(float(point_value)):
                    point_max_vals.append(float(point_value))
        if point_max_vals:
            max_val = max(max_val, max(point_max_vals))
        text_pad = max(max_val * 0.02, 0.4)
        label_x = []
        for idx, value in enumerate(values):
            if not math.isfinite(float(value)):
                continue
            row_point_vals: List[float] = []
            for profile_block in profile_metrics.values():
                profile_row = profile_block.get(rows[idx]["name"], {})
                if metric_key == "ttm":
                    point_value = profile_row.get("time_to_mastery_mean", float("nan"))
                else:
                    point_value = (1.0 - profile_row.get("feasible_rate_mean", float("nan"))) * 100.0
                if math.isfinite(float(point_value)):
                    row_point_vals.append(float(point_value))
            text_x = max([float(value), *row_point_vals]) + text_pad
            label_x.append(text_x)
            ax.text(
                text_x,
                idx,
                value_fmt.format(float(value)),
                va="center",
                ha="left",
                fontsize=value_label_size,
                color="#2b2b2b",
            )
        if label_x:
            max_val = max(max_val, max(label_x))
        ax.set_xlim(0.0, max_val * 1.12 + text_pad)
        ax.set_xlabel(xlabel, fontsize=labelsize)
        ax.set_title(title, fontsize=titlesize)
        _apply_origin_axes_style(ax, grid_axis="x")
        ax.tick_params(labelsize=ticksize)

    axes[0].set_yticks(y_pos, labels)
    axes[0].invert_yaxis()
    axes[1].tick_params(axis="y", labelleft=False)

    families_present = {row["family"] for row in rows}
    legend_handles = []
    legend_labels = []
    for family in ("PianoMPC", "Safe-AC", "LinUCB"):
        if family in families_present:
            legend_handles.append(
                plt.Rectangle((0, 0), 1, 1, color=family_colors.get(family, "#7f7f7f"), alpha=0.88)
            )
            legend_labels.append(family)
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper left",
            ncol=len(legend_labels),
            frameon=False,
            bbox_to_anchor=(0.18, 0.99),
            borderaxespad=0.2,
        )
    profile_handles, profile_labels = axes[0].get_legend_handles_labels()
    if profile_handles:
        fig.legend(
            profile_handles,
            profile_labels,
            loc="lower center",
            ncol=len(profile_labels),
            frameon=False,
            fontsize=max(STYLE["fontsize"] - 1, 7),
            title="Profile points",
            title_fontsize=max(STYLE["fontsize"] - 1, 7),
            bbox_to_anchor=(0.5, 0.005),
            borderaxespad=0.2,
        )

    fig.tight_layout(rect=[0.02, 0.08, 0.98, 0.91])
    _save_figure(fig, save_path)
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")
    return save_path


def plot_safety_consistency(csv_path: Path | None = None,
                            profile: str = "balanced",
                            diagnostic_window: int = 5,
                            save_path: Path | None = None) -> Path | None:
    if csv_path is None:
        csv_path = Path("output/data/p1_safety/safety_consistency_summary.csv")
    if not csv_path.exists():
        print(f"⚠️  Skipping plot_safety_consistency: {csv_path} not found.")
        return None

    records: Dict[int, Dict[float, Dict[str, float]]] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("profile") != profile:
                continue
            try:
                W = int(row["W"])
                Hg = int(row["Hg"])
                delta = float(row["delta_guard"])
                peak_rate = float(row["peak_violation_rate"]) * 100.0
                guard_false_negative = float(row["guard_false_negative_rate"]) * 100.0
            except (KeyError, TypeError, ValueError):
                continue
            if W != diagnostic_window:
                continue
            records.setdefault(Hg, {})[delta] = {
                "peak": peak_rate,
                "guard_false_negative": guard_false_negative,
            }

    if not records:
        print("⚠️  Skipping plot_safety_consistency: no matching rows.")
        return None

    if save_path is None:
        save_path = FIGURE_ROOT / "e5_safety_consistency.png"
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("default")
    rc_updates = {k: v for k, v in STYLE.items() if k.startswith("figure.")}
    rc_updates.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    })
    plt.rcParams.update(rc_updates)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor="white")
    labelsize = STYLE["labelsize"] + 1
    titlesize = STYLE["titlesize"] + 1
    ticksize = STYLE["fontsize"] + 1
    legend_size = STYLE["fontsize"] + 1
    horizon_colors = {
        3: ORIGIN_COLORS["ttm"],
        5: ORIGIN_COLORS["reward"],
        7: ORIGIN_COLORS["indep"],
    }
    horizon_markers = {3: "o", 5: "s", 7: "^"}
    delta_values = sorted({delta for delta_map in records.values() for delta in delta_map})

    panel_specs = [
        ("peak", "(a) Online peak-violation rate", "Peak violation rate (%)"),
        ("guard_false_negative", "(b) Guard false-negative rate", "False-negative rate (%)"),
    ]
    for ax, (metric, title, ylabel) in zip(axes, panel_specs):
        plotted_vals: List[float] = []
        for Hg in sorted(records):
            y_vals = [
                records.get(Hg, {}).get(delta, {}).get(metric, float("nan"))
                for delta in delta_values
            ]
            if all(math.isnan(val) for val in y_vals):
                continue
            plotted_vals.extend(val for val in y_vals if not math.isnan(val))
            ax.plot(
                delta_values,
                y_vals,
                marker=horizon_markers.get(Hg, "o"),
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE,
                color=horizon_colors.get(Hg, "#7f7f7f"),
                label=f"$H_g={Hg}$",
            )
        ax.set_xlabel("Guard slack $\\delta_{guard}$", fontsize=labelsize)
        ax.set_ylabel(ylabel, fontsize=labelsize)
        ax.set_title(title, fontsize=titlesize)
        ax.set_xticks(delta_values)
        if plotted_vals:
            upper = min(100.0, max(5.0, max(plotted_vals) * 1.25 + 1.0))
            ax.set_ylim(0.0, upper)
        _apply_origin_axes_style(ax, grid_axis="both")
        ax.tick_params(labelsize=ticksize)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="best", frameon=False, fontsize=legend_size)

    fig.tight_layout()
    _save_figure(fig, save_path)
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")
    return save_path


def plot_dynamics_mismatch(summary_path: Path | None = None,
                           metrics: Sequence[str] | None = None,
                           save_path: Path | None = None,
                           row_height: float = 3.3,
                           col_width: float = 3.3) -> Path | None:
    if summary_path is None:
        summary_path = Path("output/data/p1_misspec/dynamics_mismatch_summary.csv")
    summary_path = Path(summary_path)
    if not summary_path.exists():
        print(f"⚠️  Skipping plot_dynamics_mismatch: {summary_path} not found.")
        return None

    metric_specs = {
        "ttm": {"label": "ΔTTM (steps)", "scale": 1.0},
        "feasible_rate": {"label": "ΔFeas. (pp)", "scale": 100.0},
        "avg_fatigue": {"label": "ΔAvg Fatigue", "scale": 1.0},
        "learn_slope": {"label": "ΔLearn Slope", "scale": 1.0},
    }
    if metrics is None:
        metrics = ("ttm", "feasible_rate", "avg_fatigue", "learn_slope")
    metrics = [m for m in metrics if m in metric_specs]
    if not metrics:
        print("⚠️  Skipping plot_dynamics_mismatch: no valid metrics specified.")
        return None

    data: Dict[str, Dict[str, Dict[str, Dict[float, float]]]] = {}
    agents_present: set[str] = set()
    with summary_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            param_name = row.get("param_name")
            metric = row.get("metric")
            agent = row.get("agent")
            if not param_name or not metric or not agent:
                continue
            try:
                scale = float(row["scale"])
                value_mean = float(row["value_mean"])
            except (KeyError, TypeError, ValueError):
                continue
            data.setdefault(param_name, {}).setdefault(metric, {}).setdefault(agent, {})[scale] = value_mean
            agents_present.add(agent)

    if not data:
        print("⚠️  Skipping plot_dynamics_mismatch: empty summary.")
        return None

    agent_order = sorted(
        agents_present,
        key=lambda a: (0 if _agent_key(a) == "pianoMPC" else 1, AGENT_LABELS.get(_agent_key(a), a)),
    )

    param_order = ["eta_forget", "gamma_f", "kappa"]
    param_order = [p for p in param_order if p in data] or list(data.keys())

    n_rows = len(param_order)
    n_cols = len(metrics)
    figsize = (n_cols * col_width, n_rows * row_height)

    if save_path is None:
        save_path = FIGURE_ROOT / "e6_dynamics_mismatch.png"
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("default")
    rc_updates = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    }
    plt.rcParams.update(rc_updates)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        squeeze=False,
        facecolor="white",
        constrained_layout=False,
    )
    plt.subplots_adjust(hspace=0.4, wspace=0.28)
    legend_handles: Dict[str, plt.Rectangle] = {}

    from string import ascii_lowercase
    panel_labels = iter(ascii_lowercase)

    for row_idx, param in enumerate(param_order):
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx][col_idx]
            param_block = data.get(param, {})
            metric_block = param_block.get(metric, {})
            if not metric_block:
                ax.set_visible(False)
                continue

            scale_values = sorted({scale for agent_scalars in metric_block.values() for scale in agent_scalars})
            if not scale_values:
                ax.set_visible(False)
                continue

            scale_factor_labels = [f"{1.0 + s:.2f}" for s in scale_values]
            y_positions = np.arange(len(scale_values))
            bar_height = 0.8 / max(len(agent_order), 1)

            baseline_scale = min(scale_values, key=lambda s: abs(s))

            for agent_idx, agent in enumerate(agent_order):
                agent_series = metric_block.get(agent)
                if not agent_series:
                    continue
                baseline_value = agent_series.get(baseline_scale)
                if baseline_value is None:
                    continue
                deltas: List[float] = []
                for scale in scale_values:
                    value = agent_series.get(scale)
                    if value is None:
                        deltas.append(np.nan)
                        continue
                    delta = (value - baseline_value) * metric_specs[metric]["scale"]
                    deltas.append(delta)

                offsets = y_positions + (agent_idx - (len(agent_order) - 1) / 2.0) * bar_height
                color = _agent_color(agent)
                label = _agent_label(agent)
                bars = ax.barh(
                    offsets,
                    deltas,
                    height=bar_height * 0.9,
                    color=color,
                    edgecolor=color,
                    alpha=0.85,
                    label=label,
                    zorder=3,
                )
                if label not in legend_handles:
                    legend_handles[label] = bars[0]

            ax.axvline(0.0, color="#2b2b2b", linewidth=0.8, linestyle="--", alpha=0.6, zorder=1)
            ax.set_yticks(y_positions)
            ax.set_yticklabels(scale_factor_labels)
            if col_idx == 0:
                ax.set_ylabel("Scale", fontsize=STYLE["labelsize"])
            else:
                ax.set_ylabel("")
            ax.set_xlabel(metric_specs[metric]["label"], fontsize=STYLE["labelsize"])
            panel_tag = next(panel_labels, '')
            title_suffix = param
            ax.set_title(f"({panel_tag}) {title_suffix}" if panel_tag else title_suffix, fontsize=STYLE["titlesize"])
            _apply_origin_axes_style(ax, grid_axis="x")
            ax.xaxis.set_major_locator(MaxNLocator(5))

    if legend_handles:
        label_to_handle = {lab: h for lab, h in legend_handles.items()}
        ordered_labels: List[str] = []
        for agent in agent_order:
            label = _agent_label(agent)
            if label in label_to_handle and label not in ordered_labels:
                ordered_labels.append(label)
        for label in legend_handles.keys():
            if label not in ordered_labels:
                ordered_labels.append(label)
        ordered_handles = [label_to_handle[label] for label in ordered_labels]
        legend = fig.legend(
            ordered_handles,
            ordered_labels,
            loc="upper center",
            ncol=len(ordered_labels),
            frameon=False,
            fontsize=max(STYLE["fontsize"] - 1, 6),
            bbox_to_anchor=(0.5, 0.92),
            borderaxespad=0.0,
        )
        legend._legend_box.sep = 0.25

    fig.tight_layout(rect=[0.04, 0.04, 0.98, 0.905])
    _save_figure(fig, save_path)
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")
    return save_path


def plot_threshold_window_robustness(summary_path: Path | None = None,
                                     save_path: Path | None = None) -> Path | None:
    if summary_path is None:
        summary_path = Path("output/data/p1_stability/e7_robustness_summary.csv")
    summary_path = Path(summary_path)
    if not summary_path.exists():
        print(f"⚠️  Skipping plot_threshold_window_robustness: {summary_path} not found.")
        return None

    threshold_data: Dict[float, Dict[str, float]] = {}
    threshold_feas: Dict[float, Dict[str, float]] = {}
    threshold_tau: Dict[float, float] = {}
    window_data: Dict[int, Dict[str, float]] = {}
    window_feas: Dict[int, Dict[str, float]] = {}
    window_tau: Dict[int, float] = {}
    agents_present: set[str] = set()

    with summary_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            scenario = row.get("scenario")
            agent = row.get("agent")
            if not scenario or not agent:
                continue
            try:
                ttm_mean = float(row["time_to_mastery_mean"])
                feasible_mean = float(row["feasible_rate_mean"]) * 100.0
            except (KeyError, TypeError, ValueError):
                continue
            agents_present.add(agent)
            tau_val = row.get("kendall_tau")
            tau_float = None
            if tau_val not in (None, ""):
                try:
                    tau_float = float(tau_val)
                except ValueError:
                    tau_float = None

            if scenario == "threshold":
                try:
                    scale = float(row["threshold_scale"])
                except (KeyError, TypeError, ValueError):
                    continue
                threshold_data.setdefault(scale, {})[agent] = ttm_mean
                threshold_feas.setdefault(scale, {})[agent] = feasible_mean
                if tau_float is not None:
                    threshold_tau[scale] = tau_float
            elif scenario == "window":
                try:
                    window = int(float(row["mastery_window"]))
                except (KeyError, TypeError, ValueError):
                    continue
                window_data.setdefault(window, {})[agent] = ttm_mean
                window_feas.setdefault(window, {})[agent] = feasible_mean
                if tau_float is not None:
                    window_tau[window] = tau_float

    if not threshold_data and not window_data:
        print("⚠️  Skipping plot_threshold_window_robustness: no data available.")
        return None

    agent_order = sorted(
        agents_present,
        key=lambda a: (0 if _agent_key(a) == "pianoMPC" else 1, AGENT_LABELS.get(_agent_key(a), a)),
    )

    if save_path is None:
        save_path = FIGURE_ROOT / "e7_threshold_window.png"
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("default")
    rc_updates = {k: v for k, v in STYLE.items() if k.startswith("figure.")}
    rc_updates.update({
        "figure.figsize": (10, 4),
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    })
    plt.rcParams.update(rc_updates)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), facecolor="white", sharey=False)

    def _plot_agent_lines(
        ax: plt.Axes,
        x_values: Sequence[float],
        data_block: Mapping[float | int, Mapping[str, float]],
        *,
        xlabel: str,
        ylabel: str,
        title: str,
        y_is_percent: bool = False,
    ) -> None:
        for agent in agent_order:
            color = _agent_color(agent)
            y_vals = [data_block.get(x_val, {}).get(agent, float("nan")) for x_val in x_values]
            if all(math.isnan(val) for val in y_vals):
                continue
            ax.plot(
                x_values,
                y_vals,
                marker="o",
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE,
                color=color,
                label=_agent_label(agent),
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x_values)
        if y_is_percent:
            finite_vals = [
                data_block.get(x_val, {}).get(agent, float("nan"))
                for x_val in x_values
                for agent in agent_order
            ]
            finite_vals = [val for val in finite_vals if not math.isnan(val)]
            if finite_vals:
                lower = max(0.0, min(finite_vals) - 5.0)
                ax.set_ylim(lower, 100.0)
        _apply_origin_axes_style(ax, grid_axis="both")

    # Threshold panel
    ax_thr = axes[0][0]
    scales_sorted = sorted(threshold_data.keys())
    _plot_agent_lines(
        ax_thr,
        scales_sorted,
        threshold_data,
        xlabel="Threshold scale",
        ylabel="TTM (steps)",
        title="(a) TTM under fatigue-threshold sweep",
    )

    # Mastery window panel
    ax_win = axes[0][1]
    windows_sorted = sorted(window_data.keys())
    _plot_agent_lines(
        ax_win,
        windows_sorted,
        window_data,
        xlabel="Mastery window W",
        ylabel="TTM (steps)",
        title="(b) TTM under mastery-window sweep",
    )
    for ax in (ax_win,):
        ax.axvline(3, color="#2b2b2b", linewidth=0.9, linestyle="--", alpha=0.65, zorder=1)
        ax.text(
            3.05,
            0.96,
            "Default $W=3$",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=max(STYLE["fontsize"] - 1, 7),
            color="#2b2b2b",
        )

    # Feasibility under threshold sweep
    ax_thr_feas = axes[1][0]
    _plot_agent_lines(
        ax_thr_feas,
        scales_sorted,
        threshold_feas,
        xlabel="Threshold scale",
        ylabel="Feasible rate (%)",
        title="(c) Feasibility under threshold sweep",
        y_is_percent=True,
    )

    # Feasibility under mastery-window sweep
    ax_win_feas = axes[1][1]
    _plot_agent_lines(
        ax_win_feas,
        windows_sorted,
        window_feas,
        xlabel="Mastery window W",
        ylabel="Feasible rate (%)",
        title="(d) Feasibility under mastery-window sweep",
        y_is_percent=True,
    )
    for ax in (ax_win_feas,):
        ax.axvline(3, color="#2b2b2b", linewidth=0.9, linestyle="--", alpha=0.65, zorder=1)
        ax.text(
            3.05,
            0.96,
            "Default $W=3$",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=max(STYLE["fontsize"] - 1, 7),
            color="#2b2b2b",
        )

    handles, labels = ax_thr.get_legend_handles_labels()
    if handles:
        label_to_handle = {lab: h for lab, h in zip(labels, handles)}
        ordered_labels: List[str] = []
        for agent in agent_order:
            lab = _agent_label(agent)
            if lab in label_to_handle and lab not in ordered_labels:
                ordered_labels.append(lab)
        for lab in labels:
            if lab not in ordered_labels:
                ordered_labels.append(lab)
        ordered_handles = [label_to_handle[lab] for lab in ordered_labels]
        fig.legend(
            ordered_handles,
            ordered_labels,
            loc="upper center",
            ncol=min(len(ordered_labels), 3),
            frameon=False,
            bbox_to_anchor=(0.5, 0.98),
            borderaxespad=0.2,
        )

    fig.tight_layout(rect=[0.04, 0.04, 0.98, 0.92])
    _save_figure(fig, save_path)
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")
    return save_path
def plot_ope_results(ope_errors: Dict | None = None,
                     save_path: Path | None = None) -> List[Path]:
    """Legacy helper for offline policy evaluation comparisons."""
    if ope_errors is None:
        print("⚠️  Skipping plot_ope_results: no data provided.")
        return []

    if save_path is None:
        save_path = FIGURE_ROOT / "ope_comparison.png"
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = ["mse", "mae", "rmse"]
    metric_titles = {"mse": "MSE", "mae": "MAE", "rmse": "RMSE"}

    datasets = sorted(ope_errors.keys(), key=lambda name: ope_errors[name].get("coverage_metric", 0.0))
    if not datasets:
        print("⚠️  Skipping plot_ope_results: empty dataset list.")
        return []

    methods_order = list(next(iter(ope_errors.values()))["methods"].keys())
    method_labels = [m.upper() for m in methods_order]

    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4), sharey=False, sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    x_positions = np.arange(len(methods_order))
    xtick_labels = method_labels

    coverage_points = {m.upper(): {"coverage": [], "rmse": []} for m in methods_order}
    dataset_labels = {}

    for dataset in datasets:
        coverage = ope_errors[dataset].get("coverage", {})
        ess = ope_errors[dataset].get("coverage_metric", coverage.get("ess", 0.0))
        mean_p = coverage.get("mean_prob", 0.0)
        dataset_labels[dataset] = f"{dataset} (ESS={ess:.0f}, μp={mean_p:.3f})"
        for method_key in methods_order:
            vals = ope_errors[dataset]["methods"][method_key]
            coverage_points[method_key.upper()]["coverage"].append(ess)
            coverage_points[method_key.upper()]["rmse"].append(vals.get("rmse", np.sqrt(vals.get("mse", 0.0))))

    for ax, metric in zip(axes, metrics):
        for dataset in datasets:
            values = [ope_errors[dataset]["methods"][method_key][metric] for method_key in methods_order]
            ax.plot(x_positions, values, marker="o", label=dataset_labels[dataset])
        ax.set_title(metric_titles[metric])
        ax.set_ylabel(metric_titles[metric])
        ax.grid(alpha=STYLE["alpha_grid"])
        ax.set_xticks(x_positions)
        ax.set_xticklabels(xtick_labels)

    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=4, fontsize=8)

    fig.tight_layout()
    _save_figure(fig, save_path)
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")

    coverage_paths: List[Path] = []
    if coverage_points:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        for method_label, vals in coverage_points.items():
            coverage_vals = np.array(vals["coverage"])
            rmse_vals = np.array(vals["rmse"])
            order = np.argsort(coverage_vals)
            ax2.plot(coverage_vals[order], rmse_vals[order], marker="o", label=method_label)
        ax2.set_xlabel("Effective Sample Size (ESS)")
        ax2.set_ylabel("RMSE")
        ax2.set_title("OPE RMSE vs Coverage")
        ax2.grid(alpha=STYLE["alpha_grid"])
        ax2.legend()
        coverage_path = save_path.with_name(save_path.stem + "_coverage.png")
        _save_figure(fig2, coverage_path)
        plt.close(fig2)
        print(f"  ✓ Saved: {coverage_path}")
        coverage_paths.append(coverage_path)

    return [save_path, *coverage_paths]


def main() -> None:
    print("Generating figures...")
    try:
        plot_comparison()
    except FileNotFoundError as exc:
        print(f"⚠️  {exc}")

    try:
        plot_learning_curves()
    except FileNotFoundError as exc:
        print(f"⚠️  {exc}")

    try:
        plot_pianoMPC_horizon()
    except FileNotFoundError as exc:
        print(f"⚠️  {exc}")

    try:
        plot_safety_tradeoff()
    except FileNotFoundError as exc:
        print(f"⚠️  {exc}")

    try:
        plot_safety_consistency()
    except FileNotFoundError as exc:
        print(f"⚠️  {exc}")

    try:
        plot_dynamics_mismatch()
    except FileNotFoundError as exc:
        print(f"⚠️  {exc}")

    try:
        plot_threshold_window_robustness()
    except FileNotFoundError as exc:
        print(f"⚠️  {exc}")

    # OPE plots require explicit data input; skip in CLI mode.


if __name__ == "__main__":
    main()
