#!/usr/bin/env python
"""
Generate LaTeX tables for Paper 1 experiments.

Current coverage:
  - E1: Strategy comparison (Table 1)
  - E5: Suite robustness (Table 2)
All tables are written to output/paper1/tabel.tex (single aggregated file).
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from scipy import stats
except Exception:  # pragma: no cover - fallback when SciPy unavailable
    stats = None  # type: ignore


OUTPUT_ROOT = Path("output/paper1")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
TABLE_PATH = OUTPUT_ROOT / "tabel.tex"
DATA_ROOT = Path("output/data")

PROFILE_ORDER = ["balanced", "mild_left_weak", "severe_left_weak"]
PROFILE_LABELS = {
    "balanced": "Balanced",
    "mild_left_weak": "Mild left weakness",
    "severe_left_weak": "Severe left weakness",
}

AGENT_ORDER: List[str] = []

MetricSpec = Tuple[str, str, str, int, str]

TABLE_SPECS: List[Tuple[str, str, List[MetricSpec], str]] = [
    (
        "tab:e1-primary",
        "Overall comparison (time to mastery and feasible rate). Values are mean $\\pm$ 95\\% CI over {runs_desc}. Best values are highlighted in bold; second best are underlined.",
        [
            ("time_to_mastery_mean", "time_to_mastery_std", "TTM $\\downarrow$", 1, "min"),
            ("feasible_rate_mean", "feasible_rate_std", "Feas. $\\uparrow$", 2, "max"),
        ],
        "\\textbf{TTM}=time to mastery (steps); \\textbf{Feas.}=feasible rate.",
    ),
    (
        "tab:e1-secondary",
        "Complementary comparison (average fatigue and total raw reward). Values are mean $\\pm$ 95\\% CI over {runs_desc}. Best values are highlighted in bold; second best are underlined.",
        [
            ("avg_fatigue_mean", "avg_fatigue_std", "AvgFat.", 2, "min"),
            ("total_reward_raw_mean", "total_reward_raw_std", "RawRet.", 1, "max"),
        ],
        "\\textbf{AvgFat.}=average fatigue; \\textbf{RawRet.}=total raw reward.",
    ),
]


def _load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Required data file missing: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _t_critical(num_runs: int) -> float:
    if num_runs <= 1:
        return 0.0
    if stats is None:
        return 1.96  # Gaussian fallback
    try:
        return float(stats.t.ppf(0.975, num_runs - 1))
    except Exception:
        return 1.96


def _ci_half(std: float, num_runs: int, t_value: float) -> float:
    if num_runs <= 1 or std is None or math.isnan(std):
        return 0.0
    return float(std) * t_value / math.sqrt(num_runs)


def _format_cell(mean: float | None, std: float | None, decimals: int, num_runs: int, t_value: float) -> str:
    if mean is None:
        return "NA"
    try:
        mean_val = float(mean)
    except (TypeError, ValueError):
        return "NA"
    if math.isnan(mean_val):
        return "NA"

    std_val: float
    if std is None:
        std_val = 0.0
    else:
        try:
            std_val = float(std)
        except (TypeError, ValueError):
            std_val = 0.0
    if math.isnan(std_val):
        std_val = 0.0

    ci = _ci_half(std_val, num_runs, t_value)
    mean_fmt = f"{mean_val:.{decimals}f}"
    ci_fmt = f"{ci:.{decimals}f}"
    return f"{mean_fmt} $\\pm$ {ci_fmt}"


def _compute_column_best(
    metrics: Dict[str, Dict[str, Dict[str, float]]],
    metric_columns: List[MetricSpec],
) -> Dict[str, List[Tuple[float | None, float | None]]]:
    """Determine best and second-best values per profile/column for highlighting."""
    best_map: Dict[str, List[Tuple[float | None, float | None]]] = {}
    for profile in PROFILE_ORDER:
        profile_ranks: List[Tuple[float | None, float | None]] = []
        for mean_key, _, _, _, direction in metric_columns:
            values: List[float] = []
            for agent in AGENT_ORDER:
                agent_metrics = metrics.get(profile, {}).get(agent)
                if not agent_metrics:
                    continue
                mean_val = agent_metrics.get(mean_key)
                if mean_val is None:
                    continue
                try:
                    number = float(mean_val)
                except (TypeError, ValueError):
                    continue
                if not math.isnan(number):
                    values.append(number)

            if not values:
                profile_ranks.append((None, None))
                continue

            reverse = direction == "max"
            sorted_vals = sorted(values, reverse=reverse)

            unique_vals: List[float] = []
            for val in sorted_vals:
                if not unique_vals or not math.isclose(val, unique_vals[-1], rel_tol=1e-9, abs_tol=1e-9):
                    unique_vals.append(val)

            best = unique_vals[0] if unique_vals else None
            second = unique_vals[1] if len(unique_vals) > 1 else None
            profile_ranks.append((best, second))

        best_map[profile] = profile_ranks
    return best_map


def build_profile_table(
    metrics: Dict[str, Dict[str, Dict[str, float]]],
    num_runs: int,
    label: str,
    caption_template: str,
    metric_columns: List[MetricSpec],
    footnote: str,
) -> str:
    global AGENT_ORDER
    if not AGENT_ORDER:
        all_agents = list(metrics.get(PROFILE_ORDER[0], {}).keys())
        macro_order = sorted(
            all_agents,
            key=lambda agent: sum(
                metrics.get(profile, {}).get(agent, {}).get("time_to_mastery_mean", float("inf"))
                for profile in PROFILE_ORDER
            ) / max(len(PROFILE_ORDER), 1)
        )
        if "PianoMPC" in macro_order:
            macro_order.remove("PianoMPC")
            macro_order.insert(0, "PianoMPC")
        AGENT_ORDER.extend(macro_order)

    t_value = _t_critical(num_runs)
    best_map = _compute_column_best(metrics, metric_columns)

    runs_desc = (
        f"{num_runs} run{'s' if num_runs != 1 else ''}" if num_runs else "the collected runs"
    )
    caption = caption_template.format(runs_desc=runs_desc)

    header_top = [
        "    \\toprule",
        "    & " + " & ".join(
            f"\\multicolumn{{{len(metric_columns)}}}{{c}}{{{PROFILE_LABELS.get(profile, profile.title())}}}"
            for profile in PROFILE_ORDER
        ) + " \\\\",
    ]

    header_second_cells: List[str] = []
    for _ in PROFILE_ORDER:
        header_second_cells.extend(col_label for _, _, col_label, _, _ in metric_columns)
    header_second = "    Agent & " + " & ".join(header_second_cells) + " \\\\"

    lines: List[str] = [
        "\\begin{table}[t]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{l{'c' * len(header_second_cells)}}}",
    ]
    lines.extend(header_top)
    lines.append("    " + header_second)
    lines.append("    \\midrule")

    for agent in AGENT_ORDER:
        row_cells: List[str] = []
        has_data = False
        for profile in PROFILE_ORDER:
            agent_metrics = metrics.get(profile, {}).get(agent)
            if not agent_metrics:
                row_cells.extend(["NA"] * len(metric_columns))
                continue
            has_data = True
            for idx, (mean_key, std_key, _, decimals, _) in enumerate(metric_columns):
                mean = agent_metrics.get(mean_key)
                std = agent_metrics.get(std_key)
                cell = _format_cell(mean, std, decimals, num_runs, t_value)
                rank_vals = best_map.get(profile, [(None, None)] * len(metric_columns))[idx]
                best_val, second_val = rank_vals
                if mean is not None and (best_val is not None or second_val is not None):
                    try:
                        mean_number = float(mean)
                    except (TypeError, ValueError):
                        mean_number = None
                    if mean_number is not None and not math.isnan(mean_number):
                        if best_val is not None and math.isclose(mean_number, best_val, rel_tol=1e-9, abs_tol=1e-9):
                            cell = f"\\textbf{{{cell}}}"
                        elif second_val is not None and math.isclose(mean_number, second_val, rel_tol=1e-9, abs_tol=1e-9):
                            cell = f"\\underline{{{cell}}}"
                row_cells.append(cell)
        if not has_data:
            continue
        lines.append(f"    {agent} & " + " & ".join(row_cells) + " \\\\")

    total_columns = 1 + len(header_second_cells)
    lines.append("    \\bottomrule")
    lines.append(f"    \\multicolumn{{{total_columns}}}{{l}}{{\\small {footnote}}} \\\\")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def build_tables() -> List[str]:
    profile_path = DATA_ROOT / "comparison_profiles.json"
    try:
        data = _load_json(profile_path)
    except FileNotFoundError:
        return []

    num_runs = int(data.get("meta", {}).get("num_runs", 0))
    metrics: Dict[str, Dict[str, Dict[str, float]]] = data.get("metrics", {})
    tables: List[str] = []

    for label, caption, metric_columns, footnote in TABLE_SPECS:
        table_str = build_profile_table(metrics, num_runs, label, caption, metric_columns, footnote)
        tables.append(table_str)

    return tables


SUITE_CATEGORY_ORDER = ["basic", "shift", "safe"]
SUITE_CATEGORY_LABELS = {
    "basic": "basic",
    "shift": "shift",
    "safe": "safe",
}
SUITE_TRANSFER_ORDER = ["weak", "medium", "strong"]
SUITE_TOP_ORDER = ["PianoMPC", "CCB-DF", "BayesianMAB", "Thompson"]
SUITE_JSON_PATH = Path("output/data/p1_suite/e5_suite_results.json")


def _format_ttm_feas_pair(metrics: Dict[str, float], num_runs: int) -> Tuple[str, str]:
    if not metrics:
        return ("NA", "NA")
    ttm_mean = metrics.get("time_to_mastery_mean")
    feas_mean = metrics.get("feasible_rate_mean")
    if ttm_mean is None or feas_mean is None:
        return ("NA", "NA")
    try:
        ttm_val = float(ttm_mean)
        feas_val = float(feas_mean)
    except (TypeError, ValueError):
        return ("NA", "NA")
    if math.isnan(ttm_val) or math.isnan(feas_val):
        return ("NA", "NA")

    std_ttm = float(metrics.get("time_to_mastery_std", 0.0) or 0.0)
    std_feas = float(metrics.get("feasible_rate_std", 0.0) or 0.0)
    runs = max(int(metrics.get("num_runs", num_runs) or num_runs), 1)
    t_value = _t_critical(runs)
    ttm_ci = _ci_half(std_ttm, runs, t_value)
    feas_ci = _ci_half(std_feas, runs, t_value)
    ttm_str = f"{ttm_val:.1f} $\\pm$ {ttm_ci:.1f}"
    feas_str = f"{feas_val:.2f} $\\pm$ {feas_ci:.2f}"
    return (ttm_str, feas_str)


def build_suite_table() -> str | None:
    if not SUITE_JSON_PATH.exists():
        return None
    data = _load_json(SUITE_JSON_PATH)
    records: List[Dict[str, object]] = data.get("records", [])  # type: ignore
    if not records:
        return None

    agent_metrics: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for rec in records:
        agent = rec.get("agent")
        category = rec.get("category")
        transfer = rec.get("transfer_strength")
        if not isinstance(agent, str) or not isinstance(category, str) or not isinstance(transfer, str):
            continue
        metrics: Dict[str, float] = {}
        for key, value in rec.items():
            if isinstance(value, (int, float)) and (key.endswith("_mean") or key.endswith("_std")):
                metrics[key] = float(value)
        metrics["num_runs"] = int(rec.get("num_runs", data.get("meta", {}).get("num_runs", 0)))
        agent_metrics[agent][category][transfer] = metrics

    agent_order = [
        agent for agent in SUITE_TOP_ORDER if agent in agent_metrics
    ]
    for agent in sorted(agent_metrics.keys()):
        if agent not in agent_order:
            agent_order.append(agent)

    if not agent_order:
        return None

    caption = (
        "Robustness of top agents across the 3$\\times$3 task suite "
        "(category $\\times$ transfer strength). Each cell reports "
        "TTM / Feas. (mean $\\pm$ 95\\% CI)."
    )
    lines: List[str] = [
        "\\begin{table}[t]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        "  \\label{tab:e5-suite}",
        "  \\begin{tabular}{llcccccc}",
        "    \\toprule",
        "    Agent & Category & \\multicolumn{2}{c}{weak} & \\multicolumn{2}{c}{medium} & \\multicolumn{2}{c}{strong} \\\\",
        "     &  & TTM $\\downarrow$ & Feas. $\\uparrow$ & TTM $\\downarrow$ & Feas. $\\uparrow$ & TTM $\\downarrow$ & Feas. $\\uparrow$ \\\\",
        "    \\midrule",
    ]

    for idx, agent in enumerate(agent_order):
        suites = agent_metrics[agent]
        for cat_idx, category in enumerate(SUITE_CATEGORY_ORDER):
            row_cells: List[str] = []
            if cat_idx == 0:
                row_cells.append(agent)
            else:
                row_cells.append("")
            row_cells.append(SUITE_CATEGORY_LABELS.get(category, category))
            for transfer in SUITE_TRANSFER_ORDER:
                metrics = suites.get(category, {}).get(transfer)
                ttm_cell, feas_cell = _format_ttm_feas_pair(metrics, data.get("meta", {}).get("num_runs", 0))
                row_cells.extend([ttm_cell, feas_cell])
            lines.append("    " + " & ".join(row_cells) + " \\\\")
        if idx != len(agent_order) - 1:
            lines.append("    \\midrule")

    footnote = (
        "\\textbf{TTM}=time to mastery (steps); "
        "\\textbf{Feas.}=feasible rate."
    )

    lines.append("    \\bottomrule")
    lines.append(f"    \\multicolumn{{8}}{{l}}{{\\small {footnote}}} \\\\")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main() -> None:
    tables: List[str] = ["% Auto-generated tables for RhythmGym Paper 1"]

    profile_tables = build_tables()
    if profile_tables:
        tables.extend(profile_tables)
    else:
        tables.append("% E1 tables skipped: comparison_profiles.json not found.")

    suite_table = build_suite_table()
    if suite_table:
        tables.append(suite_table)
    else:
        tables.append("% E5 table skipped: e5_suite_results.json not found or empty.")

    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text("\n\n".join(tables) + "\n", encoding="utf-8")
    print(f"✓ Wrote {TABLE_PATH}")


if __name__ == "__main__":
    main()
