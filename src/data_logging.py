"""
Data logging helpers for RhythmGym experiments.

Provides a consistent way to create output directories and dump
per-experiment records (CSV/JSON/JSONL) before plotting.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


OUTPUT_DATA_ROOT = Path("output/data")


# Directory scaffold for experiments
DATA_SUBDIRS = {
    "p1_safety": OUTPUT_DATA_ROOT / "p1_safety",
    "p1_misspec": OUTPUT_DATA_ROOT / "p1_misspec",
    "p1_stability": OUTPUT_DATA_ROOT / "p1_stability",
}


@dataclass(frozen=True)
class CsvSpec:
    """Convenience wrapper describing a CSV output target."""

    section: str  # key from DATA_SUBDIRS
    filename: str
    fieldnames: Sequence[str]

    def resolve_path(self) -> Path:
        base = DATA_SUBDIRS.get(self.section)
        if base is None:
            raise KeyError(f"Unknown data section: {self.section}")
        return base / self.filename


def ensure_data_dirs() -> dict[str, Path]:
    """Create the standard output/data directory scaffold if missing."""
    OUTPUT_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for path in DATA_SUBDIRS.values():
        path.mkdir(parents=True, exist_ok=True)
    return DATA_SUBDIRS.copy()


def resolve_data_path(section: str, filename: str) -> Path:
    """Return a concrete Path under output/data for the given section."""
    if section not in DATA_SUBDIRS:
        raise KeyError(f"Unknown data section: {section}")
    ensure_data_dirs()
    return DATA_SUBDIRS[section] / filename


def write_csv_records(
    path: Path,
    records: Iterable[Mapping[str, object]],
    fieldnames: Sequence[str],
    append: bool = False,
) -> None:
    """
    Write a sequence of records to CSV.

    Parameters:
        path: destination file path.
        records: iterable of dict-like objects.
        fieldnames: column order for the CSV.
        append: when True, append to the file; otherwise overwrite.
    """
    # Ensure parent directory exists even if caller bypassed resolve_data_path.
    path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"
    record_iter = iter(records)
    peek = None
    try:
        peek = next(record_iter)
    except StopIteration:
        return  # nothing to write

    # Reconstruct iterator including the peeked record.
    def _record_iterator():
        yield peek
        yield from record_iter

    write_header = True
    if append and path.exists():
        try:
            write_header = path.stat().st_size == 0
        except OSError:
            write_header = True

    with path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in _record_iterator():
            writer.writerow(row)


def write_json(data: object, path: Path, indent: int = 2) -> None:
    """Persist JSON with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=True)


def write_jsonl(records: Iterable[Mapping[str, object]], path: Path) -> None:
    """Write records to a JSON Lines file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            json.dump(record, f, ensure_ascii=True)
            f.write("\n")


# Common field templates, reused by multiple experiments.
COMMON_STEP_FIELDS = [
    "seed",
    "profile",
    "agent",
    "step",
    "action_id",
    "reward",
    "mastery_flag",
    "f_t",
    "window_max",
    "predicted_peak",
    "actual_peak",
    "guard_pass",
    "guard_threshold",
    "lambda_t",
    "will_block",
    "violation_window_flag",
    "violation_peak_flag",
]


def default_step_fieldnames(extra_fields: Sequence[str] | None = None) -> list[str]:
    """Return the canonical step-level field list with optional extras."""
    if not extra_fields:
        return list(COMMON_STEP_FIELDS)
    merged = list(COMMON_STEP_FIELDS)
    for field in extra_fields:
        if field not in merged:
            merged.append(field)
    return merged
