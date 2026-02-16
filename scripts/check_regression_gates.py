#!/usr/bin/env python3
"""Evaluate must-pass trust regression gates against evaluation artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent.parent


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _get_nested(data: dict[str, Any], dotted_path: str) -> float | None:
    node: Any = data
    for key in dotted_path.split("."):
        if isinstance(node, dict) and key in node:
            node = node[key]
        else:
            return None
    if isinstance(node, (int, float)):
        return float(node)
    return None


def _pick_gate_sweep_row(rows: list[dict[str, Any]], profile_name: str | None) -> dict[str, Any]:
    """
    Pick the sweep row used for regression gates.

    With --profile, gates MUST run on the profile-default row, not the best row.
    """
    if not rows:
        raise ValueError("Sweep results are empty.")
    if not profile_name:
        return rows[0]

    profile_rows = [row for row in rows if row.get("trust_profile_name") == profile_name]
    if not profile_rows:
        raise ValueError(f"No sweep rows found for profile '{profile_name}'.")

    default_rows = [row for row in profile_rows if bool(row.get("is_profile_default_row"))]
    if len(default_rows) == 1:
        return default_rows[0]
    if len(default_rows) > 1:
        raise ValueError(
            f"Expected one profile-default row for profile '{profile_name}', found {len(default_rows)}."
        )
    raise ValueError(
        f"No profile-default row found for profile '{profile_name}'. "
        "Re-run sweep_retrieval_thresholds.py with updated row metadata."
    )


def _passes(operator: str, actual: float, expected: float) -> bool:
    if operator == "<=":
        return actual <= expected
    if operator == "<":
        return actual < expected
    if operator == ">=":
        return actual >= expected
    if operator == ">":
        return actual > expected
    if operator == "==":
        return actual == expected
    raise ValueError(f"Unsupported operator: {operator}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check trust regression gates.")
    parser.add_argument(
        "--contamination-report",
        type=Path,
        default=ROOT_DIR / "eval" / "law_contamination_report.json",
        help="Path to law contamination report JSON.",
    )
    parser.add_argument(
        "--sweep-results",
        type=Path,
        default=ROOT_DIR / "eval" / "retrieval_sweep_results.json",
        help="Path to retrieval sweep results JSON.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT_DIR / "eval" / "baselines" / "production_trust_baseline_v1.json",
        help="Path to baseline gate definition JSON.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Optional trust profile name to select from sweep rows (e.g. balanced_v1, strict_v1).",
    )
    args = parser.parse_args()

    contamination = _read_json(args.contamination_report)
    sweep_rows = _read_json(args.sweep_results)
    baseline = _read_json(args.baseline)

    if args.profile:
        contamination_profile = contamination.get("trust_profile_name")
        if contamination_profile and contamination_profile != args.profile:
            raise ValueError(
                f"Contamination report profile mismatch: report={contamination_profile} "
                f"requested={args.profile}"
            )

    selected_row = _pick_gate_sweep_row(sweep_rows, args.profile)
    metrics = {
        "contamination": contamination,
        "sweep": {"top": selected_row},
    }
    logger.info(
        "Gate sweep row selected: profile=%s is_profile_default_row=%s balanced_score=%.6f",
        selected_row.get("trust_profile_name"),
        selected_row.get("is_profile_default_row"),
        float(selected_row.get("balanced_score", 0.0)),
    )

    failures: list[str] = []
    passes = 0
    for gate in baseline.get("gates", []):
        metric_path = gate.get("metric")
        operator = gate.get("operator")
        expected = gate.get("value")
        if metric_path is None or operator is None or expected is None:
            failures.append(f"Invalid gate entry: {gate}")
            continue
        actual = _get_nested(metrics, metric_path)
        if actual is None:
            failures.append(f"{metric_path}: metric not found")
            continue
        ok = _passes(operator, actual, float(expected))
        marker = "PASS" if ok else "FAIL"
        logger.info(
            "[%s] %s (actual=%.6f, expected %s %.6f)",
            marker,
            metric_path,
            actual,
            operator,
            float(expected),
        )
        if ok:
            passes += 1
        else:
            failures.append(
                f"{metric_path}: actual={actual:.6f} expected {operator} {float(expected):.6f}"
            )

    logger.info("Gate checks passed: %s/%s", passes, len(baseline.get("gates", [])))
    if failures:
        logger.error("Regression gates failed:")
        for item in failures:
            logger.error("- %s", item)
        raise SystemExit(1)
    logger.info("All regression gates passed.")


if __name__ == "__main__":
    main()
