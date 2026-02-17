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


def _artifact_metadata_from_sweep_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract run metadata from sweep rows (uses first row as source of truth)."""
    if not rows:
        return {}
    first = rows[0]
    return {
        "run_id": first.get("run_id"),
        "run_started_at": first.get("run_started_at"),
        "git_commit": first.get("git_commit"),
        "questions_sha256": first.get("questions_sha256"),
        "question_count": first.get("question_count"),
        "trust_profile_name": first.get("trust_profile_name"),
        "trust_profile_version": first.get("trust_profile_version"),
    }


def _validate_artifact_compatibility(
    contamination: dict[str, Any],
    sweep_rows: list[dict[str, Any]],
) -> None:
    """Fail fast if contamination/sweep artifacts appear to come from different runs."""
    contamination_meta = contamination.get("artifact_metadata") or {}
    sweep_meta = _artifact_metadata_from_sweep_rows(sweep_rows)
    required_keys = ["run_id", "git_commit", "questions_sha256", "question_count"]
    missing_meta: list[str] = []
    for key in required_keys:
        if contamination_meta.get(key) in (None, ""):
            missing_meta.append(f"contamination.{key}")
        if sweep_meta.get(key) in (None, ""):
            missing_meta.append(f"sweep.{key}")
    if missing_meta:
        raise ValueError(
            "Artifact metadata missing required fields. "
            "Re-run contamination+sweep in the same run envelope. Missing: "
            + ", ".join(missing_meta)
        )
    logger.info(
        "Artifact metadata: contamination.run_id=%s sweep.run_id=%s contamination.git=%s sweep.git=%s",
        contamination_meta.get("run_id"),
        sweep_meta.get("run_id"),
        contamination_meta.get("git_commit"),
        sweep_meta.get("git_commit"),
    )
    keys_to_match = required_keys + ["trust_profile_name", "trust_profile_version"]
    mismatches: list[str] = []
    for key in keys_to_match:
        left = contamination_meta.get(key)
        right = sweep_meta.get(key)
        if left is None or right is None:
            continue
        if str(left) != str(right):
            mismatches.append(f"{key}: contamination={left} sweep={right}")
    if mismatches:
        raise ValueError(
            "Artifact compatibility check failed. Contamination and sweep reports do not match: "
            + "; ".join(mismatches)
        )


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
    parser.add_argument(
        "--gate-tier",
        type=str,
        default="v1",
        choices=["v1", "v2", "v3"],
        help="Gate strictness tier to evaluate (v1 current baseline, v2 stricter, v3 target).",
    )
    args = parser.parse_args()

    contamination = _read_json(args.contamination_report)
    sweep_rows = _read_json(args.sweep_results)
    baseline = _read_json(args.baseline)
    _validate_artifact_compatibility(contamination, sweep_rows)

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

    if args.gate_tier == "v1":
        gates = baseline.get("gates", [])
    elif args.gate_tier == "v2":
        gates = baseline.get("gates_v2", baseline.get("gates", []))
    else:
        gates = baseline.get("gates_v3", baseline.get("gates_v2", baseline.get("gates", [])))

    failures: list[str] = []
    passes = 0
    logger.info("Evaluating gate tier: %s (%s checks)", args.gate_tier, len(gates))
    for gate in gates:
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

    logger.info("Gate checks passed: %s/%s", passes, len(gates))
    if failures:
        logger.error("Regression gates failed:")
        for item in failures:
            logger.error("- %s", item)
        raise SystemExit(1)
    logger.info("All regression gates passed.")


if __name__ == "__main__":
    main()
