"""Dependency-free aggregation for method-runtime breakdown reports."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping


class RuntimeProfiler:
    """Accumulate per-iteration and one-off timings for a training run."""

    COMPONENT_NAMES = (
        "data_fetch",
        "forward",
        "composition_loss",
        "backward",
        "densification_pruning",
        "optimizer",
    )
    ONE_OFF_NAMES = (
        "pyramid_precompute",
        "stage_cache_build",
        "inheritance",
        "archive_io",
    )
    # CUDA events expose float milliseconds. Independently accumulated outer
    # and component timings may therefore differ by up to 2 microseconds per
    # training iteration without indicating missing work.
    _RESIDUAL_ABSOLUTE_TOLERANCE_S = 1e-9
    _RESIDUAL_TOLERANCE_PER_ITERATION_S = 2e-6
    _RESIDUAL_RELATIVE_TOLERANCE = 1e-12

    def __init__(self) -> None:
        self._components = {name: self._empty_measurement() for name in self.COMPONENT_NAMES}
        self._one_off_components = {
            name: self._empty_measurement() for name in self.ONE_OFF_NAMES
        }
        self._triggered_total_s = 0.0
        self._triggered_count = 0
        self._stage_iterations: Dict[str, int] = {}
        self._total_iterations = 0
        self._optimization_time_s: float | None = None

    @staticmethod
    def _empty_measurement() -> Dict[str, Any]:
        return {"total_s": 0.0, "count": 0}

    @staticmethod
    def _validate_seconds(seconds: float) -> float:
        value = float(seconds)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("seconds must be finite and non-negative")
        return value

    @staticmethod
    def _validate_count(count: int) -> int:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("count must be a non-negative integer")
        return count

    @staticmethod
    def _add(measurement: Dict[str, Any], seconds: float, count: int) -> None:
        measurement["total_s"] += seconds
        measurement["count"] += count

    def add_iteration(self, stage: str) -> None:
        if not isinstance(stage, str) or not stage:
            raise ValueError("stage must be a non-empty string")
        self._total_iterations += 1
        self._stage_iterations[stage] = self._stage_iterations.get(stage, 0) + 1

    def add_component_seconds(self, name: str, seconds: float, count: int = 1) -> None:
        if name not in self._components:
            raise ValueError("unknown runtime component: {}".format(name))
        self._add(
            self._components[name], self._validate_seconds(seconds), self._validate_count(count)
        )

    def add_triggered_seconds(self, seconds: float, count: int = 1) -> None:
        seconds_value = self._validate_seconds(seconds)
        count_value = self._validate_count(count)
        self._triggered_total_s += seconds_value
        self._triggered_count += count_value

    def add_one_off_seconds(self, name: str, seconds: float, count: int = 1) -> None:
        if name not in self._one_off_components:
            raise ValueError("unknown one-off component: {}".format(name))
        self._add(
            self._one_off_components[name],
            self._validate_seconds(seconds),
            self._validate_count(count),
        )

    def set_optimization_time(self, seconds: float) -> None:
        self._optimization_time_s = self._validate_seconds(seconds)

    @staticmethod
    def _measurement_report(measurement: Mapping[str, Any], iterations: int) -> Dict[str, Any]:
        total_s = measurement["total_s"]
        count = measurement["count"]
        return {
            "total_s": total_s,
            "count": count,
            "mean_ms": (1000.0 * total_s / count) if count else 0.0,
            "amortized_ms_per_iteration": (1000.0 * total_s / iterations)
            if iterations
            else 0.0,
        }

    def to_dict(self, metadata: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be a mapping")

        component_total_s = sum(
            measurement["total_s"] for measurement in self._components.values()
        )
        optimization_time_s = (
            self._optimization_time_s
            if self._optimization_time_s is not None
            else component_total_s
        )
        known_optimization_s = component_total_s
        raw_residual_s = optimization_time_s - known_optimization_s
        reconciliation_tolerance_s = max(
            self._RESIDUAL_ABSOLUTE_TOLERANCE_S,
            self._total_iterations * self._RESIDUAL_TOLERANCE_PER_ITERATION_S,
            self._RESIDUAL_RELATIVE_TOLERANCE
            * max(abs(optimization_time_s), abs(known_optimization_s)),
        )
        if raw_residual_s < -reconciliation_tolerance_s:
            raise ValueError("component timings exceed optimization_time_s")
        other_optimization_s = max(0.0, raw_residual_s)
        reconciliation_error_s = optimization_time_s - (
            known_optimization_s + other_optimization_s
        )

        components = {
            name: self._measurement_report(measurement, self._total_iterations)
            for name, measurement in self._components.items()
        }
        components["other_optimization"] = self._measurement_report(
            {"total_s": other_optimization_s, "count": self._total_iterations},
            self._total_iterations,
        )
        one_off_components = {
            name: self._measurement_report(measurement, self._total_iterations)
            for name, measurement in self._one_off_components.items()
        }
        one_off_total_s = sum(
            measurement["total_s"] for measurement in self._one_off_components.values()
        )
        triggered = {
            "triggered_total_s": self._triggered_total_s,
            "count": self._triggered_count,
            "mean_ms": (1000.0 * self._triggered_total_s / self._triggered_count)
            if self._triggered_count
            else 0.0,
        }

        report = {"metadata": dict(metadata)}
        report.update(
            {
                "schema_version": 1,
                "total_iterations": self._total_iterations,
                "stage_iterations": dict(self._stage_iterations),
                "optimization_time_s": optimization_time_s,
                "known_optimization_s": known_optimization_s,
                "other_optimization_s": other_optimization_s,
                "reconciliation_error_s": reconciliation_error_s,
                "reconciliation_tolerance_s": reconciliation_tolerance_s,
                "inclusive_time_s": optimization_time_s + one_off_total_s,
                "one_off_components": one_off_components,
                "components": components,
                "triggered_densification": triggered,
            }
        )
        return report

    def write_json(self, path: str | Path, metadata: Mapping[str, Any]) -> Dict[str, Any]:
        report = self.to_dict(metadata)
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return report
