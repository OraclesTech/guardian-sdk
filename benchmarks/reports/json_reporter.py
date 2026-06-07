"""
JSONReporter — writes machine-parseable benchmark results.

JSON output is designed for:
  - CI/CD pipeline consumption (pass/fail thresholds)
  - Dashboard ingestion
  - Historical trend tracking
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class JSONReporter:
    """
    Writes benchmark results to a structured JSON file.

    Schema:
    {
      "meta": { version, tier, generated_at, datasets, total_samples },
      "summary": { precision, recall, f1, fpr, fnr, roc_auc, p50_ms, p99_ms },
      "datasets": [ { ...ClassificationMetrics fields + latency } ],
      "baselines": { "lakera_full_context": {...}, ... }
    }
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or (
            Path(__file__).parent.parent / "results"
        )

    def write(
        self,
        run_data: Dict[str, Any],
        baselines: Optional[Dict] = None,
        filename: Optional[str] = None,
        filename_suffix: str = "",
    ) -> Path:
        """Render and write JSON results to disk.

        Args:
            run_data:        Benchmark data dict from build_run_data().
            baselines:       Published baseline metrics to merge into output.
            filename:        Override the full filename (including extension).
            filename_suffix: Appended before the .json extension in the auto-
                             generated filename (e.g. "_pass1", "_pass2").
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        meta = run_data.get("meta", {})
        if filename is None:
            version = meta.get("guardian_version", "unknown")
            tier = meta.get("tier", "community")
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            filename = f"guardian_v{version}_{tier}_{date}{filename_suffix}.json"

        # Merge baselines into output
        payload = dict(run_data)
        if baselines:
            payload["baselines"] = {
                k: v.as_dict() if hasattr(v, "as_dict") else v
                for k, v in baselines.items()
            }

        output_path = self.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

        logger.info("JSON report written: %s", output_path)
        return output_path

    def check_thresholds(
        self,
        summary: Dict[str, Any],
        min_precision: float = 0.80,
        min_recall: float = 0.70,
        max_fpr: float = 0.05,
        max_p99_ms: float = 200.0,
    ) -> Dict[str, bool]:
        """
        Returns a pass/fail dict for CI integration.

        Returns:
            dict mapping metric → True (passed) / False (failed)
        """
        return {
            "precision": (summary.get("precision") or 0.0) >= min_precision,
            "recall": (summary.get("recall") or 0.0) >= min_recall,
            "fpr": (summary.get("fpr") or 1.0) <= max_fpr,
            "p99_latency": (summary.get("p99_ms") or float("inf")) <= max_p99_ms,
        }
