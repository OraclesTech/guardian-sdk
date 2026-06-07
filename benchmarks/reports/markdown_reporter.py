"""
MarkdownReporter — renders a human-readable Markdown benchmark report.

Uses Jinja2 for templating. Output is a single .md file suitable for
committing to the repo or publishing as a GitHub Releases asset.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

TEMPLATE_PATH = Path(__file__).parent / "templates" / "report.md.jinja"


def _pct(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}"


def _fmt1(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value:.1f}"


def _fmt3(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}"


def _delta_pct(delta: float) -> str:
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.3f}"


def _delta_ms(delta: float) -> str:
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}ms"


def _format_int(value: int) -> str:
    return f"{value:,}"


class MarkdownReporter:
    """
    Renders benchmark results to a Markdown report file.

    Usage:
        reporter = MarkdownReporter(output_dir=Path("benchmarks/results"))
        reporter.write(run_data, filename="guardian_v1.3.0_quick.md")
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or (
            Path(__file__).parent.parent / "results"
        )

    def render(self, run_data: Dict[str, Any]) -> str:
        """Render the report template to a string."""
        try:
            from jinja2 import Environment, FileSystemLoader, StrictUndefined  # noqa: PLC0415
        except ImportError:
            raise ImportError(
                "jinja2 is required for report generation.\n"
                "Install: pip install jinja2"
            ) from None

        env = Environment(
            loader=FileSystemLoader(str(TEMPLATE_PATH.parent)),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Register custom filters
        env.filters["pct"] = _pct
        env.filters["fmt1"] = _fmt1
        env.filters["fmt3"] = _fmt3
        env.filters["delta_pct"] = _delta_pct
        env.filters["delta_ms"] = _delta_ms
        env.filters["format_int"] = _format_int

        template = env.get_template(TEMPLATE_PATH.name)
        return template.render(**run_data)

    def write(
        self,
        run_data: Dict[str, Any],
        filename: Optional[str] = None,
        filename_suffix: str = "",
    ) -> Path:
        """Render and write the report to disk.

        Args:
            run_data:        Template context dict from build_run_data().
            filename:        Override the full filename (including extension).
            filename_suffix: Appended before the .md extension in the auto-
                             generated filename (e.g. "_pass1", "_pass2").
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            version = run_data.get("meta", {}).get("guardian_version", "unknown")
            tier = run_data.get("meta", {}).get("tier", "community")
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            filename = f"guardian_v{version}_{tier}_{date}{filename_suffix}.md"

        content = self.render(run_data)
        output_path = self.output_dir / filename
        output_path.write_text(content, encoding="utf-8")
        logger.info("Markdown report written: %s", output_path)
        return output_path


def build_run_data(
    classification_results: List[Any],   # List[ClassificationMetrics]
    latency_results: List[Any],          # List[LatencyMetrics]
    guardian_version: str,
    tier: str,
) -> Dict[str, Any]:
    """
    Build the context dict for the Jinja2 template from raw metric objects.
    """
    from ..metrics.classification import ClassificationMetrics  # noqa: PLC0415
    from ..metrics.latency import LatencyMetrics  # noqa: PLC0415

    # Aggregate summary (macro-average across all datasets)
    if classification_results:
        avg = lambda key: sum(getattr(r, key) for r in classification_results) / len(classification_results)  # noqa: E731
        summary = {
            "precision": avg("precision"),
            "recall": avg("recall"),
            "f1": avg("f1"),
            "fpr": avg("fpr"),
            "fnr": avg("fnr"),
            "roc_auc": avg("roc_auc"),
        }
    else:
        summary = {k: None for k in ["precision", "recall", "f1", "fpr", "fnr", "roc_auc"]}

    # Aggregate latency
    if latency_results:
        all_p50 = [lr.p50_ms for lr in latency_results]
        all_p99 = [lr.p99_ms for lr in latency_results]
        summary["p50_ms"] = sum(all_p50) / len(all_p50)
        summary["p99_ms"] = sum(all_p99) / len(all_p99)
    else:
        summary["p50_ms"] = None
        summary["p99_ms"] = None

    # Per-dataset rows (merge classification + latency by index)
    datasets_rows = []
    for i, cr in enumerate(classification_results):
        lr = latency_results[i] if i < len(latency_results) else None
        row = cr.as_dict()
        if lr:
            row.update({
                "p50_ms": lr.p50_ms,
                "p95_ms": lr.p95_ms,
                "p99_ms": lr.p99_ms,
                "mean_ms": lr.mean_ms,
            })
        else:
            row.update({"p50_ms": None, "p95_ms": None, "p99_ms": None, "mean_ms": None})
        datasets_rows.append(row)

    total_samples = sum(cr.n_samples for cr in classification_results)
    dataset_names = list({cr.dataset for cr in classification_results})

    # Dataset info for methodology table
    dataset_info = []
    for cr in classification_results:
        attack_only = cr.n_benign == 0
        benign_only = cr.n_attacks == 0
        label_type = (
            "attack only" if attack_only
            else "benign only" if benign_only
            else "binary (attack + benign)"
        )
        dataset_info.append({
            "name": cr.dataset,
            "n_samples": cr.n_samples,
            "label_type": label_type,
            "source": "HuggingFace" if cr.dataset != "bundled" else "Bundled (local)",
        })

    return {
        "meta": {
            "guardian_version": guardian_version,
            "tier": tier,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "datasets": dataset_names,
            "total_samples": total_samples,
            "dataset_info": dataset_info,
        },
        "summary": summary,
        "datasets": datasets_rows,
    }
