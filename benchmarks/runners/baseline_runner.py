"""
BaselineRunner — returns published benchmark numbers as static BenchmarkResult.

Used to populate the comparison table without live API calls.
All numbers sourced from peer-reviewed publications (see SOURCES below).

SOURCES:
  - Lakera Guard: Palit et al. (arXiv:2505.13028, 2025)
  - RF + OpenAI embeddings: Perez & Ribeiro (2022), academic baseline
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class PublishedBaseline:
    """Published benchmark result from an external system."""
    system: str
    context: str                   # "full-context" | "no-context"
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    fpr: Optional[float] = None
    fnr: Optional[float] = None
    roc_auc: Optional[float] = None
    latency_p50_ms: Optional[float] = None
    source: str = ""               # citation
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "system": self.system,
            "context": self.context,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "fpr": self.fpr,
            "fnr": self.fnr,
            "roc_auc": self.roc_auc,
            "latency_p50_ms": self.latency_p50_ms,
            "source": self.source,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Published baselines table
# ---------------------------------------------------------------------------

PUBLISHED_BASELINES: Dict[str, PublishedBaseline] = {
    "lakera_full_context": PublishedBaseline(
        system="Lakera Guard",
        context="full-context",
        precision=0.984,
        recall=0.707,
        f1=0.823,
        fpr=0.007,
        fnr=0.293,
        latency_p50_ms=51.0,
        source="Palit et al., arXiv:2505.13028 (2025)",
        notes="Scenario-based evaluation with full document context",
    ),
    "lakera_no_context": PublishedBaseline(
        system="Lakera Guard",
        context="no-context",
        precision=0.964,
        recall=0.501,
        f1=None,
        fpr=0.057,
        fnr=0.499,
        latency_p50_ms=51.0,
        source="Palit et al., arXiv:2505.13028 (2025)",
        notes="Standalone prompt evaluation without document context",
    ),
    "rf_openai_embeddings": PublishedBaseline(
        system="RF + OpenAI Embeddings",
        context="no-context",
        precision=0.867,
        recall=0.870,
        f1=None,
        roc_auc=0.764,
        source="Academic baseline (embedding-based classification, 2022-2023)",
        notes="Random Forest classifier on OpenAI text-embedding-ada-002 features",
    ),
}


def get_baselines() -> Dict[str, PublishedBaseline]:
    """Return all published baselines."""
    return PUBLISHED_BASELINES


def get_baseline(key: str) -> Optional[PublishedBaseline]:
    """Return a specific baseline by key."""
    return PUBLISHED_BASELINES.get(key)
