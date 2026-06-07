"""
BenchmarkRunner — abstract base class for all runners.

A runner takes a list of texts and returns a list of Predictions,
each including a binary label and an optional threat score (for AUC).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Prediction:
    """Single prediction from a runner."""
    text: str
    predicted_label: int          # 1 = threat detected, 0 = clean
    threat_score: float           # 0.0 – 1.0 continuous score for AUC
    latency_ms: float             # per-request latency
    threat_level: str = "NONE"    # NONE / LOW / MEDIUM / HIGH / CRITICAL
    recommended_action: str = "ALLOW"   # ALLOW / CHALLENGE / BLOCK
    metadata: Dict = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Aggregate result from one benchmark run."""
    runner_name: str
    dataset_name: str
    predictions: List[Prediction] = field(default_factory=list)
    run_metadata: Dict = field(default_factory=dict)

    @property
    def y_pred(self) -> List[int]:
        return [p.predicted_label for p in self.predictions]

    @property
    def y_scores(self) -> List[float]:
        return [p.threat_score for p in self.predictions]

    @property
    def latencies_ms(self) -> List[float]:
        return [p.latency_ms for p in self.predictions]


class BenchmarkRunner(ABC):
    """Abstract base class for benchmark runners."""

    name: str = "base"

    @abstractmethod
    async def run(self, texts: List[str]) -> List[Prediction]:
        """
        Run inference on a list of texts and return predictions.

        Args:
            texts: List of prompt texts to classify.

        Returns:
            List of Prediction objects, one per text, in the same order.
        """

    async def run_dataset(
        self,
        samples: List,                    # List[BenchmarkSample]
        show_progress: bool = True,
    ) -> BenchmarkResult:
        """
        Convenience wrapper: run on dataset samples, return BenchmarkResult.
        """
        from tqdm.asyncio import tqdm as async_tqdm  # noqa: PLC0415

        texts = [s.text for s in samples]
        dataset_name = samples[0].source if samples else "unknown"

        if show_progress:
            predictions = []
            with async_tqdm(total=len(texts), desc=f"[{self.name}]", unit="prompt") as pbar:
                # Process in batches to show progress
                batch_size = 50
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    batch_preds = await self.run(batch)
                    predictions.extend(batch_preds)
                    pbar.update(len(batch))
        else:
            predictions = await self.run(texts)

        return BenchmarkResult(
            runner_name=self.name,
            dataset_name=dataset_name,
            predictions=predictions,
        )
