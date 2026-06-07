"""
DatasetLoader — abstract base class and registry for all benchmark datasets.

Each concrete loader returns a list of BenchmarkSample with:
  - text: the prompt to classify
  - label: 1 = attack/injection, 0 = benign
  - category: attack type string if known, else None
  - source: dataset name for audit trail
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Type

logger = logging.getLogger(__name__)

BUNDLED_DIR = Path(__file__).parent / "bundled"


# ---------------------------------------------------------------------------
# Core dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkSample:
    """A single prompt with its ground-truth label."""
    text: str
    label: int              # 1 = attack, 0 = benign
    category: Optional[str] = None   # attack category if labeled
    source: str = ""        # dataset name for traceability
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class DatasetLoader(ABC):
    """Abstract base class for all benchmark dataset loaders."""

    name: str = "unnamed"          # Override in each subclass
    requires_hf: bool = False      # True → needs `datasets` + HF download
    default_sample_limit: int = 0  # 0 = no limit; override for expensive datasets
    cl_safe: bool = False
    """
    Whether false negatives (missed attacks) from this dataset can safely be
    submitted to the AdversarialLearner in the 2-pass benchmark.

    cl_safe=True  — Dataset has verified attack AND benign labels; FNs represent
                    genuine missed detections and teach the AL valid patterns
                    without risk of broadening fingerprints into benign space.
    cl_safe=False — Exclude from the learning pass. Reasons include:
                    • Benign class contains mislabeled harmful samples (neuralchemy)
                    • Dataset is attack-only with abstract harmful behaviors rather
                      than actual injection text — semantics differ from what
                      Guardian is tuned to detect, so learned fingerprints pollute
                      the pattern space and raise FPR on unrelated benign content
                      (jailbreakbench, advbench, harmbench, b3, salad_data)
                    • Threat surface too broad to fingerprint safely without
                      contaminating the in-memory Guardian instance used for Pass 2
    """

    @abstractmethod
    def load(self, sample_limit: int = 0) -> List[BenchmarkSample]:
        """
        Load and return benchmark samples.

        Args:
            sample_limit: Max samples to return. 0 = no limit.
                          Negative → stratified sample of abs(sample_limit) per class.

        Returns:
            List of BenchmarkSample.
        """

    def _enforce_limit(
        self,
        samples: List[BenchmarkSample],
        limit: int,
    ) -> List[BenchmarkSample]:
        """Stratified truncation: equal attack/benign distribution."""
        if limit <= 0 or len(samples) <= limit:
            return samples

        attacks = [s for s in samples if s.label == 1]
        benigns = [s for s in samples if s.label == 0]
        half = limit // 2
        return attacks[:half] + benigns[:half]


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

class DatasetRegistry:
    """Registry mapping dataset name → DatasetLoader class."""

    _registry: Dict[str, Type[DatasetLoader]] = {}

    @classmethod
    def register(cls, loader_cls: Type[DatasetLoader]) -> Type[DatasetLoader]:
        """Decorator to register a loader class."""
        cls._registry[loader_cls.name] = loader_cls
        return loader_cls

    @classmethod
    def get(cls, name: str) -> DatasetLoader:
        """Instantiate and return a loader by name."""
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(
                f"Unknown dataset '{name}'. Available: {available}"
            )
        return cls._registry[name]()

    @classmethod
    def all_names(cls) -> List[str]:
        return sorted(cls._registry.keys())

    @classmethod
    def quick_names(cls) -> List[str]:
        """Datasets that don't require HF download (usable offline)."""
        return [
            name for name, cls_ in cls._registry.items()
            if not cls_.requires_hf
        ]


# ---------------------------------------------------------------------------
# Bundled loader (always available, no internet required)
# ---------------------------------------------------------------------------

@DatasetRegistry.register
class BundledLoader(DatasetLoader):
    """
    Bundled dataset — benign_prompts.jsonl + attack_prompts.jsonl.
    No internet required. Used for --quick mode and CI.
    """
    name = "bundled"
    requires_hf = False
    cl_safe = True  # Manually curated, all labels verified — ground-truth FPR reference

    def load(self, sample_limit: int = 0) -> List[BenchmarkSample]:
        samples: List[BenchmarkSample] = []

        benign_path = BUNDLED_DIR / "benign_prompts.jsonl"
        attack_path = BUNDLED_DIR / "attack_prompts.jsonl"

        for path, label in [(benign_path, 0), (attack_path, 1)]:
            if not path.exists():
                logger.warning("Bundled file not found: %s", path)
                continue
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        samples.append(BenchmarkSample(
                            text=row["text"],
                            label=label,
                            category=row.get("category"),
                            source="bundled",
                            metadata=row.get("metadata", {}),
                        ))
                    except (json.JSONDecodeError, KeyError) as exc:
                        logger.debug("Skipping malformed bundled line: %s", exc)

        logger.info("BundledLoader: loaded %d samples", len(samples))
        return self._enforce_limit(samples, sample_limit)


@DatasetRegistry.register
class AgenticBundledLoader(DatasetLoader):
    """
    Agentic-AI bundled dataset — agentic_attack_prompts.jsonl + agentic_benign_prompts.jsonl.

    Covers the surfaces the classic bundled corpus does NOT touch: tool-call args
    (path traversal, argument smuggling, SSRF), retrieval ranking poisoning,
    cross-user context bleeding, support-bot impersonation, carrier-framed evasion
    (proof point for sub-span semantic matching), multi-turn crescendo, execution-
    plan injection. Mirrors the Gaps 77–83 threat library and the FP-stress
    hard-negatives corpus.

    Offline, no internet required.
    """
    name = "agentic"
    requires_hf = False
    cl_safe = True  # Manually curated; all labels verified

    def load(self, sample_limit: int = 0) -> List[BenchmarkSample]:
        samples: List[BenchmarkSample] = []

        benign_path = BUNDLED_DIR / "agentic_benign_prompts.jsonl"
        attack_path = BUNDLED_DIR / "agentic_attack_prompts.jsonl"

        for path, label in [(benign_path, 0), (attack_path, 1)]:
            if not path.exists():
                logger.warning("Agentic bundled file not found: %s", path)
                continue
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        samples.append(BenchmarkSample(
                            text=row["text"],
                            label=label,
                            category=row.get("category"),
                            source="agentic",
                            metadata=row.get("metadata", {}),
                        ))
                    except (json.JSONDecodeError, KeyError) as exc:
                        logger.debug("Skipping malformed agentic line: %s", exc)

        logger.info("AgenticBundledLoader: loaded %d samples", len(samples))
        return self._enforce_limit(samples, sample_limit)


# ---------------------------------------------------------------------------
# HuggingFace-backed loaders
# ---------------------------------------------------------------------------

def _require_datasets():
    """
    Import the HuggingFace `datasets` package.

    IMPORTANT: our `benchmarks/datasets/` directory shadows the HuggingFace
    package when `benchmarks/` is on sys.path (which it is when running the
    benchmark via `python benchmarks/run_benchmark.py`).  We work around this
    by temporarily removing the local package paths from sys.path so that
    Python finds the real HuggingFace installation in site-packages.

    PERFORMANCE / STABILITY NOTE:
    After the first successful import we cache the HF module in sys.modules.
    Subsequent calls return the cached module immediately without any sys.path
    manipulation or re-import.  This is critical on Windows: re-importing the
    HuggingFace datasets module reinitialises its internal threading.RLock
    objects; dill (used by HF for fingerprint hashing) then fails to serialise
    those locks with:
        "RLock objects should only be shared between processes through inheritance"
    By caching the import we avoid this entirely from the second call onward.
    """
    import importlib
    import os
    import sys  # noqa: PLC0415

    # Fast path: if we already have the real HF datasets module cached, use it.
    # This avoids re-import (and the associated RLock reinitialisation) on
    # every call after the first successful load.
    existing = sys.modules.get("datasets")
    if existing is not None and hasattr(existing, "load_dataset"):
        return existing

    # Absolute paths that shadow the real HuggingFace 'datasets' package
    # when benchmarks/ is on sys.path (either as an absolute path or relative).
    shadow_abs = {
        os.path.abspath(str(Path(__file__).parent)),         # benchmarks/datasets/
        os.path.abspath(str(Path(__file__).parent.parent)),  # benchmarks/
        os.getcwd(),  # '' (empty string) resolves to CWD
    }

    def _abs(p: str) -> str:
        """Normalize a sys.path entry to an absolute path."""
        return os.path.abspath(p) if p else os.getcwd()

    # Save then temporarily remove conflicting paths.
    removed: list = []
    for i in range(len(sys.path) - 1, -1, -1):
        if _abs(sys.path[i]) in shadow_abs:
            removed.append((i, sys.path.pop(i)))

    # Remove the local 'datasets' package if it was accidentally cached
    # (only needed on the very first call; fast path handles subsequent calls).
    sys.modules.pop("datasets", None)

    try:
        hf_datasets = importlib.import_module("datasets")
        if not hasattr(hf_datasets, "load_dataset"):
            raise ImportError(
                "Imported 'datasets' has no load_dataset — "
                "still resolving to local package despite path fix."
            )
        return hf_datasets
    except ImportError:
        raise ImportError(
            "HuggingFace 'datasets' package is required for this loader.\n"
            "Install it with:  pip install 'ethicore-guardian-benchmarks[hf]'"
        ) from None
    finally:
        # Restore sys.path in original order.
        for i, p in sorted(removed, key=lambda x: x[0]):
            sys.path.insert(i, p)
        # Keep the HuggingFace module cached in sys.modules['datasets'] so the
        # fast-path check above works on all subsequent calls.  The local
        # benchmarks.datasets package is never imported as bare 'datasets'.


@DatasetRegistry.register
class NeuralchemyLoader(DatasetLoader):
    """
    neuralchemy/Prompt-injection-dataset (HuggingFace)
    16,918 samples — binary labeled (injection / benign).

    ⚠️ DATASET QUALITY CAVEAT:
    Manual inspection reveals that the 'benign' class contains a significant number
    of mislabeled samples — including explicit phishing email generation requests,
    drug synthesis prompts, and cybersecurity exploitation scenarios that are labeled
    as benign. As a result, FPR computed on this dataset is NOT a reliable measure of
    false positive rate; it reflects Guardian correctly rejecting actually-harmful content
    that the dataset incorrectly labeled as benign.

    The bundled dataset (curated in-house) is the ground-truth FPR reference.
    Neuralchemy results remain useful for recall / ROC-AUC ranking tasks since
    the attack class (injection / jailbreak) is well-labeled.

    cl_safe=False — Mislabeled benign class means any FP Guardian raises is likely a
    correct detection; submitting these to AdversarialLearner would teach wrong patterns.
    """
    name = "neuralchemy"
    requires_hf = True
    cl_safe = False  # Mislabeled benign class — FPR unreliable, do not use for AL training

    def load(self, sample_limit: int = 0) -> List[BenchmarkSample]:
        hf = _require_datasets()
        logger.info("NeuralchemyLoader: downloading from HuggingFace…")
        logger.warning(
            "neuralchemy: FPR on this dataset is unreliable due to mislabeled benign samples "
            "(confirmed harmful content labeled as benign). Use bundled dataset for FPR ground truth."
        )
        ds = hf.load_dataset("neuralchemy/Prompt-injection-dataset", split="train")

        samples = []
        for row in ds:
            # Dataset columns: 'text', 'label' (1=injection, 0=benign)
            text = row.get("text") or row.get("prompt") or ""
            raw_label = row.get("label", row.get("injected", 0))
            label = int(bool(raw_label))
            if text:
                samples.append(BenchmarkSample(
                    text=text,
                    label=label,
                    source="neuralchemy",
                ))

        logger.info("NeuralchemyLoader: %d samples loaded", len(samples))
        return self._enforce_limit(samples, sample_limit)


@DatasetRegistry.register
class DeepsetLoader(DatasetLoader):
    """
    deepset/prompt-injections (HuggingFace)
    ~1,000 multi-domain samples — binary labeled (injection / benign).

    cl_safe=True — Well-curated, manually labeled prompt injection examples.
    Note: in 2-pass benchmarks the AdversarialLearner will learn missed injection
    patterns from this dataset. Expect a moderate FPR increase on deepset in Pass 2
    (learned fingerprints may generalize to some benign samples) — this is expected
    AL behavior and reflects genuine learning, not CL contamination.
    """
    name = "deepset"
    requires_hf = True
    cl_safe = True  # Well-curated injection dataset with verified benign labels

    def load(self, sample_limit: int = 0) -> List[BenchmarkSample]:
        hf = _require_datasets()
        logger.info("DeepsetLoader: downloading from HuggingFace…")
        ds = hf.load_dataset("deepset/prompt-injections", split="train")

        samples = []
        for row in ds:
            text = row.get("text") or row.get("prompt") or ""
            raw_label = row.get("label", 0)
            label = int(bool(raw_label))
            if text:
                samples.append(BenchmarkSample(
                    text=text,
                    label=label,
                    source="deepset",
                ))

        logger.info("DeepsetLoader: %d samples loaded", len(samples))
        return self._enforce_limit(samples, sample_limit)


@DatasetRegistry.register
class JailbreakBenchLoader(DatasetLoader):
    """
    JailbreakBench/JBB-Behaviors (HuggingFace)
    100 standardized jailbreak behaviors — NeurIPS 2024.
    All samples are attacks (label=1). Paired with bundled benigns for FPR eval.

    Config note: the dataset requires config='behaviors' (not the default).

    cl_safe=False — Attack-only; behaviors are abstract harmful goals ("write instructions
    for X") rather than actual injection prompt text. Semantics differ from Guardian's
    injection/jailbreak detection space. Learning these FNs would pollute AL fingerprints
    and raise FPR on unrelated benign content.
    """
    name = "jailbreakbench"
    requires_hf = True
    cl_safe = False  # Attack-only; abstract behaviors not in injection-prompt format

    def load(self, sample_limit: int = 0) -> List[BenchmarkSample]:
        hf = _require_datasets()
        logger.info("JailbreakBenchLoader: downloading from HuggingFace…")
        # Must specify config='behaviors'; splits available: 'harmful' (attacks) and 'benign'.
        # We load 'harmful' for the attack behaviors; supplemented with bundled benigns at run time.
        ds = hf.load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")

        samples = []
        for row in ds:
            # Columns: Goal, Target, Category, Behavior
            text = row.get("Goal") or row.get("Behavior") or ""
            category = row.get("Category")
            if text:
                samples.append(BenchmarkSample(
                    text=text,
                    label=1,  # all JBB samples are attacks
                    category=category,
                    source="jailbreakbench",
                    metadata={"target": row.get("Target", "")},
                ))

        logger.info("JailbreakBenchLoader: %d samples loaded", len(samples))
        return self._enforce_limit(samples, sample_limit)


@DatasetRegistry.register
class AdvBenchLoader(DatasetLoader):
    """
    walledai/AdvBench (HuggingFace)
    500 harmful behaviors — cybercrime, chemical/biological, misinfo, illegal activity.
    All samples are attacks (label=1).

    ⚠️ ACCESS NOTE: This is a gated dataset on HuggingFace Hub. You must request
    access at https://huggingface.co/datasets/walledai/AdvBench and authenticate
    with `huggingface-cli login` before this loader will work.

    cl_safe=False — Gated; attack-only harmful behaviors not in injection format.
    """
    name = "advbench"
    requires_hf = True
    cl_safe = False  # Gated; harmful behaviors, not injection prompts

    def load(self, sample_limit: int = 0) -> List[BenchmarkSample]:
        hf = _require_datasets()
        logger.info("AdvBenchLoader: downloading from HuggingFace…")
        logger.info(
            "advbench: gated dataset — ensure HF access is granted and you are authenticated "
            "(huggingface-cli login). Visit https://huggingface.co/datasets/walledai/AdvBench"
        )
        ds = hf.load_dataset("walledai/AdvBench", split="train")

        samples = []
        for row in ds:
            text = row.get("goal") or row.get("prompt") or row.get("behavior") or ""
            if text:
                samples.append(BenchmarkSample(
                    text=text,
                    label=1,  # all AdvBench samples are attacks
                    source="advbench",
                    metadata={"target": row.get("target", "")},
                ))

        logger.info("AdvBenchLoader: %d samples loaded", len(samples))
        return self._enforce_limit(samples, sample_limit)


@DatasetRegistry.register
class HarmBenchLoader(DatasetLoader):
    """
    walledai/HarmBench (HuggingFace) — mirror of centerforaisafety/HarmBench.
    400+ harmful behaviors across semantic harm domains (cybercrime, violence,
    bioweapons, misinformation, etc.). All samples are attacks (label=1).
    Supplemented with bundled benigns for FPR evaluation.

    Note: The original centerforaisafety/harmbench HF dataset was unavailable
    as of 2026-03-20; this loader uses the walledai mirror which is publicly
    accessible. Falls back to the GitHub CSV via direct HTTP if HF is unavailable.

    cl_safe=False — Attack-only harmful behaviors; semantics differ from prompt
    injection text. Learning these FNs would teach AL patterns that generalize
    too broadly and raise FPR on unrelated content.
    """
    name = "harmbench"
    requires_hf = True
    cl_safe = False  # Harmful behaviors, not injection prompts; attack-only

    def load(self, sample_limit: int = 0) -> List[BenchmarkSample]:
        hf = _require_datasets()
        logger.info("HarmBenchLoader: downloading from HuggingFace (walledai/HarmBench)…")

        try:
            ds = hf.load_dataset("walledai/HarmBench", split="train")
        except Exception:
            # Secondary fallback: original centerforaisafety path (may be unavailable)
            logger.warning("walledai/HarmBench unavailable, trying centerforaisafety/harmbench…")
            try:
                ds = hf.load_dataset("centerforaisafety/harmbench", "standard", split="train")
            except Exception:
                ds = hf.load_dataset("centerforaisafety/harmbench", split="train")

        samples = []
        for row in ds:
            text = (
                row.get("Behavior") or row.get("behavior")
                or row.get("goal") or row.get("prompt") or ""
            )
            category = row.get("SemanticCategory") or row.get("FunctionalCategory")
            if text:
                samples.append(BenchmarkSample(
                    text=text,
                    label=1,  # all HarmBench samples are attacks
                    category=category,
                    source="harmbench",
                ))

        logger.info("HarmBenchLoader: %d samples loaded", len(samples))
        return self._enforce_limit(samples, sample_limit)


@DatasetRegistry.register
class B3Loader(DatasetLoader):
    """
    Lakera/b3-agent-security-benchmark-weak (HuggingFace)
    B3 — Backbone Breaker Benchmark (Lakera, 2024).
    210 crowdsourced adversarial attacks against AI-agent backbone LLMs,
    evaluated across 3 defense levels (L1/L2/L3) for 630 total entries.
    All samples are attacks (label=1). Supplemented with bundled benigns for FPR eval.

    Attacks are contextual agentic security attacks — prompt injections and
    jailbreaks targeting LLMs embedded in agent applications (Gandalf challenge).
    Directly relevant to Guardian's agentic threat coverage (SkillPhishing,
    IndirectInjection, ToolCallManipulation, etc.).

    cl_safe=False — Attack-only dataset. The attack texts are agentic in nature
    and semantically different from the canonical injection prompts Guardian is
    trained on. Feeding FNs into the AdversarialLearner risks teaching fingerprints
    that are too context-specific and would raise FPR on normal agent interactions.
    """
    name = "b3"
    requires_hf = True
    cl_safe = False  # Attack-only agentic security attacks; different semantic space

    def load(self, sample_limit: int = 0) -> List[BenchmarkSample]:
        hf = _require_datasets()
        logger.info("B3Loader: downloading Lakera/b3-agent-security-benchmark-weak…")
        ds = hf.load_dataset("Lakera/b3-agent-security-benchmark-weak", split="test")

        samples = []
        for row in ds:
            # Dataset columns vary; try common attack text fields in priority order
            text = (
                row.get("attack") or row.get("input") or row.get("prompt")
                or row.get("user_input") or row.get("text") or row.get("message") or ""
            )
            category = row.get("application") or row.get("threat_type") or row.get("category")
            level = row.get("level") or row.get("defense_level")
            if text:
                samples.append(BenchmarkSample(
                    text=text,
                    label=1,  # all B3 samples are attacks
                    category=str(category) if category else None,
                    source="b3",
                    metadata={"level": level} if level is not None else {},
                ))

        logger.info("B3Loader: %d samples loaded", len(samples))
        return self._enforce_limit(samples, sample_limit)


@DatasetRegistry.register
class SaladDataLoader(DatasetLoader):
    """
    OpenSafetyLab/Salad-Data (HuggingFace) — base_set subset.
    SALAD-Bench: A Hierarchical and Comprehensive Safety Benchmark for LLMs (ACL 2024).
    21,335 harmful questions classified into 6 domains, 16 tasks, and 66 categories.
    All samples are harmful questions (label=1). Supplemented with bundled benigns for FPR eval.

    This is a broad LLM safety benchmark covering topics such as violence, cybercrime,
    hate speech, and self-harm — NOT specifically prompt injection or jailbreaking.
    Useful for measuring Guardian's coverage of harmful content beyond injection attacks.

    cl_safe=False — Attack-only (all questions are harmful); the threat surface is
    far broader than prompt injection. Learning FNs from this dataset would teach the
    AdversarialLearner semantics from drug synthesis / violence / hate speech content,
    which would pollute injection-specific fingerprints and cause significant FPR increases
    on legitimate user queries touching similar topics.
    """
    name = "salad_data"
    requires_hf = True
    cl_safe = False  # Broad safety questions (not injection-specific); attack-only

    def load(self, sample_limit: int = 0) -> List[BenchmarkSample]:
        hf = _require_datasets()
        logger.info("SaladDataLoader: downloading OpenSafetyLab/Salad-Data (base_set)…")

        try:
            ds = hf.load_dataset("OpenSafetyLab/Salad-Data", "base_set", split="train")
        except Exception:
            # Fallback: try without subset name
            logger.warning("base_set config unavailable, trying default split…")
            ds = hf.load_dataset("OpenSafetyLab/Salad-Data", split="train")

        samples = []
        for row in ds:
            text = row.get("question") or row.get("prompt") or row.get("input") or ""
            domain = row.get("1-category") or row.get("domain") or row.get("category")
            task = row.get("2-category") or row.get("task")
            specific = row.get("3-category") or row.get("specific_category")
            # Build a combined category label from the hierarchy
            category_parts = [p for p in [domain, task, specific] if p]
            category = " / ".join(category_parts) if category_parts else None

            if text:
                samples.append(BenchmarkSample(
                    text=text,
                    label=1,  # all Salad-Data questions are harmful
                    category=category,
                    source="salad_data",
                ))

        logger.info("SaladDataLoader: %d samples loaded", len(samples))
        return self._enforce_limit(samples, sample_limit)


def _try_load_hf(hf, ids: List[str], split: str = "test"):
    """Try multiple HF dataset IDs in order; return first that succeeds.

    Public agentic-AI benchmarks frequently move between mirrors / get re-uploaded
    under different org names. This wrapper tries the most-stable known ID first
    and falls back to alternates, so a single broken/renamed source doesn't kill
    the loader. Returns the loaded dataset or raises the last exception.
    """
    last_exc: Optional[Exception] = None
    for hid in ids:
        try:
            return hf.load_dataset(hid, split=split)
        except Exception as exc:  # noqa: BLE001 — try-all-and-fall-back
            last_exc = exc
            logger.debug("HF load failed for %s: %s", hid, exc)
    if last_exc:
        raise last_exc
    raise RuntimeError("no HF ids provided to _try_load_hf")


@DatasetRegistry.register
class AgentDojoLoader(DatasetLoader):
    """
    AgentDojo (Debenedetti et al., NeurIPS 2024) — ETH Zurich / Princeton.
    https://arxiv.org/abs/2406.13352

    Dynamic agent-injection benchmark: AI assistants doing realistic tasks
    (workspace, banking, Slack, travel) face injected instructions embedded in
    tool outputs. The injections target the AGENT'S TOOL-USE — not the user-
    facing prompt — so they specifically exercise the tool-call / indirect-
    injection surface Guardian covers via Layer 13 (ToolCallValidator),
    indirectInjection patterns, and the new Gap 79/81 SSRF + retrieval coverage.

    All attack-side samples are injection strings (label=1); supplemented with
    bundled benigns for FPR eval.

    cl_safe=False — attack-only, agent-context-specific. Learning FNs would
    teach context-specific fingerprints that risk FPR on legitimate agent tasks.
    """
    name = "agentdojo"
    requires_hf = True
    cl_safe = False

    def load(self, sample_limit: int = 0) -> List[BenchmarkSample]:
        hf = _require_datasets()
        logger.info("AgentDojoLoader: downloading AgentDojo injections…")
        # Order: most-canonical first, fallbacks for renames/mirrors
        candidate_ids = [
            "ethz-spylab/agentdojo",
            "anonymous-agent-research/agentdojo",
            "JailbreakBench/AgentDojo",
        ]
        ds = _try_load_hf(hf, candidate_ids, split="test")

        samples = []
        for row in ds:
            # AgentDojo schemas: injection task strings live in 'goal' or 'injection'
            text = (
                row.get("injection") or row.get("goal") or row.get("attack")
                or row.get("prompt") or row.get("text") or ""
            )
            category = row.get("suite") or row.get("task_suite") or row.get("category")
            if text:
                samples.append(BenchmarkSample(
                    text=text,
                    label=1,
                    category=str(category) if category else None,
                    source="agentdojo",
                ))

        logger.info("AgentDojoLoader: %d samples loaded", len(samples))
        return self._enforce_limit(samples, sample_limit)


@DatasetRegistry.register
class InjecAgentLoader(DatasetLoader):
    """
    InjecAgent (Zhan et al., ACL 2024) — UIUC / Tsinghua / Stanford.
    https://arxiv.org/abs/2403.02691

    Test cases for indirect prompt injection against tool-augmented LLM agents.
    The user instructs the agent to use a tool; the tool RESPONSE contains an
    injected attacker instruction (data theft, harmful action). Direct exercise
    of the indirect-injection surface (Guardian Layer 8 ToolOutputScanner +
    ragPoisoning + crossSourceEvidenceManipulation categories).

    1,054 test cases (~50/50 data-stealing vs harmful-action attack types).
    All samples are attacks (label=1).

    cl_safe=False — attack-only, indirect-injection context.
    """
    name = "injecagent"
    requires_hf = True
    cl_safe = False

    def load(self, sample_limit: int = 0) -> List[BenchmarkSample]:
        hf = _require_datasets()
        logger.info("InjecAgentLoader: downloading InjecAgent…")
        candidate_ids = [
            "uiuc-kang-lab/InjecAgent",
            "RoyZry98/InjecAgent",
            "InjecAgent/InjecAgent",
        ]
        ds = _try_load_hf(hf, candidate_ids, split="test")

        samples = []
        for row in ds:
            # InjecAgent fields: 'Attacker Instruction', 'Tool Response Template',
            # or fall back to generic fields
            text = (
                row.get("Attacker Instruction") or row.get("attacker_instruction")
                or row.get("attack") or row.get("instruction") or row.get("text")
                or row.get("prompt") or ""
            )
            category = (
                row.get("Attack Type") or row.get("attack_type")
                or row.get("category") or "indirect_injection"
            )
            if text:
                samples.append(BenchmarkSample(
                    text=text,
                    label=1,
                    category=str(category) if category else None,
                    source="injecagent",
                ))

        logger.info("InjecAgentLoader: %d samples loaded", len(samples))
        return self._enforce_limit(samples, sample_limit)


@DatasetRegistry.register
class AgentHarmLoader(DatasetLoader):
    """
    AgentHarm (Andriushchenko et al., 2024) — UK AI Safety Institute / Anthropic.
    https://arxiv.org/abs/2410.09024

    110+ explicitly harmful agent tasks across 11 categories (fraud, cybercrime,
    harassment, etc.) measuring whether a tool-using LLM agent will EXECUTE a
    harmful tool sequence. Tests refusal-rate + harmful-tool-use rate — directly
    relevant to Guardian's Layer 17 AgenticExecutionMonitor + tool-call validation.

    NOTE: This dataset may be GATED on HuggingFace. If load fails with a 403,
    visit the dataset page and accept the terms (free, requires HF account login):
        https://huggingface.co/datasets/ai-safety-institute/AgentHarm
    Then `huggingface-cli login` before running the benchmark.

    cl_safe=False — attack-only, harmful-agent-task context.
    """
    name = "agentharm"
    requires_hf = True
    cl_safe = False

    def load(self, sample_limit: int = 0) -> List[BenchmarkSample]:
        hf = _require_datasets()
        logger.info("AgentHarmLoader: downloading AgentHarm (gated — may need HF login)…")
        candidate_ids = [
            "ai-safety-institute/AgentHarm",
            "AISI/AgentHarm",
        ]
        try:
            ds = _try_load_hf(hf, candidate_ids, split="test_public")
        except Exception:
            # Fallback split name
            ds = _try_load_hf(hf, candidate_ids, split="test")

        samples = []
        for row in ds:
            # AgentHarm columns: 'prompt' is the harmful task, 'category' is one
            # of the 11 harm categories
            text = (
                row.get("prompt") or row.get("task") or row.get("goal")
                or row.get("text") or row.get("instruction") or ""
            )
            category = row.get("category") or row.get("harm_category") or "harmful_agent_task"
            if text:
                samples.append(BenchmarkSample(
                    text=text,
                    label=1,
                    category=str(category) if category else None,
                    source="agentharm",
                ))

        logger.info("AgentHarmLoader: %d samples loaded", len(samples))
        return self._enforce_limit(samples, sample_limit)
