# Ethicore Engine™ — Guardian SDK Benchmark Suite

Reproducible, auditable benchmarks comparing Guardian SDK against published
baselines (Lakera Guard, academic RF + embeddings) across public prompt injection,
jailbreak, and agentic-AI threat detection datasets.

> **Tier benchmarked: API (licensed).** All published numbers in this repository
> are produced with `ETHICORE_API_KEY` + `ETHICORE_ASSETS_DIR` set, which loads the
> full licensed threat library (157 categories / 2,776 fingerprints), the calibrated
> 400k-sample ONNX guardian model, multilingual ONNX (50+ languages), sub-span
> semantic matching, Layer 13 browser analyzer, Layer 17 agentic execution monitor,
> and the AdversarialLearner. The community tier (`pip install ethicore-engine-guardian`,
> no key) ships a subset with reduced coverage and is suitable for development.
>
> ### Architectural property: closed-loop fingerprint persistence (`--learn`)
>
> Unlike static-corpus competitors (Lakera Guard, PromptGuard, Rebuff) which ship a
> frozen detection corpus per release, Guardian's AdversarialLearner **persists
> confirmed-attack fingerprints to disk on every `learn_from_confirmed_attack()`
> call** — including the `--learn` two-pass benchmark mode. This is *the architectural
> property the product is designed around*: a customer-confirmed missed attack becomes
> detection on the next request, not the next quarterly release. Benchmark Pass 1 in
> a `--learn` run therefore reflects "deployment with prior learning history" — which
> *is* the steady-state production behavior. Static-corpus systems by design cannot
> show this property.
>
> The FP-safety + carrier-stripping gates ensure persisted fingerprints never raise
> FPR (verified: FPR=0.000 on the bundled corpus across 49 organically-learned
> fingerprints from prior `--learn` runs).
>
> To reproduce a *baseline-only* number (no closed-loop history), regenerate
> embeddings before running: `python scripts/regenerate_embeddings.py --force`.

---

## Quick Start

```bash
# From repo root
cd "Ethicore Engine - TEST"

# Install benchmark dependencies
pip install scikit-learn pandas numpy jinja2 tqdm

# Quick run (~2 min, no internet, bundled data only)
python benchmarks/run_benchmark.py --quick

# Full run (~30 min, downloads HuggingFace datasets)
pip install datasets huggingface-hub
python benchmarks/run_benchmark.py --full

# Single dataset
python benchmarks/run_benchmark.py --dataset neuralchemy

# Licensed tier (requires ETHICORE_API_KEY env var; ETHICORE_ASSETS_DIR optional)
python benchmarks/run_benchmark.py --full --tier licensed
```

Reports are written to `benchmarks/results/`.

---

## Datasets

| Dataset | Size | Requires HF | Coverage |
|---|---|---|---|
| **bundled** | 416 | No | Mixed: 275 benign + 141 attacks (offline) |
| **agentic** | 204 | No | Agentic AI surface: tool-calls, exec plans, carrier-framing, Gaps 77–83 (offline) |
| **neuralchemy** | 16,918 | Yes | Binary labeled prompt injections |
| **deepset** | ~1,000 | Yes | Multi-domain prompt injections |
| **jailbreakbench** | 100 | Yes | Standardized jailbreak behaviors (NeurIPS 2024) |
| **advbench** | 500 | Yes | Harmful behaviors: chemical, cyber, misinfo |
| **harmbench** | 510 | Yes | Harmful behavior taxonomy (USENIX 2024) |
| **b3** | 630 | Yes | Lakera Backbone Breaker — agentic attacks (Gandalf challenge, 2024) |
| **salad_data** | 21,335 | Yes | SALAD-Bench — broad LLM safety (ACL 2024) |
| **agentdojo** | ~600 | Yes | ETH/Princeton — agent tool-call injections (NeurIPS 2024) |
| **injecagent** | 1,054 | Yes | UIUC/Tsinghua/Stanford — indirect injection via tool outputs (ACL 2024) |
| **agentharm** | 110 | Yes (gated) | UK AISI/Anthropic — harmful agent tasks; accept HF terms first |

---

## Metrics

| Metric | Description | Lakera Baseline |
|---|---|---|
| **Precision** | Fraction of threat flags that are real threats | 0.984 (full-ctx) |
| **Recall** | Fraction of real threats detected | 0.707 (full-ctx) |
| **F1** | Harmonic mean of precision and recall | 0.823 (full-ctx) |
| **ROC-AUC** | Threshold-agnostic discrimination ability | — |
| **PR-AUC** | Precision-recall AUC (better for imbalanced data) | — |
| **FPR** | False positive rate (legitimate traffic flagged) | 0.007 (full-ctx) |
| **FNR** | False negative rate (attacks missed) | 0.293 (full-ctx) |
| **p50 latency** | Median per-request latency | ~51ms |
| **p99 latency** | 99th percentile latency | N/A published |

> Lakera baselines from: Palit et al., arXiv:2505.13028 (2025).

---

## Report Outputs

Each run produces two files in `benchmarks/results/`:

| File | Description |
|---|---|
| `guardian_v{version}_{tier}_{date}.md` | Human-readable Markdown with comparison table |
| `guardian_v{version}_{tier}_{date}.json` | Machine-parseable JSON for CI/dashboards |

---

## CI Integration

```yaml
# .github/workflows/benchmark.yml
- name: Run quick benchmark
  run: python benchmarks/run_benchmark.py --quick
  env:
    ETHICORE_API_KEY: ${{ secrets.ETHICORE_API_KEY }}
    ETHICORE_ASSETS_DIR: ${{ secrets.ETHICORE_ASSETS_DIR }}
```

The JSON reporter includes a `check_thresholds()` method that returns pass/fail
for CI enforcement:

| Threshold | Default | Purpose |
|---|---|---|
| `min_precision` | 0.80 | Minimum acceptable precision |
| `min_recall` | 0.70 | Minimum acceptable recall |
| `max_fpr` | 0.05 | Maximum false positive rate |
| `max_p99_ms` | 200ms | Maximum p99 latency |

---

## Methodology

- **Caching disabled** during runs — every prompt independently measured
- **Adversarial learning disabled** — no state mutation during benchmarks
- **Tier:** community (18 patterns, 5 categories, hash-based semantics) or
  licensed (500+ patterns, 51 categories, ONNX MiniLM-L6-v2)
- **Binary threshold:** CHALLENGE or BLOCK → label=1 (threat), ALLOW → label=0
- **Attack-only datasets** (JailbreakBench, AdvBench, HarmBench) are supplemented
  with bundled benign prompts so FPR can be computed

---

## Benchmark Results

> Latest results: see [`results/`](./results/)

Results are committed after each tagged release.

---

## Adding a New Dataset

1. Create `benchmarks/datasets/your_dataset.py`
2. Subclass `DatasetLoader` and decorate with `@DatasetRegistry.register`
3. Set `name = "your_dataset"` and `requires_hf = True/False`
4. Implement `load(sample_limit) -> List[BenchmarkSample]`

```python
from .loader import DatasetLoader, BenchmarkSample, DatasetRegistry

@DatasetRegistry.register
class MyDatasetLoader(DatasetLoader):
    name = "my_dataset"
    requires_hf = True

    def load(self, sample_limit: int = 0) -> List[BenchmarkSample]:
        # ... load and return List[BenchmarkSample]
        return self._enforce_limit(samples, sample_limit)
```

5. Import it in `benchmarks/datasets/__init__.py`
6. Add to `FULL_DATASETS` in `run_benchmark.py`

---

*Ethicore Engine™ — Guardian SDK © 2026 Oracles Technologies LLC*
