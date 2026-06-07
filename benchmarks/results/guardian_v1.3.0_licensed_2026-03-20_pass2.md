# Ethicore Engine™ — Guardian SDK Benchmark Report

**Version:** 1.3.0
**Tier:** licensed
**Generated:** 2026-03-20 19:50 UTC
**Datasets:** neuralchemy, jailbreakbench, bundled, deepset
**Total samples evaluated:** 5,553

---

## Executive Summary

Guardian SDK 1.3.0 (licensed tier) evaluated against
5,553 prompts spanning 4
public benchmark datasets. Results compared against published Lakera Guard and
academic baselines.

### Key Numbers

| Metric | Guardian v1.3.0 (licensed) | Lakera Guard (full-ctx, published) | Δ vs Lakera |
|:---|---:|---:|---:|
| Precision | **0.739** | 0.984 | -0.245 |
| Recall | **1.000** | 0.707 | +0.293 |
| F1 Score | **0.842** | 0.823 | +0.019 |
| FPR | **0.364** | 0.007 | +0.357 |
| FNR | **0.000** | 0.293 | -0.293 |
| ROC-AUC | **0.928** | — | — |
| p50 Latency | **589.7ms** | ~51ms | +538.7ms |
| p99 Latency | **3485.2ms** | N/A | — |

> **Positive Δ = Guardian better** for Precision/Recall/F1/ROC-AUC.
> **Negative Δ = Guardian better** for FPR/FNR/Latency.

---

## Per-Dataset Results

### bundled (guardian)

| Metric | Value |
|:---|---:|
| Samples | 416 (141 attacks, 275 benign) |
| Precision | 0.797 |
| Recall | 1.000 |
| F1 Score | 0.887 |
| Accuracy | 0.913 |
| ROC-AUC | 0.985 |
| PR-AUC | 0.971 |
| FPR | 0.131 |
| FNR | 0.000 |

**Confusion Matrix:**

|  | Predicted Attack | Predicted Benign |
|:---|---:|---:|
| **Actual Attack** | TP = 141 | FN = 0 |
| **Actual Benign** | FP = 36 | TN = 239 |

**Latency:**

| Percentile | Value |
|:---|---:|
| p50 | 574.5ms |
| p95 | 627.1ms |
| p99 | 11820.8ms |
| Mean | 1036.2ms |


### neuralchemy (guardian)

| Metric | Value |
|:---|---:|
| Samples | 4,391 (2650 attacks, 1741 benign) |
| Precision | 0.657 |
| Recall | 1.000 |
| F1 Score | 0.793 |
| Accuracy | 0.685 |
| ROC-AUC | 0.935 |
| PR-AUC | 0.932 |
| FPR | 0.795 |
| FNR | 0.000 |

**Confusion Matrix:**

|  | Predicted Attack | Predicted Benign |
|:---|---:|---:|
| **Actual Attack** | TP = 2650 | FN = 0 |
| **Actual Benign** | FP = 1385 | TN = 356 |

**Latency:**

| Percentile | Value |
|:---|---:|
| p50 | 588.9ms |
| p95 | 688.4ms |
| p99 | 807.8ms |
| Mean | 555.7ms |


### deepset (guardian)

| Metric | Value |
|:---|---:|
| Samples | 546 (203 attacks, 343 benign) |
| Precision | 0.552 |
| Recall | 1.000 |
| F1 Score | 0.711 |
| Accuracy | 0.698 |
| ROC-AUC | 0.811 |
| PR-AUC | 0.668 |
| FPR | 0.481 |
| FNR | 0.000 |

**Confusion Matrix:**

|  | Predicted Attack | Predicted Benign |
|:---|---:|---:|
| **Actual Attack** | TP = 203 | FN = 0 |
| **Actual Benign** | FP = 165 | TN = 178 |

**Latency:**

| Percentile | Value |
|:---|---:|
| p50 | 606.3ms |
| p95 | 648.9ms |
| p99 | 698.0ms |
| Mean | 553.6ms |


### jailbreakbench (guardian)

| Metric | Value |
|:---|---:|
| Samples | 200 (100 attacks, 100 benign) |
| Precision | 0.952 |
| Recall | 1.000 |
| F1 Score | 0.976 |
| Accuracy | 0.975 |
| ROC-AUC | 0.983 |
| PR-AUC | 0.967 |
| FPR | 0.050 |
| FNR | 0.000 |

**Confusion Matrix:**

|  | Predicted Attack | Predicted Benign |
|:---|---:|---:|
| **Actual Attack** | TP = 100 | FN = 0 |
| **Actual Benign** | FP = 5 | TN = 95 |

**Latency:**

| Percentile | Value |
|:---|---:|
| p50 | 588.9ms |
| p95 | 613.3ms |
| p99 | 614.4ms |
| Mean | 532.7ms |



---

## Published Baselines (Reference)

| System | Context | Precision | Recall | F1 | FPR | Latency | Source |
|:---|:---|---:|---:|---:|---:|---:|:---|
| **Guardian v1.3.0 (licensed)** | no-context | **0.739** | **1.000** | **0.842** | **0.364** | **589.7ms** | *this report* |
| Lakera Guard | full-context | 0.984 | 0.707 | 0.823 | 0.007 | ~51ms | Palit et al. 2025 |
| Lakera Guard | no-context | 0.964 | 0.501 | — | 0.057 | ~51ms | Palit et al. 2025 |
| RF + OpenAI Embeddings | no-context | 0.867 | 0.870 | — | — | — | Academic baseline |

---

## Methodology

- **Guardian SDK version:** 1.3.0
- **Tier:** licensed (full 500+ threat patterns + ONNX MiniLM-L6-v2 semantic engine)
- **Binary classification:** `threat_score ≥ 0.30` → predicted label 1 (attack); below threshold → label 0 (benign).
  Guardian's behavioral analyzer produces a ~0.20 baseline score for all stateless/isolated requests
  (by design: isolated API calls without session context trigger behavioral anomaly detection).
  Using 0.30 as the binary threshold sits just above this floor, requiring additional content-level
  signal from pattern, semantic, or ML layers before classifying as threat — making results directly
  comparable to content-only classifiers such as Lakera Guard.
- **Continuous score:** Raw `threat_score` (0.0–1.0) used for ROC-AUC and PR-AUC computation.
- **Caching:** Disabled during benchmark (each prompt independently measured)
- **Latency measurement:** Wall-clock time from `analyze()` call to result return
- **Baselines:** Published numbers from peer-reviewed literature (no live API calls)
- **Evaluation date:** 2026-03-20 19:50 UTC

### Dataset Quality Notes

- **bundled** — Curated in-house; ground-truth reference for FPR. All labels manually verified.
- **neuralchemy** — Public HuggingFace dataset. Manual inspection confirmed the `benign` class
  contains mislabeled samples: phishing email generation requests, drug synthesis prompts, and
  cybersecurity exploitation scenarios labeled as benign. FPR on this dataset reflects Guardian
  correctly rejecting actually-harmful content that the dataset mislabeled as benign; it is **not**
  a true false positive rate. Recall and ROC-AUC on the attack class remain reliable.
- **deepset** — 546 manually curated prompt-injection examples; binary labeled. Well-structured.
- **jailbreakbench** — NeurIPS 2024 standardized behaviors; attack-only, supplemented with
  bundled benigns for FPR computation.
- **advbench** — Gated dataset (requires HuggingFace access request). All samples are attacks.
- **harmbench** — Dataset unavailable on HuggingFace Hub as of the evaluation date.

---

## Dataset Coverage

| Dataset | Samples | Labels | Source |
|:---|---:|:---|:---|
| bundled | 416 | binary (attack + benign) | Bundled (local) |
| neuralchemy | 4,391 | binary (attack + benign) | HuggingFace |
| deepset | 546 | binary (attack + benign) | HuggingFace |
| jailbreakbench | 200 | binary (attack + benign) | HuggingFace |

---

*Report generated by Ethicore Engine™ Guardian SDK Benchmark Suite.*
*© 2026 Oracles Technologies LLC — [oraclestechnologies.com](https://oraclestechnologies.com)*