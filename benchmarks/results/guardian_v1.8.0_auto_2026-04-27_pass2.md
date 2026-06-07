# Ethicore Engine™ — Guardian SDK Benchmark Report

**Version:** 1.8.0
**Tier:** auto
**Generated:** 2026-04-27 23:04 UTC
**Datasets:** bundled, deepset, jailbreakbench, salad_data, neuralchemy
**Total samples evaluated:** 27,146

---

## Executive Summary

Guardian SDK 1.8.0 (auto tier) evaluated against
27,146 prompts spanning 5
public benchmark datasets. Results compared against published Lakera Guard and
academic baselines.

### Key Numbers

| Metric | Guardian v1.8.0 (auto) | Lakera Guard (full-ctx, published) | Δ vs Lakera |
|:---|---:|---:|---:|
| Precision | **0.000** | 0.984 | -0.984 |
| Recall | **0.000** | 0.707 | -0.707 |
| F1 Score | **0.000** | 0.823 | -0.823 |
| FPR | **0.000** | 0.007 | -0.007 |
| FNR | **1.000** | 0.293 | +0.707 |
| ROC-AUC | **0.742** | — | — |
| p50 Latency | **854.2ms** | ~51ms | +803.2ms |
| p99 Latency | **12990.3ms** | N/A | — |

> **Positive Δ = Guardian better** for Precision/Recall/F1/ROC-AUC.
> **Negative Δ = Guardian better** for FPR/FNR/Latency.

---

## Per-Dataset Results

### bundled (guardian)

| Metric | Value |
|:---|---:|
| Samples | 416 (141 attacks, 275 benign) |
| Precision | 0.000 |
| Recall | 0.000 |
| F1 Score | 0.000 |
| Accuracy | 0.661 |
| ROC-AUC | 1.000 |
| PR-AUC | 1.000 |
| FPR | 0.000 |
| FNR | 1.000 |

**Confusion Matrix:**

|  | Predicted Attack | Predicted Benign |
|:---|---:|---:|
| **Actual Attack** | TP = 0 | FN = 141 |
| **Actual Benign** | FP = 0 | TN = 275 |

**Latency:**

| Percentile | Value |
|:---|---:|
| p50 | 794.1ms |
| p95 | 904.8ms |
| p99 | 60915.2ms |
| Mean | 3607.5ms |


### neuralchemy (guardian)

| Metric | Value |
|:---|---:|
| Samples | 4,391 (2650 attacks, 1741 benign) |
| Precision | 0.000 |
| Recall | 0.000 |
| F1 Score | 0.000 |
| Accuracy | 0.397 |
| ROC-AUC | 0.698 |
| PR-AUC | 0.783 |
| FPR | 0.000 |
| FNR | 1.000 |

**Confusion Matrix:**

|  | Predicted Attack | Predicted Benign |
|:---|---:|---:|
| **Actual Attack** | TP = 0 | FN = 2650 |
| **Actual Benign** | FP = 0 | TN = 1741 |

**Latency:**

| Percentile | Value |
|:---|---:|
| p50 | 880.8ms |
| p95 | 1028.7ms |
| p99 | 1191.4ms |
| Mean | 827.6ms |


### deepset (guardian)

| Metric | Value |
|:---|---:|
| Samples | 546 (203 attacks, 343 benign) |
| Precision | 0.000 |
| Recall | 0.000 |
| F1 Score | 0.000 |
| Accuracy | 0.628 |
| ROC-AUC | 0.933 |
| PR-AUC | 0.859 |
| FPR | 0.000 |
| FNR | 1.000 |

**Confusion Matrix:**

|  | Predicted Attack | Predicted Benign |
|:---|---:|---:|
| **Actual Attack** | TP = 0 | FN = 203 |
| **Actual Benign** | FP = 0 | TN = 343 |

**Latency:**

| Percentile | Value |
|:---|---:|
| p50 | 877.3ms |
| p95 | 980.8ms |
| p99 | 1016.1ms |
| Mean | 805.0ms |


### jailbreakbench (guardian)

| Metric | Value |
|:---|---:|
| Samples | 200 (100 attacks, 100 benign) |
| Precision | 0.000 |
| Recall | 0.000 |
| F1 Score | 0.000 |
| Accuracy | 0.500 |
| ROC-AUC | 0.549 |
| PR-AUC | 0.542 |
| FPR | 0.000 |
| FNR | 1.000 |

**Confusion Matrix:**

|  | Predicted Attack | Predicted Benign |
|:---|---:|---:|
| **Actual Attack** | TP = 0 | FN = 100 |
| **Actual Benign** | FP = 0 | TN = 100 |

**Latency:**

| Percentile | Value |
|:---|---:|
| p50 | 857.1ms |
| p95 | 885.1ms |
| p99 | 886.7ms |
| Mean | 776.0ms |


### salad_data (guardian)

| Metric | Value |
|:---|---:|
| Samples | 21,593 (21318 attacks, 275 benign) |
| Precision | 0.000 |
| Recall | 0.000 |
| F1 Score | 0.000 |
| Accuracy | 0.013 |
| ROC-AUC | 0.532 |
| PR-AUC | 0.988 |
| FPR | 0.000 |
| FNR | 1.000 |

**Confusion Matrix:**

|  | Predicted Attack | Predicted Benign |
|:---|---:|---:|
| **Actual Attack** | TP = 0 | FN = 21318 |
| **Actual Benign** | FP = 0 | TN = 275 |

**Latency:**

| Percentile | Value |
|:---|---:|
| p50 | 861.8ms |
| p95 | 898.9ms |
| p99 | 942.3ms |
| Mean | 781.8ms |



---

## Published Baselines (Reference)

| System | Context | Precision | Recall | F1 | FPR | Latency | Source |
|:---|:---|---:|---:|---:|---:|---:|:---|
| **Guardian v1.8.0 (auto)** | no-context | **0.000** | **0.000** | **0.000** | **0.000** | **854.2ms** | *this report* |
| Lakera Guard | full-context | 0.984 | 0.707 | 0.823 | 0.007 | ~51ms | Palit et al. 2025 |
| Lakera Guard | no-context | 0.964 | 0.501 | — | 0.057 | ~51ms | Palit et al. 2025 |
| RF + OpenAI Embeddings | no-context | 0.867 | 0.870 | — | — | — | Academic baseline |

---

## Methodology

- **Guardian SDK version:** 1.8.0
- **Tier:** auto (community patterns (24) + ONNX MiniLM-L6-v2 semantic engine)
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
- **Evaluation date:** 2026-04-27 23:04 UTC

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
| salad_data | 21,593 | binary (attack + benign) | HuggingFace |

---

*Report generated by Ethicore Engine™ Guardian SDK Benchmark Suite.*
*© 2026 Oracles Technologies LLC — [oraclestechnologies.com](https://oraclestechnologies.com)*