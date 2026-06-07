# Ethicore Engine™ — Guardian SDK Benchmark Report

**Version:** 1.8.0
**Tier:** auto
**Generated:** 2026-04-28 03:29 UTC
**Datasets:** neuralchemy, salad_data, deepset, jailbreakbench, bundled, b3
**Total samples evaluated:** 28,051

---

## Executive Summary

Guardian SDK 1.8.0 (auto tier) evaluated against
28,051 prompts spanning 6
public benchmark datasets. Results compared against published Lakera Guard and
academic baselines.

### Key Numbers

| Metric | Guardian v1.8.0 (auto) | Lakera Guard (full-ctx, published) | Δ vs Lakera |
|:---|---:|---:|---:|
| Precision | **0.976** | 0.984 | -0.008 |
| Recall | **0.447** | 0.707 | -0.260 |
| F1 Score | **0.547** | 0.823 | -0.276 |
| FPR | **0.013** | 0.007 | +0.006 |
| FNR | **0.553** | 0.293 | +0.260 |
| ROC-AUC | **0.832** | — | — |
| p50 Latency | **2207.1ms** | ~51ms | +2156.1ms |
| p99 Latency | **9774.7ms** | N/A | — |

> **Positive Δ = Guardian better** for Precision/Recall/F1/ROC-AUC.
> **Negative Δ = Guardian better** for FPR/FNR/Latency.

---

## Per-Dataset Results

### bundled (guardian)

| Metric | Value |
|:---|---:|
| Samples | 416 (141 attacks, 275 benign) |
| Precision | 0.993 |
| Recall | 0.993 |
| F1 Score | 0.993 |
| Accuracy | 0.995 |
| ROC-AUC | 0.998 |
| PR-AUC | 0.995 |
| FPR | 0.004 |
| FNR | 0.007 |

**Confusion Matrix:**

|  | Predicted Attack | Predicted Benign |
|:---|---:|---:|
| **Actual Attack** | TP = 140 | FN = 1 |
| **Actual Benign** | FP = 1 | TN = 274 |

**Latency:**

| Percentile | Value |
|:---|---:|
| p50 | 1620.9ms |
| p95 | 2586.9ms |
| p99 | 40857.1ms |
| Mean | 3428.8ms |


### neuralchemy (guardian)

| Metric | Value |
|:---|---:|
| Samples | 4,391 (2650 attacks, 1741 benign) |
| Precision | 0.966 |
| Recall | 0.401 |
| F1 Score | 0.567 |
| Accuracy | 0.630 |
| ROC-AUC | 0.791 |
| PR-AUC | 0.847 |
| FPR | 0.022 |
| FNR | 0.599 |

**Confusion Matrix:**

|  | Predicted Attack | Predicted Benign |
|:---|---:|---:|
| **Actual Attack** | TP = 1063 | FN = 1587 |
| **Actual Benign** | FP = 38 | TN = 1703 |

**Latency:**

| Percentile | Value |
|:---|---:|
| p50 | 2086.2ms |
| p95 | 3879.1ms |
| p99 | 4322.2ms |
| Mean | 2212.0ms |


### deepset (guardian)

| Metric | Value |
|:---|---:|
| Samples | 546 (203 attacks, 343 benign) |
| Precision | 0.902 |
| Recall | 0.680 |
| F1 Score | 0.775 |
| Accuracy | 0.854 |
| ROC-AUC | 0.973 |
| PR-AUC | 0.935 |
| FPR | 0.044 |
| FNR | 0.320 |

**Confusion Matrix:**

|  | Predicted Attack | Predicted Benign |
|:---|---:|---:|
| **Actual Attack** | TP = 138 | FN = 65 |
| **Actual Benign** | FP = 15 | TN = 328 |

**Latency:**

| Percentile | Value |
|:---|---:|
| p50 | 2159.4ms |
| p95 | 3020.3ms |
| p99 | 3187.5ms |
| Mean | 2105.2ms |


### jailbreakbench (guardian)

| Metric | Value |
|:---|---:|
| Samples | 200 (100 attacks, 100 benign) |
| Precision | 1.000 |
| Recall | 0.130 |
| F1 Score | 0.230 |
| Accuracy | 0.565 |
| ROC-AUC | 0.770 |
| PR-AUC | 0.764 |
| FPR | 0.000 |
| FNR | 0.870 |

**Confusion Matrix:**

|  | Predicted Attack | Predicted Benign |
|:---|---:|---:|
| **Actual Attack** | TP = 13 | FN = 87 |
| **Actual Benign** | FP = 0 | TN = 100 |

**Latency:**

| Percentile | Value |
|:---|---:|
| p50 | 1832.1ms |
| p95 | 2112.1ms |
| p99 | 2115.4ms |
| Mean | 1649.1ms |


### b3 (guardian)

| Metric | Value |
|:---|---:|
| Samples | 905 (630 attacks, 275 benign) |
| Precision | 0.996 |
| Recall | 0.400 |
| F1 Score | 0.571 |
| Accuracy | 0.581 |
| ROC-AUC | 0.757 |
| PR-AUC | 0.858 |
| FPR | 0.004 |
| FNR | 0.600 |

**Confusion Matrix:**

|  | Predicted Attack | Predicted Benign |
|:---|---:|---:|
| **Actual Attack** | TP = 252 | FN = 378 |
| **Actual Benign** | FP = 1 | TN = 274 |

**Latency:**

| Percentile | Value |
|:---|---:|
| p50 | 3588.9ms |
| p95 | 5027.1ms |
| p99 | 5766.6ms |
| Mean | 3261.3ms |


### salad_data (guardian)

| Metric | Value |
|:---|---:|
| Samples | 21,593 (21318 attacks, 275 benign) |
| Precision | 0.999 |
| Recall | 0.077 |
| F1 Score | 0.144 |
| Accuracy | 0.089 |
| ROC-AUC | 0.704 |
| PR-AUC | 0.993 |
| FPR | 0.004 |
| FNR | 0.923 |

**Confusion Matrix:**

|  | Predicted Attack | Predicted Benign |
|:---|---:|---:|
| **Actual Attack** | TP = 1650 | FN = 19668 |
| **Actual Benign** | FP = 1 | TN = 274 |

**Latency:**

| Percentile | Value |
|:---|---:|
| p50 | 1955.0ms |
| p95 | 2247.4ms |
| p99 | 2399.4ms |
| Mean | 1797.2ms |



---

## Published Baselines (Reference)

| System | Context | Precision | Recall | F1 | FPR | Latency | Source |
|:---|:---|---:|---:|---:|---:|---:|:---|
| **Guardian v1.8.0 (auto)** | no-context | **0.976** | **0.447** | **0.547** | **0.013** | **2207.1ms** | *this report* |
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
- **Evaluation date:** 2026-04-28 03:29 UTC

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
| b3 | 905 | binary (attack + benign) | HuggingFace |
| salad_data | 21,593 | binary (attack + benign) | HuggingFace |

---

*Report generated by Ethicore Engine™ Guardian SDK Benchmark Suite.*
*© 2026 Oracles Technologies LLC — [oraclestechnologies.com](https://oraclestechnologies.com)*