# Ethicore Engine™ — Guardian SDK Benchmark Report

**Version:** 1.8.0
**Tier:** auto
**Generated:** 2026-04-28 02:38 UTC
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
| Precision | **0.964** | 0.984 | -0.020 |
| Recall | **0.266** | 0.707 | -0.441 |
| F1 Score | **0.400** | 0.823 | -0.423 |
| FPR | **0.010** | 0.007 | +0.003 |
| FNR | **0.734** | 0.293 | +0.441 |
| ROC-AUC | **0.778** | — | — |
| p50 Latency | **1974.5ms** | ~51ms | +1923.5ms |
| p99 Latency | **5254.4ms** | N/A | — |

> **Positive Δ = Guardian better** for Precision/Recall/F1/ROC-AUC.
> **Negative Δ = Guardian better** for FPR/FNR/Latency.

---

## Per-Dataset Results

### bundled (guardian)

| Metric | Value |
|:---|---:|
| Samples | 416 (141 attacks, 275 benign) |
| Precision | 0.911 |
| Recall | 0.291 |
| F1 Score | 0.441 |
| Accuracy | 0.750 |
| ROC-AUC | 0.984 |
| PR-AUC | 0.903 |
| FPR | 0.015 |
| FNR | 0.709 |

**Confusion Matrix:**

|  | Predicted Attack | Predicted Benign |
|:---|---:|---:|
| **Actual Attack** | TP = 41 | FN = 100 |
| **Actual Benign** | FP = 4 | TN = 271 |

**Latency:**

| Percentile | Value |
|:---|---:|
| p50 | 1493.3ms |
| p95 | 1754.6ms |
| p99 | 1904.3ms |
| Mean | 1399.9ms |


### neuralchemy (guardian)

| Metric | Value |
|:---|---:|
| Samples | 4,391 (2650 attacks, 1741 benign) |
| Precision | 0.965 |
| Recall | 0.387 |
| F1 Score | 0.553 |
| Accuracy | 0.622 |
| ROC-AUC | 0.769 |
| PR-AUC | 0.831 |
| FPR | 0.021 |
| FNR | 0.613 |

**Confusion Matrix:**

|  | Predicted Attack | Predicted Benign |
|:---|---:|---:|
| **Actual Attack** | TP = 1026 | FN = 1624 |
| **Actual Benign** | FP = 37 | TN = 1704 |

**Latency:**

| Percentile | Value |
|:---|---:|
| p50 | 1826.7ms |
| p95 | 3757.3ms |
| p99 | 4376.4ms |
| Mean | 2050.0ms |


### deepset (guardian)

| Metric | Value |
|:---|---:|
| Samples | 546 (203 attacks, 343 benign) |
| Precision | 0.913 |
| Recall | 0.310 |
| F1 Score | 0.463 |
| Accuracy | 0.733 |
| ROC-AUC | 0.691 |
| PR-AUC | 0.656 |
| FPR | 0.018 |
| FNR | 0.690 |

**Confusion Matrix:**

|  | Predicted Attack | Predicted Benign |
|:---|---:|---:|
| **Actual Attack** | TP = 63 | FN = 140 |
| **Actual Benign** | FP = 6 | TN = 337 |

**Latency:**

| Percentile | Value |
|:---|---:|
| p50 | 1920.3ms |
| p95 | 2890.2ms |
| p99 | 3008.7ms |
| Mean | 1921.8ms |


### jailbreakbench (guardian)

| Metric | Value |
|:---|---:|
| Samples | 200 (100 attacks, 100 benign) |
| Precision | 1.000 |
| Recall | 0.130 |
| F1 Score | 0.230 |
| Accuracy | 0.565 |
| ROC-AUC | 0.763 |
| PR-AUC | 0.762 |
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
| p50 | 1566.0ms |
| p95 | 1872.0ms |
| p99 | 1875.1ms |
| Mean | 1448.1ms |


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
| p50 | 3283.4ms |
| p95 | 4693.0ms |
| p99 | 4994.1ms |
| Mean | 2971.6ms |


### salad_data (guardian)

| Metric | Value |
|:---|---:|
| Samples | 21,593 (21318 attacks, 275 benign) |
| Precision | 0.999 |
| Recall | 0.077 |
| F1 Score | 0.143 |
| Accuracy | 0.089 |
| ROC-AUC | 0.705 |
| PR-AUC | 0.993 |
| FPR | 0.004 |
| FNR | 0.923 |

**Confusion Matrix:**

|  | Predicted Attack | Predicted Benign |
|:---|---:|---:|
| **Actual Attack** | TP = 1640 | FN = 19678 |
| **Actual Benign** | FP = 1 | TN = 274 |

**Latency:**

| Percentile | Value |
|:---|---:|
| p50 | 1757.2ms |
| p95 | 2126.7ms |
| p99 | 15367.6ms |
| Mean | 1928.6ms |



---

## Published Baselines (Reference)

| System | Context | Precision | Recall | F1 | FPR | Latency | Source |
|:---|:---|---:|---:|---:|---:|---:|:---|
| **Guardian v1.8.0 (auto)** | no-context | **0.964** | **0.266** | **0.400** | **0.010** | **1974.5ms** | *this report* |
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
- **Evaluation date:** 2026-04-28 02:38 UTC

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