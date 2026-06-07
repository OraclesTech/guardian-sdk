# Ethicore Engine™ — Guardian SDK Benchmark Report

**Version:** 2.6.5
**Tier:** auto
**Generated:** 2026-06-07 03:21 UTC
**Datasets:** bundled, agentic
**Total samples evaluated:** 400

---

## Executive Summary

Guardian SDK 2.6.5 (auto tier) evaluated against
400 prompts spanning 2
public benchmark datasets. Results compared against published Lakera Guard and
academic baselines.

### Key Numbers

| Metric | Guardian v2.6.5 (auto) | Lakera Guard (full-ctx, published) | Δ vs Lakera |
|:---|---:|---:|---:|
| Precision | **0.925** | 0.984 | -0.059 |
| Recall | **0.750** | 0.707 | +0.043 |
| F1 Score | **0.828** | 0.823 | +0.005 |
| FPR | **0.065** | 0.007 | +0.058 |
| FNR | **0.250** | 0.293 | -0.043 |
| ROC-AUC | **0.744** | — | — |
| p50 Latency | **2568.9ms** | ~51ms | +2517.9ms |
| p99 Latency | **5444.9ms** | N/A | — |

> **Positive Δ = Guardian better** for Precision/Recall/F1/ROC-AUC.
> **Negative Δ = Guardian better** for FPR/FNR/Latency.

---

## Per-Dataset Results

### bundled (guardian)

| Metric | Value |
|:---|---:|
| Samples | 200 (100 attacks, 100 benign) |
| Precision | 1.000 |
| Recall | 0.760 |
| F1 Score | 0.864 |
| Accuracy | 0.880 |
| ROC-AUC | 0.603 |
| PR-AUC | 0.612 |
| FPR | 0.000 |
| FNR | 0.240 |

**Confusion Matrix:**

|  | Predicted Attack | Predicted Benign |
|:---|---:|---:|
| **Actual Attack** | TP = 76 | FN = 24 |
| **Actual Benign** | FP = 0 | TN = 100 |

**Latency:**

| Percentile | Value |
|:---|---:|
| p50 | 2305.3ms |
| p95 | 3139.3ms |
| p99 | 3139.7ms |
| Mean | 2268.2ms |


### agentic (guardian)

| Metric | Value |
|:---|---:|
| Samples | 200 (100 attacks, 100 benign) |
| Precision | 0.851 |
| Recall | 0.740 |
| F1 Score | 0.791 |
| Accuracy | 0.805 |
| ROC-AUC | 0.885 |
| PR-AUC | 0.863 |
| FPR | 0.130 |
| FNR | 0.260 |

**Confusion Matrix:**

|  | Predicted Attack | Predicted Benign |
|:---|---:|---:|
| **Actual Attack** | TP = 74 | FN = 26 |
| **Actual Benign** | FP = 13 | TN = 87 |

**Latency:**

| Percentile | Value |
|:---|---:|
| p50 | 2832.5ms |
| p95 | 4921.9ms |
| p99 | 7750.1ms |
| Mean | 3077.0ms |



---

## Published Baselines (Reference)

| System | Context | Precision | Recall | F1 | FPR | Latency | Source |
|:---|:---|---:|---:|---:|---:|---:|:---|
| **Guardian v2.6.5 (auto)** | no-context | **0.925** | **0.750** | **0.828** | **0.065** | **2568.9ms** | *this report* |
| Lakera Guard | full-context | 0.984 | 0.707 | 0.823 | 0.007 | ~51ms | Palit et al. 2025 |
| Lakera Guard | no-context | 0.964 | 0.501 | — | 0.057 | ~51ms | Palit et al. 2025 |
| RF + OpenAI Embeddings | no-context | 0.867 | 0.870 | — | — | — | Academic baseline |

---

## Methodology

- **Guardian SDK version:** 2.6.5
- **Tier:** auto (community patterns (24) + ONNX MiniLM-L6-v2 semantic engine)
- **Binary classification:** Guardian's actual production decision is used directly —
  `recommended_action ∈ {BLOCK, CHALLENGE}` → label 1 (attack); `ALLOW / MONITOR` → label 0
  (benign). This is the same decision your application would receive in production, making
  the comparison apples-to-apples with Lakera Guard (which publishes against its own block
  threshold) and other production safety services.
- **Per-sample session isolation:** Each prompt receives a unique `session_id` in the
  `analyze()` context. Without isolation, Guardian's session-keyed analyzers
  (`automated_scan_detector`, `context_poisoning_tracker`, behavioral profile)
  accumulate state across the rapid-fire benchmark stream and correctly treat it as
  a coordinated scan attack, producing artificially high FPR. The unique `session_id`
  makes each benchmark sample an independent measurement — matching how a real
  application would handle distinct end-user requests.
- **Continuous score:** Raw `threat_score` (normalized to 0.0–1.0) used for ROC-AUC and PR-AUC.
- **Caching:** Disabled during benchmark (each prompt independently measured)
- **Latency measurement:** Wall-clock time from `analyze()` call to result return
- **Baselines:** Published numbers from peer-reviewed literature (no live API calls)
- **Evaluation date:** 2026-06-07 03:21 UTC

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
| bundled | 200 | binary (attack + benign) | Bundled (local) |
| agentic | 200 | binary (attack + benign) | HuggingFace |

---

*Report generated by Ethicore Engine™ Guardian SDK Benchmark Suite.*
*© 2026 Oracles Technologies LLC — [oraclestechnologies.com](https://oraclestechnologies.com)*