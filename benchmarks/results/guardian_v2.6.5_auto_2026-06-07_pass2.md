# Ethicore Engine™ — Guardian SDK Benchmark Report

**Version:** 2.6.5
**Tier:** auto
**Generated:** 2026-06-07 03:52 UTC
**Datasets:** agentic, bundled
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
| Precision | **0.967** | 0.984 | -0.017 |
| Recall | **0.995** | 0.707 | +0.288 |
| F1 Score | **0.981** | 0.823 | +0.158 |
| FPR | **0.035** | 0.007 | +0.028 |
| FNR | **0.005** | 0.293 | -0.288 |
| ROC-AUC | **0.800** | — | — |
| p50 Latency | **5690.8ms** | ~51ms | +5639.8ms |
| p99 Latency | **15251.4ms** | N/A | — |

> **Positive Δ = Guardian better** for Precision/Recall/F1/ROC-AUC.
> **Negative Δ = Guardian better** for FPR/FNR/Latency.

---

## Per-Dataset Results

### bundled (guardian)

| Metric | Value |
|:---|---:|
| Samples | 200 (100 attacks, 100 benign) |
| Precision | 1.000 |
| Recall | 0.990 |
| F1 Score | 0.995 |
| Accuracy | 0.995 |
| ROC-AUC | 0.634 |
| PR-AUC | 0.653 |
| FPR | 0.000 |
| FNR | 0.010 |

**Confusion Matrix:**

|  | Predicted Attack | Predicted Benign |
|:---|---:|---:|
| **Actual Attack** | TP = 99 | FN = 1 |
| **Actual Benign** | FP = 0 | TN = 100 |

**Latency:**

| Percentile | Value |
|:---|---:|
| p50 | 4102.6ms |
| p95 | 6532.0ms |
| p99 | 6533.0ms |
| Mean | 3906.9ms |


### agentic (guardian)

| Metric | Value |
|:---|---:|
| Samples | 200 (100 attacks, 100 benign) |
| Precision | 0.935 |
| Recall | 1.000 |
| F1 Score | 0.966 |
| Accuracy | 0.965 |
| ROC-AUC | 0.966 |
| PR-AUC | 0.945 |
| FPR | 0.070 |
| FNR | 0.000 |

**Confusion Matrix:**

|  | Predicted Attack | Predicted Benign |
|:---|---:|---:|
| **Actual Attack** | TP = 100 | FN = 0 |
| **Actual Benign** | FP = 7 | TN = 93 |

**Latency:**

| Percentile | Value |
|:---|---:|
| p50 | 7279.0ms |
| p95 | 12617.4ms |
| p99 | 23969.8ms |
| Mean | 8158.6ms |



---

## Published Baselines (Reference)

| System | Context | Precision | Recall | F1 | FPR | Latency | Source |
|:---|:---|---:|---:|---:|---:|---:|:---|
| **Guardian v2.6.5 (auto)** | no-context | **0.967** | **0.995** | **0.981** | **0.035** | **5690.8ms** | *this report* |
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
- **Evaluation date:** 2026-06-07 03:52 UTC

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