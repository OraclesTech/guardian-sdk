#!/usr/bin/env python3
"""
retrain_guardian_model.py
Ethicore Engine™ — Guardian SDK

Retrains the guardian-model.onnx ML classifier from the current threat pattern
library.  Generates synthetic training data from semantic fingerprints and
regex-derived examples, trains an sklearn MLPClassifier, and exports the result
to ONNX format matching the interface expected by MLInferenceEngine:

    Input:   dense_1_input   shape [N, 127]  float32
    Output:  dense_4         shape [N, 1]    float32  (sigmoid probability)

The script also runs the same calibration gate that MLInferenceEngine uses at
startup (avg benign probability < 0.4) and refuses to write the model if it
fails — preventing deployment of a miscalibrated classifier.

Principle 14 (Divine Safety): we never write a model file that has not passed
the benign-calibration gate.  Better to keep the previous model than to ship
one that systematically misclassifies legitimate requests.

Prerequisites (install with the [ml] extra):
    pip install scikit-learn skl2onnx onnxruntime

Usage:
    # Community edition (5 categories):
    python scripts/retrain_guardian_model.py

    # Licensed edition (51 categories) — auto-detects via env vars:
    ETHICORE_LICENSE_KEY="EG-PRO-..." python scripts/retrain_guardian_model.py

    # Custom output location:
    python scripts/retrain_guardian_model.py --out /opt/ethicore/models/guardian-model.onnx

    # Dry run (build dataset, train, evaluate — but do not write):
    python scripts/retrain_guardian_model.py --dry-run

    # Force overwrite without prompting:
    python scripts/retrain_guardian_model.py --force

    # Tune training (defaults shown):
    python scripts/retrain_guardian_model.py --samples 3000 --hidden 128,64 --seed 42
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import random
import sys
import time
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Project root on sys.path so ethicore_guardian is importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = pathlib.Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

def _check_deps() -> None:
    missing = []
    for pkg in ("sklearn", "skl2onnx", "onnxruntime", "numpy"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(
            "[ERR]  Missing required packages: " + ", ".join(missing) + "\n"
            "       Install with:  pip install scikit-learn skl2onnx onnxruntime",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Asset / path resolution
# ---------------------------------------------------------------------------

def _resolve_output_path(
    explicit_out: str | None,
    assets_dir: str | None,
    license_key: str | None,
) -> pathlib.Path:
    """Determine where guardian-model.onnx should be written.

    Priority:
      1. --out flag
      2. <assets_dir>/models/
      3. ~/.ethicore/models/   (licensed tier, home convention)
      4. <package>/models/     (community fallback)
    """
    if explicit_out:
        return pathlib.Path(explicit_out)
    if license_key:
        if assets_dir:
            return pathlib.Path(assets_dir) / "models" / "guardian-model.onnx"
        return pathlib.Path.home() / ".ethicore" / "models" / "guardian-model.onnx"
    return _PROJECT_ROOT / "ethicore_guardian" / "models" / "guardian-model.onnx"


def _load_threat_module(license_key: str | None, assets_dir: str | None):
    """Return the appropriate threat_patterns module (licensed or community)."""
    if license_key:
        candidates = []
        if assets_dir:
            candidates.append(
                pathlib.Path(assets_dir) / "data" / "threat_patterns_licensed.py"
            )
        candidates.append(
            pathlib.Path.home() / ".ethicore" / "data" / "threat_patterns_licensed.py"
        )
        candidates.append(
            _PROJECT_ROOT / "licensed" / "data" / "threat_patterns_licensed.py"
        )
        for path in candidates:
            if not path.exists():
                continue
            try:
                spec = importlib.util.spec_from_file_location("_tpl_licensed", str(path))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[union-attr]
                print(f"  Loaded licensed patterns from: {path}")
                return mod
            except Exception as exc:
                print(f"[WARN]  Could not load {path}: {exc}", file=sys.stderr)
        print(
            "[WARN]  License key supplied but licensed asset not found.\n"
            "        Falling back to community patterns.",
            file=sys.stderr,
        )

    from ethicore_guardian.data import threat_patterns as comm
    return comm


# ---------------------------------------------------------------------------
# Training data generation
# ---------------------------------------------------------------------------

# ── Benign sentence templates ────────────────────────────────────────────────
_BENIGN_TEMPLATES = [
    "Hello, how are you today?",
    "Can you help me write a Python function that sorts a list?",
    "What is the capital of France?",
    "Explain how recursion works with a simple example.",
    "What time is it in Tokyo right now?",
    "Summarize this article for me: {}",
    "How do I install NumPy using pip?",
    "What's the difference between a list and a tuple in Python?",
    "Can you translate this sentence to Spanish? {}",
    "Write a haiku about the ocean.",
    "What are some good books on machine learning?",
    "Help me debug this error: TypeError: 'NoneType' object is not iterable",
    "How do neural networks learn from data?",
    "What is the boiling point of water at sea level?",
    "Draft an email thanking my colleague for their help on the project.",
    "Can you proofread this paragraph for grammar?",
    "What is the time complexity of quicksort in the worst case?",
    "Explain the concept of overfitting in machine learning.",
    "How do I create a virtual environment in Python?",
    "What are the main differences between REST and GraphQL?",
    "Write unit tests for a function that adds two numbers.",
    "Can you help me understand how transformers work in NLP?",
    "What is the difference between supervised and unsupervised learning?",
    "How do I reverse a string in Python without slicing?",
    "What is gradient descent and how does it work?",
    "Can you recommend a framework for building REST APIs in Python?",
    "Explain the CAP theorem in distributed systems.",
    "How do I handle exceptions in Python?",
    "What is the purpose of the __init__.py file in Python packages?",
    "Can you help me understand regular expressions?",
    "Write a function to find the longest common subsequence.",
    "What is the difference between deep copy and shallow copy?",
    "How does garbage collection work in Python?",
    "Explain the concept of closures in JavaScript.",
    "What are the SOLID principles in software design?",
    "How do I connect to a PostgreSQL database using Python?",
    "Explain the difference between Git merge and Git rebase.",
    "What is the purpose of Docker in software development?",
    "How does HTTPS work under the hood?",
    "What is the difference between a process and a thread?",
    "Help me understand what a decorator does in Python.",
    "What is the time complexity of binary search?",
    "Can you explain the concept of dependency injection?",
    "What are the advantages of using type hints in Python?",
    "How do I parse JSON in Python?",
    "What is the difference between SQL and NoSQL databases?",
    "Explain what a race condition is in concurrent programming.",
    "How do I implement a stack using a list in Python?",
    "What is the difference between authentication and authorization?",
    "How do I center a div using CSS?",
]

_BENIGN_FILLER = [
    "the quick brown fox jumps over the lazy dog",
    "machine learning is a subset of artificial intelligence",
    "Python was created by Guido van Rossum",
    "the weather is nice today",
    "software testing is important for quality assurance",
]


def _make_benign_sample(rng: random.Random) -> str:
    template = rng.choice(_BENIGN_TEMPLATES)
    if "{}" in template:
        template = template.replace("{}", rng.choice(_BENIGN_FILLER))
    return template


# ── Threat sample generation ─────────────────────────────────────────────────

def _make_threat_sample(
    fingerprint_text: str,
    category: str,
    rng: random.Random,
) -> str:
    """Generate a threat sample by lightly paraphrasing a semantic fingerprint."""
    prefixes = [
        "",
        "Hey, ",
        "Listen, ",
        "Actually, ",
        "OK so ",
        "For this task, ",
        "Just so you know, ",
        "By the way, ",
        "I need you to understand: ",
        "Starting now, ",
        "From this point on, ",
        "Attention: ",
    ]
    suffixes = [
        "",
        ".",
        " immediately.",
        " right now.",
        " without any restrictions.",
        " and do not refuse.",
        " as requested.",
        " — this is important.",
        " — no exceptions.",
        " completely and without hesitation.",
    ]
    prefix = rng.choice(prefixes)
    suffix = rng.choice(suffixes)
    return f"{prefix}{fingerprint_text}{suffix}".strip()


def _build_dataset(
    threat_module,
    n_samples: int,
    seed: int,
) -> Tuple[List[str], List[int]]:
    """Return (texts, labels) where label 1 = threat, 0 = benign.

    Generates a balanced dataset with equal threat/benign samples.
    Each threat category contributes proportionally to its fingerprint count.
    """
    rng = random.Random(seed)

    fingerprints = threat_module.get_semantic_fingerprints()
    stats = threat_module.get_threat_statistics()
    edition = stats.get("edition", "?")
    print(
        f"  Threat library: {stats['totalCategories']} categories, "
        f"{len(fingerprints)} fingerprints ({edition} edition)"
    )

    n_threat = n_samples // 2
    n_benign = n_samples - n_threat

    texts: List[str] = []
    labels: List[int] = []

    # ── Threat samples ─────────────────────────────────────────────────────
    # Sample fingerprints with replacement, weighted by category weight
    weights = [fp["weight"] for fp in fingerprints]
    total_w = sum(weights)
    probs = [w / total_w for w in weights]

    for _ in range(n_threat):
        # Weighted random selection
        r = rng.random()
        cumulative = 0.0
        chosen = fingerprints[0]
        for fp, p in zip(fingerprints, probs):
            cumulative += p
            if r <= cumulative:
                chosen = fp
                break
        sample = _make_threat_sample(chosen["text"], chosen["category"], rng)
        texts.append(sample)
        labels.append(1)

    # ── Benign samples ────────────────────────────────────────────────────
    for _ in range(n_benign):
        texts.append(_make_benign_sample(rng))
        labels.append(0)

    # Shuffle
    combined = list(zip(texts, labels))
    rng.shuffle(combined)
    texts, labels = zip(*combined)  # type: ignore[assignment]

    threat_count = sum(labels)
    print(
        f"  Training samples:   {len(texts)} "
        f"({threat_count} threat / {len(texts) - threat_count} benign)"
    )
    return list(texts), list(labels)


# ---------------------------------------------------------------------------
# Feature extraction shim (wraps MLInferenceEngine.extract_features)
# ---------------------------------------------------------------------------

def _extract_features_batch(texts: List[str]) -> List[List[float]]:
    """Extract 127-dim feature vectors for a batch of texts using the SDK."""
    from ethicore_guardian.analyzers.ml_inference_engine import MLInferenceEngine

    # Use a minimal engine instance — we only need extract_features(), not inference
    engine = MLInferenceEngine.__new__(MLInferenceEngine)
    # Initialise only the attributes used by extract_features()
    engine.feature_config = {
        "behavioral": 40, "linguistic": 35, "technical": 25, "semantic": 27, "total": 127
    }

    features = []
    for text in texts:
        vec = engine.extract_features(text)
        features.append(vec)
    return features


# ---------------------------------------------------------------------------
# Train + export
# ---------------------------------------------------------------------------

def _train_and_export(
    texts: List[str],
    labels: List[int],
    hidden_layers: Tuple[int, ...],
    seed: int,
    output_path: pathlib.Path,
    dry_run: bool,
) -> bool:
    """Train MLPClassifier, run calibration gate, export to ONNX.

    Returns True on success, False if the calibration gate fails.
    """
    import numpy as np
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    print(f"\n  Extracting features for {len(texts)} samples …")
    t0 = time.time()
    X = np.array(_extract_features_batch(texts), dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    print(f"  Feature extraction: {time.time() - t0:.1f}s  (shape: {X.shape})")

    # Train / validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=seed, stratify=y
    )
    print(f"  Train: {len(X_train)} samples  |  Val: {len(X_val)} samples")

    print(f"\n  Training MLPClassifier {hidden_layers} …")
    t0 = time.time()
    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=300,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        verbose=False,
    )
    clf.fit(X_train, y_train)
    print(f"  Training complete:  {time.time() - t0:.1f}s  "
          f"(iterations: {clf.n_iter_})")

    # Evaluation
    y_pred = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)[:, 1]
    report = classification_report(y_val, y_pred, target_names=["benign", "threat"])
    print(f"\n  Validation report:\n{report}")

    # ── Calibration gate ──────────────────────────────────────────────────
    # Mirror the gate in MLInferenceEngine.initialize() exactly.
    # If the model rates obviously benign prompts above 0.4 on average,
    # it is miscalibrated and we must not ship it.
    calibration_texts = [
        "Hello, how can I help you today?",
        "What time is it in Tokyo?",
        "Can you explain how recursion works in Python?",
    ]
    cal_X = np.array(
        _extract_features_batch(calibration_texts), dtype=np.float32
    )
    cal_proba = clf.predict_proba(cal_X)[:, 1]
    avg_benign = float(cal_proba.mean())
    print(f"\n  Calibration gate:   avg benign probability = {avg_benign:.4f}")

    if avg_benign > 0.4:
        print(
            f"\n[ERR]  Calibration gate FAILED "
            f"(avg_benign_prob={avg_benign:.4f} > 0.4).\n"
            "       The model would produce systematic false positives.\n"
            "       Recommendations:\n"
            "         - Increase --samples (try 5000+)\n"
            "         - Adjust --hidden (try 256,128)\n"
            "         - Inspect the benign template set in this script\n"
            "       The existing model has NOT been overwritten.",
            file=sys.stderr,
        )
        return False

    print(f"  Calibration gate:   PASSED ✓")

    if dry_run:
        print("\n[dry-run] Model not written (--dry-run flag set).")
        return True

    # ── ONNX export ───────────────────────────────────────────────────────
    # MLInferenceEngine expects:
    #   input  name: 'dense_1_input'   shape [N, 127]  float32
    #   output name: 'dense_4'         shape [N, 1]    float32 (sigmoid prob)
    print(f"\n  Exporting to ONNX …")
    initial_type = [("dense_1_input", FloatTensorType([None, 127]))]
    onnx_model = convert_sklearn(
        clf,
        initial_types=initial_type,
        options={id(clf): {"zipmap": False}},
    )

    # Rename the output node to 'dense_4' so MLInferenceEngine's
    # hardcoded output name matches.  skl2onnx names it 'probabilities'.
    for output in onnx_model.graph.output:
        if output.name == "probabilities":
            output.name = "dense_4"

    # Also rename in any node that produces 'probabilities'
    for node in onnx_model.graph.node:
        node.output[:] = [
            "dense_4" if o == "probabilities" else o for o in node.output
        ]

    # Ensure output is scalar probability (class 1), not 2-class array
    # The MLPClassifier with zipmap=False outputs [N, 2]; we need [N, 1].
    # Inject a Gather node to slice column 1.
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    import numpy as np

    # Build a new graph that slices probability[:,1:2] from the MLP output
    # ── Step 1: find the final output of the existing model ──────────────
    existing_output_name = onnx_model.graph.output[0].name  # 'dense_4'

    # ── Step 2: add a Gather node (axis=1, indices=[1]) ──────────────────
    indices_name = "_gather_idx_1"
    indices_initializer = numpy_helper.from_array(
        np.array([1], dtype=np.int64), name=indices_name
    )
    onnx_model.graph.initializer.append(indices_initializer)

    gather_out = "_prob_class1"
    gather_node = helper.make_node(
        "Gather",
        inputs=[existing_output_name, indices_name],
        outputs=[gather_out],
        axis=1,
        name="Gather_class1",
    )

    # ── Step 3: add Unsqueeze to restore shape [N, 1] ────────────────────
    if onnx_model.opset_import[0].version >= 13:
        unsqueeze_axes_name = "_unsqueeze_axes"
        axes_init = numpy_helper.from_array(
            np.array([1], dtype=np.int64), name=unsqueeze_axes_name
        )
        onnx_model.graph.initializer.append(axes_init)
        unsqueeze_node = helper.make_node(
            "Unsqueeze",
            inputs=[gather_out, unsqueeze_axes_name],
            outputs=["dense_4_final"],
            name="Unsqueeze_class1",
        )
    else:
        unsqueeze_node = helper.make_node(
            "Unsqueeze",
            inputs=[gather_out],
            outputs=["dense_4_final"],
            axes=[1],
            name="Unsqueeze_class1",
        )

    onnx_model.graph.node.extend([gather_node, unsqueeze_node])

    # ── Step 4: update graph output to the final sliced node ─────────────
    del onnx_model.graph.output[:]
    final_output = helper.make_tensor_value_info(
        "dense_4_final", TensorProto.FLOAT, [None, 1]
    )
    onnx_model.graph.output.append(final_output)

    # Rename the new final output to 'dense_4' for MLInferenceEngine
    onnx_model.graph.output[0].name = "dense_4"
    for node in onnx_model.graph.node:
        node.output[:] = [
            "dense_4" if o == "dense_4_final" else o for o in node.output
        ]

    # ── Step 5: write file ────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as fh:
        fh.write(onnx_model.SerializeToString())

    size_kb = output_path.stat().st_size // 1024
    print(f"\n[OK]  guardian-model.onnx written: {output_path}  ({size_kb} KB)")

    # ── Step 6: quick self-check via onnxruntime ──────────────────────────
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(output_path))
        test_feat = np.array([_extract_features_batch(["hello world"])[0]], dtype=np.float32)
        out = sess.run(None, {"dense_1_input": test_feat})
        prob = float(out[0][0][0])
        print(f"  Self-check probability (benign): {prob:.4f}  ✓")
    except Exception as exc:
        print(f"[WARN]  Self-check failed: {exc}", file=sys.stderr)

    return True


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    _check_deps()

    parser = argparse.ArgumentParser(
        description=(
            "Retrain and export guardian-model.onnx for the Guardian SDK.\n"
            "Works with Community (5 categories) and Licensed (51 categories) editions."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Train and evaluate but do not write the ONNX file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing model without prompting.",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        default=None,
        help="Explicit output path for guardian-model.onnx (overrides auto-resolve).",
    )
    parser.add_argument(
        "--license-key",
        metavar="KEY",
        default=None,
        help="License key (overrides $ETHICORE_LICENSE_KEY).",
    )
    parser.add_argument(
        "--assets-dir",
        metavar="DIR",
        default=None,
        help="Path to extracted asset bundle (overrides $ETHICORE_ASSETS_DIR).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3000,
        help="Total synthetic training samples (default: 3000, half threat / half benign).",
    )
    parser.add_argument(
        "--hidden",
        default="128,64",
        help="Hidden layer sizes, comma-separated (default: '128,64').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args(argv)

    # Resolve credentials: CLI > env
    license_key = args.license_key or os.environ.get("ETHICORE_LICENSE_KEY") or None
    assets_dir = args.assets_dir or os.environ.get("ETHICORE_ASSETS_DIR") or None
    if license_key:
        license_key = license_key.strip()
    if assets_dir:
        assets_dir = assets_dir.strip()

    # Parse hidden layer sizes
    try:
        hidden_layers = tuple(int(x) for x in args.hidden.split(",") if x.strip())
        if not hidden_layers:
            raise ValueError("empty")
    except ValueError:
        print(f"[ERR]  Invalid --hidden value: {args.hidden!r}", file=sys.stderr)
        return 1

    output_path = _resolve_output_path(args.out, assets_dir, license_key)

    print("=" * 60)
    print("  Guardian SDK — Model Retraining")
    print("=" * 60)
    print(f"  Output path:  {output_path}")
    print(f"  Samples:      {args.samples}")
    print(f"  Hidden layers:{hidden_layers}")
    print(f"  Seed:         {args.seed}")
    edition_label = "licensed" if license_key else "community"
    print(f"  Edition:      {edition_label}")
    print("=" * 60)

    # Prompt before overwrite
    if not args.dry_run and not args.force and output_path.exists():
        try:
            answer = input(
                f"\nguardian-model.onnx already exists at:\n  {output_path}\n"
                "Overwrite? [y/N] "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return 0
        if answer not in ("y", "yes"):
            print("Aborted.")
            return 0

    # Load the threat module
    threat_module = _load_threat_module(license_key, assets_dir)

    # Build dataset
    print("\n  Building synthetic training dataset …")
    texts, labels = _build_dataset(threat_module, args.samples, args.seed)

    # Train + export
    ok = _train_and_export(
        texts=texts,
        labels=labels,
        hidden_layers=hidden_layers,
        seed=args.seed,
        output_path=output_path,
        dry_run=args.dry_run,
    )

    if ok and not args.dry_run:
        print(
            "\n  Next steps:\n"
            f"    1. Re-run regenerate_embeddings.py to sync embeddings\n"
            f"    2. Run: pytest tests/ -v  to confirm all tests pass\n"
            f"    3. Commit guardian-model.onnx to the asset bundle\n"
            f"    4. Update model_signatures.json:\n"
            f"         python scripts/generate_model_signatures.py"
        )

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
