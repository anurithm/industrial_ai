"""
End-to-end training pipeline script.
Run: python -m src.train_pipeline

Executes all phases:
  1. Download dataset (if needed)
  2. Validate schema
  3. Preprocess and split
  4. Train and compare models
  5. Select best model
  6. Final test set evaluation
  7. Train anomaly detector
  8. Build RAG vector store
"""
import logging
import sys
import json
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_pipeline")


def main():
    logger.info("=" * 60)
    logger.info("IndustrialAI Training Pipeline")
    logger.info("=" * 60)

    # ── Phase 1: Load and validate dataset ────────────────────────
    logger.info("\n[Phase 1] Loading dataset...")
    from src.data_loader import load_raw_data, validate_schema
    try:
        df = load_raw_data()
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    validation = validate_schema(df)
    if not validation["valid"]:
        logger.error(f"Schema validation failed: {validation['issues']}")
        sys.exit(1)

    for w in validation["warnings"]:
        logger.info(f"  ⚠ {w}")
    logger.info(f"✓ Dataset validated: {df.shape[0]} rows, {df.shape[1]} columns")

    # ── Phase 2: Preprocess ────────────────────────────────────────
    logger.info("\n[Phase 2] Preprocessing and splitting data...")
    from src.preprocessing import prepare_pipeline, PROCESSED_DATA_DIR
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = prepare_pipeline(df)
    logger.info(
        f"✓ Splits: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}"
    )
    logger.info(
        f"  Failure rate: Train={y_train.mean():.2%}, Val={y_val.mean():.2%}, Test={y_test.mean():.2%}"
    )

    # ── Phase 3: Train and compare models ─────────────────────────
    logger.info("\n[Phase 3] Training models (LR, RF, XGBoost)...")
    from src.train_model import train_and_compare, evaluate_on_test
    results = train_and_compare(X_train, y_train.values, X_val, y_val.values)
    best = results["best_model"]
    logger.info(f"✓ Best model: {best}")

    # Print validation comparison
    logger.info("\nValidation Metrics Comparison:")
    for name, metrics in results["model_comparison"].items():
        logger.info(
            f"  {name}: F1={metrics['f1_score']:.4f}, "
            f"PR-AUC={metrics.get('pr_auc', 'N/A')}, "
            f"ROC-AUC={metrics.get('roc_auc', 'N/A')}, "
            f"Recall={metrics['recall']:.4f}"
        )

    # ── Phase 4: Test set evaluation ──────────────────────────────
    logger.info("\n[Phase 4] Final evaluation on held-out test set...")
    test_metrics = evaluate_on_test(X_test, y_test.values)
    logger.info(
        f"✓ Test: F1={test_metrics['f1_score']:.4f}, "
        f"PR-AUC={test_metrics.get('pr_auc')}, "
        f"ROC-AUC={test_metrics.get('roc_auc')}, "
        f"Recall={test_metrics['recall']:.4f}"
    )
    logger.info(
        f"  FP={test_metrics['false_positives']}, FN={test_metrics['false_negatives']}"
    )

    # ── Phase 5: Anomaly detection ────────────────────────────────
    logger.info("\n[Phase 5] Training anomaly detector (Isolation Forest)...")
    from src.anomaly_detection import train_anomaly_detector
    import pandas as pd
    X_train_raw = pd.read_parquet(PROCESSED_DATA_DIR / "X_train_raw.parquet")
    detector = train_anomaly_detector(X_train_raw, y_train.values)
    logger.info("✓ Anomaly detector trained")

    # ── Phase 6: RAG vector store ─────────────────────────────────
    logger.info("\n[Phase 6] Building RAG vector store...")
    try:
        from src.maintenance_rag import build_vector_store
        collection = build_vector_store(force_rebuild=True)
        logger.info(f"✓ Vector store built: {collection.count()} chunks indexed")
    except Exception as e:
        logger.warning(f"Vector store build failed (non-fatal): {e}")

    logger.info("\n" + "=" * 60)
    logger.info("✓ Training pipeline complete!")
    logger.info("  Run API:       uvicorn src.api:app --reload --port 8000")
    logger.info("  Run Dashboard: streamlit run frontend/dashboard.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
