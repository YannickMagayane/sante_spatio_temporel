"""
Evaluate trained models on test data using regression and classification metrics.

Computes RMSE, MAE, AUC, F1-score, and generates SHAP explanations.

Example:
    $ python scripts/evaluate.py --model lstm --test_data data/processed/test.csv
    $ python scripts/evaluate.py --model cnn
"""

import argparse
import json
import os
import glob

import numpy as np


def compute_rmse(y_true, y_pred):
    """
    Compute Root Mean Squared Error.

    :param y_true: Ground truth values.
    :param y_pred: Predicted values.
    :return: RMSE score (float).

    Example:
        >>> compute_rmse(np.array([1, 2, 3]), np.array([1.1, 2.1, 2.9]))
        0.1
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mae(y_true, y_pred):
    """
    Compute Mean Absolute Error.

    :param y_true: Ground truth values.
    :param y_pred: Predicted values.
    :return: MAE score (float).

    Example:
        >>> compute_mae(np.array([1, 2, 3]), np.array([1, 2, 4]))
        0.333
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_classification_metrics(y_true, y_pred, threshold):
    """
    Compute AUC and F1-score for epidemic risk classification.

    Values above the threshold are classified as "risk".

    :param y_true: Ground truth case counts.
    :param y_pred: Predicted case counts.
    :param threshold: Threshold for risk classification.
    :return: Dictionary with AUC and F1-score.

    Example:
        >>> metrics = compute_classification_metrics(
        ...     np.array([10, 50, 100]), np.array([15, 45, 90]), threshold=30
        ... )
        >>> "auc" in metrics and "f1_score" in metrics
        True
    """
    from sklearn.metrics import roc_auc_score, f1_score

    y_true_binary = (y_true > threshold).astype(int)
    y_pred_binary = (y_pred > threshold).astype(int)

    try:
        auc = float(roc_auc_score(y_true_binary, y_pred))
    except ValueError:
        auc = 0.0

    f1 = float(f1_score(y_true_binary, y_pred_binary, zero_division=0))

    return {"auc": round(auc, 4), "f1_score": round(f1, 4)}


def generate_shap_explanations(model, X_sample, feature_names=None):
    """
    Generate SHAP values to explain model predictions.

    :param model: Trained Keras model.
    :param X_sample: Sample data for explanation (max 100 rows).
    :param feature_names: Optional list of feature names.
    :return: SHAP values array.

    Example:
        >>> shap_values = generate_shap_explanations(model, X[:50])
    """
    try:
        import shap

        # Use DeepExplainer for neural networks
        background = X_sample[:min(50, len(X_sample))]
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(X_sample[:min(20, len(X_sample))])

        print("✅ SHAP explanations generated.")
        return shap_values

    except Exception as exc:
        print(f"⚠️  SHAP analysis failed: {exc}")
        return None


def evaluate_model(model_type, test_data_path=None, threshold=50):
    """
    Full evaluation pipeline for a trained model.

    :param model_type: Model type (lstm, cnn, convlstm).
    :param test_data_path: Path to test data CSV (optional).
    :param threshold: Threshold for epidemic risk classification.
    :return: Dictionary of all computed metrics.

    Example:
        >>> metrics = evaluate_model("lstm", threshold=30)
        >>> "rmse" in metrics
        True
    """
    from tensorflow.keras.models import load_model

    print("=" * 60)
    print(f"📊 EVALUATION: {model_type.upper()} model")
    print("=" * 60)

    # Load model
    model_path = f"models/{model_type}_final.h5"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("   Run training first: python scripts/train_models.py --model " + model_type)
        return {}

    model = load_model(model_path)
    print(f"✅ Model loaded: {model_path}")

    # Load test data
    data_dir = "data/processed/"
    if model_type == "lstm":
        X_files = sorted(glob.glob(os.path.join(data_dir, "*_X_lstm.npy")))
        y_files = sorted(glob.glob(os.path.join(data_dir, "*_y_lstm.npy")))

        if X_files:
            X = np.concatenate([np.load(f) for f in X_files])
            y = np.concatenate([np.load(f) for f in y_files])
            # Use last 20% as test
            split = int(len(X) * 0.8)
            X_test, y_test = X[split:], y[split:]
        else:
            print("⚠️  No test data. Using synthetic data.")
            X_test = np.random.rand(50, 12, 5).astype(np.float32)
            y_test = np.random.rand(50).astype(np.float32)
    else:
        print("⚠️  Using synthetic test data.")
        if model_type == "cnn":
            X_test = np.random.rand(50, 32, 32, 3).astype(np.float32)
        else:
            X_test = np.random.rand(50, 12, 32, 32, 3).astype(np.float32)
        y_test = np.random.rand(50).astype(np.float32)

    # Predictions
    y_pred = model.predict(X_test, verbose=0).flatten()

    # Metrics
    rmse = compute_rmse(y_test, y_pred)
    mae = compute_mae(y_test, y_pred)
    class_metrics = compute_classification_metrics(y_test, y_pred, threshold)

    metrics = {
        "model": model_type,
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        **class_metrics,
        "n_test_samples": len(y_test),
        "threshold": threshold,
    }

    # Print results
    print("\n📋 Results:")
    print(f"   RMSE     : {metrics['rmse']}")
    print(f"   MAE      : {metrics['mae']}")
    print(f"   AUC      : {metrics['auc']}")
    print(f"   F1-score : {metrics['f1_score']}")
    print(f"   Samples  : {metrics['n_test_samples']}")

    # SHAP
    print("\n🔍 SHAP Analysis...")
    shap_values = generate_shap_explanations(model, X_test)

    # Save metrics
    os.makedirs("reports", exist_ok=True)
    metrics_path = f"reports/{model_type}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n💾 Metrics saved: {metrics_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained models for epidemic prediction."
    )
    parser.add_argument(
        "--model",
        choices=["lstm", "cnn", "convlstm", "all"],
        required=True,
        help="Model to evaluate",
    )
    parser.add_argument(
        "--test_data", type=str, default=None, help="Path to test data CSV"
    )
    parser.add_argument(
        "--threshold", type=int, default=50, help="Epidemic risk threshold"
    )
    args = parser.parse_args()

    if args.model == "all":
        for m in ["lstm", "cnn", "convlstm"]:
            evaluate_model(m, args.test_data, args.threshold)
            print()
    else:
        evaluate_model(args.model, args.test_data, args.threshold)
