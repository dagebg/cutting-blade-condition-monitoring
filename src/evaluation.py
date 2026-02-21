"""
Model evaluation utilities for the cutting blade condition monitoring project.

This module bundles reusable functions to assess trained models on the
engineered feature tables.

Responsibilities include:
- Computing standard metrics (e.g. accuracy, F1, ROC-AUC, regression scores).
- Creating confusion matrices and simple diagnostic plots or tables.
- Running cross-validation or train/validation/test evaluations.
- Collecting and formatting results for comparison between models.

The goal is to keep evaluation logic consistent and centralized, so different
models and experiments can be compared in a reliable way.
"""
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from scipy import stats


def evaluate_mode_classifier_folds(
    predictions_classification: List[Dict[str, Any]],
) -> None:
    """Full evaluation of the mode classifier over all CV folds.

      1. Prints per-fold metrics (accuracy, precision, recall, F1-score)
         based on the provided prediction results.
      2. Aggregates metrics across folds and prints mean and standard
         deviation for each metric.
      3. Computes and plots the average confusion matrix using seaborn.
      4. Prints a summary including number of folds, best and worst F1-score,
         and a simple stability assessment based on the F1-score standard
         deviation.

    Args:
        predictions_classification: List of dictionaries, one per fold, each
            containing:
              - 'fold': Fold index starting at 1.
              - 'y_val': Validation target values for that fold (pd.Series).
              - 'y_pred': Predicted mode labels for that fold (np.ndarray).

    Returns:
        None. All results are printed to stdout and the confusion matrix is
        shown as a matplotlib plot.
    """
    print("FULL MODEL EVALUATION for Mode Classifier")
    print("=" * 60)

    # lists for all metrics
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_f1_scores = []
    all_conf_matrices = []

    # evaluation for saved prediction results
    for fold_result in predictions_classification:
        print(f"\nFOLD {fold_result['fold']}")
        print("-" * 30)

        y_val = fold_result["y_val"]
        y_pred = fold_result["y_pred"]

        # calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_val, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_val, y_pred)

        all_accuracies.append(accuracy)
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1_scores.append(f1)
        all_conf_matrices.append(cm)

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")

    # ================================
    # RESULTS (AVERAGE OVER ALL FOLDS)
    # ================================
    print("\n" + "=" * 60)
    print("RESULTS (AVERAGE OVER ALL FOLDS)")
    print("=" * 60)

    mean_accuracy = np.mean(all_accuracies)
    mean_precision = np.mean(all_precisions)
    mean_recall = np.mean(all_recalls)
    mean_f1 = np.mean(all_f1_scores)

    std_accuracy = np.std(all_accuracies)
    std_precision = np.std(all_precisions)
    std_recall = np.std(all_recalls)
    std_f1 = np.std(all_f1_scores)

    print(f"Accuracy:  {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Precision: {mean_precision:.4f} ± {std_precision:.4f}")
    print(f"Recall:    {mean_recall:.4f} ± {std_recall:.4f}")
    print(f"F1-Score:  {mean_f1:.4f} ± {std_f1:.4f}")

    # ================================
    # CONFUSION MATRIX (AVERAGE)
    # ================================
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX (AVERAGE)")
    print("=" * 60)

    mean_cm = np.mean(all_conf_matrices, axis=0)

    plt.figure(figsize=(8, 6))
    sns.heatmap(mean_cm, annot=True, fmt=".1f", cmap="Blues")
    plt.title("Average Confusion Matrix")
    plt.xlabel("Predicted mode")
    plt.ylabel("Actual mode")
    plt.tight_layout()
    plt.show()

    # ================================
    # SUMMARY
    # ================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Number of folds: {len(all_accuracies)}")
    print(f"Best F1-score:   {max(all_f1_scores):.4f}")
    print(f"Worst F1-score:  {min(all_f1_scores):.4f}")
    print(f"Stability (std): F1-score ± {std_f1:.4f}")

    if std_f1 < 0.01:
        print("\nVery stable results across all folds!")
    elif std_f1 < 0.05:
        print("\nStable results across all folds!")
    else:
        print("\nVarying results across the folds!")


def evaluate_ridge_regression_folds(
    predictions_ridge: List[Dict[str, Any]],
) -> None:
    """
    Full evaluation of the ridge regression model over all CV folds.

    Parameters
    ----------
    predictions_ridge : list of dict
        One dict per fold with keys:
        - 'fold' : int, fold index starting at 1
        - 'y_val': validation targets (pd.Series or np.ndarray)
        - 'y_pred': predicted values (np.ndarray)

    Returns
    -------
    None
        Prints metrics and shows residual plots.
    """
    print("REGRESSION EVALUATION")
    print("=" * 60)

    # lists for all metrics
    all_maes: List[float] = []
    all_mses: List[float] = []
    all_rmses: List[float] = []
    all_r2_scores: List[float] = []
    all_mapes: List[float] = []
    all_residuals: List[float] = []

    # evaluation for saved prediction results
    for fold_result in predictions_ridge:
        print(f"\nFOLD {fold_result['fold']}")
        print("-" * 30)

        y_val = fold_result["y_val"]
        y_pred = fold_result["y_pred"]

        # calculate metrics
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_pred)
        mape = (
            mean_absolute_percentage_error(y_val, y_pred) * 100
            if np.all(y_val != 0)
            else np.nan
        )

        residuals = y_val - y_pred

        all_maes.append(mae)
        all_mses.append(mse)
        all_rmses.append(rmse)
        all_r2_scores.append(r2)
        all_mapes.append(mape)
        all_residuals.extend(residuals)

        print(f"MAE:   {mae:.4f}")
        print(f"MSE:   {mse:.4f}")
        print(f"RMSE:  {rmse:.4f}")
        print(f"R²:    {r2:.4f}")
        if not np.isnan(mape):
            print(f"MAPE:  {mape:.2f}%")
        print(f"Residuals Range: " f"[{residuals.min():.4f}, {residuals.max():.4f}]")

    # ================================
    # RESULTS (AVERAGE OVER ALL FOLDS)
    # ================================
    print("\n" + "=" * 60)
    print("RESULTS (AVERAGE OVER ALL FOLDS)")
    print("=" * 60)

    mean_mae = np.mean(all_maes)
    mean_mse = np.mean(all_mses)
    mean_rmse = np.mean(all_rmses)
    mean_r2 = np.mean(all_r2_scores)
    mean_mape = np.nanmean(all_mapes)

    std_mae = np.std(all_maes)
    std_mse = np.std(all_mses)
    std_rmse = np.std(all_rmses)
    std_r2 = np.std(all_r2_scores)
    std_mape = np.nanstd(all_mapes)

    print(f"MAE:   {mean_mae:.4f} ± {std_mae:.4f}")
    print(f"MSE:   {mean_mse:.4f} ± {std_mse:.4f}")
    print(f"RMSE:  {mean_rmse:.4f} ± {std_rmse:.4f}")
    print(f"R²:    {mean_r2:.4f} ± {std_r2:.4f}")
    if not np.isnan(mean_mape):
        print(f"MAPE:  {mean_mape:.2f}% ± {std_mape:.2f}%")

    # ================================
    # RESIDUALS-ANALYSIS
    # ================================
    print("\n" + "=" * 60)
    print("RESIDUALS-ANALYSIS")
    print("=" * 60)

    residuals_array = np.array(all_residuals)
    print(f"Mean:   {np.mean(residuals_array):.4f}")
    print(f"Std:    {np.std(residuals_array):.4f}")
    print(f"Min:    {np.min(residuals_array):.4f}")
    print(f"Max:    {np.max(residuals_array):.4f}")
    print(f"Median: {np.median(residuals_array):.4f}")

    # Residuals Histogram
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.hist(residuals_array, bins=30, alpha=0.7, edgecolor="black")
    plt.title("Residuals Histogram")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")

    # Residuals vs Fitted (last fold)
    plt.subplot(2, 2, 2)
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.title("Residuals vs Fitted Values")
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")

    # Actual vs Predicted (last fold)
    plt.subplot(2, 2, 3)
    plt.scatter(y_val, y_pred, alpha=0.6)
    plt.plot(
        [y_val.min(), y_val.max()],
        [y_val.min(), y_val.max()],
        "r--",
        lw=2,
    )
    plt.title("Actual vs Predicted")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")

    # Q-Q Plot of residuals
    plt.subplot(2, 2, 4)
    stats.probplot(residuals_array, dist="norm", plot=plt)
    plt.title("Q-Q Plot of Residuals")

    plt.tight_layout()
    plt.show()

    # ================================
    # MODEL PERFORMANCE RATING
    # ================================
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE RATING")
    print("=" * 60)

    if mean_r2 >= 0.9:
        print(f"R² performance: Excellent ({mean_r2:.4f})")
    elif mean_r2 >= 0.7:
        print(f"R² performance: Good ({mean_r2:.4f})")
    elif mean_r2 >= 0.5:
        print(f"R² performance: Acceptable ({mean_r2:.4f})")
    else:
        print(f"R² performance: Poor ({mean_r2:.4f})")

    rmse_mae_ratio = mean_rmse / mean_mae
    print(f"RMSE/MAE ratio: {rmse_mae_ratio:.2f}")
    if rmse_mae_ratio > 1.5:
        print(" >> Indicates outliers in the data")
    else:
        print(" >> Normal error distribution")

    # ================================
    # SUMMARY
    # ================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Number of folds: {len(predictions_ridge)}")
    print(f"Best R²:         {max(all_r2_scores):.4f}")
    print(f"Worst R²:        {min(all_r2_scores):.4f}")
    print(f"Stability (std): R² ± {std_r2:.4f}")

    if std_r2 < 0.01:
        print("\nVery stable results across all folds!")
    elif std_r2 < 0.05:
        print("\nStable results across all folds!")
    else:
        print("\nVarying results across the folds!")


def evaluate_anomaly_detection_folds(
    predictions_anomaly: List[Dict[str, Any]],
    X_anom,
    y_anom,
    cv,
) -> None:
    """
    Full evaluation of the anomaly detector over all CV folds.

    Parameters
    ----------
    predictions_anomaly : list of dict
        One dict per fold with keys:
        - 'fold' : int, fold index starting at 1
        - 'y_val': validation targets (pd.Series or np.ndarray, binary 0/1)
        - 'y_pred': predicted labels (np.ndarray, binary 0/1)
    X_anom : pd.DataFrame
        Feature matrix used for anomaly detection (for imbalance stats).
    y_anom : pd.Series
        Binary anomaly labels aligned with X_anom.
    cv : cross-validator
        Same splitter that was used to build predictions_anomaly.

    Returns
    -------
    None
        Prints metrics and aggregated analysis.
    """
    print("ANOMALY DETECTION EVALUATION")
    print("=" * 60)

    # lists for all metrics
    all_accuracies: List[float] = []
    all_precisions: List[float] = []
    all_recalls: List[float] = []
    all_f1_scores: List[float] = []
    all_specificities: List[float] = []
    all_conf_matrices: List[np.ndarray] = []

    # evaluation for saved prediction results
    for fold_result in predictions_anomaly:
        print(f"\nFOLD {fold_result['fold']}")
        print("-" * 30)

        y_val = fold_result["y_val"]
        y_pred = fold_result["y_pred"]

        # calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        # confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        tn, fp, fn, tp = cm.ravel()

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # store
        all_accuracies.append(accuracy)
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1_scores.append(f1)
        all_specificities.append(specificity)
        all_conf_matrices.append(cm)

        # per-fold output
        print(f"Accuracy:    {accuracy:.4f}")
        print(f"Precision:   {precision:.4f}")
        print(f"Recall:      {recall:.4f}")
        print(f"F1-Score:    {f1:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print("Confusion Matrix:")
        print(f"  TN: {tn:3d}  FP: {fp:3d}")
        print(f"  FN: {fn:3d}  TP: {tp:3d}")

        anomaly_rate = y_pred.sum() / len(y_pred) * 100
        true_anomaly_rate = y_val.sum() / len(y_val) * 100
        print(
            f"Detected Anomalies: {anomaly_rate:.1f}% "
            f"(True: {true_anomaly_rate:.1f}%)"
        )

    # ================================
    # RESULTS
    # ================================
    print("\n" + "=" * 60)
    print("RESULTS (AVERAGE OVER ALL FOLDS)")
    print("=" * 60)

    mean_accuracy = np.mean(all_accuracies)
    mean_precision = np.mean(all_precisions)
    mean_recall = np.mean(all_recalls)
    mean_f1 = np.mean(all_f1_scores)
    mean_specificity = np.mean(all_specificities)

    std_accuracy = np.std(all_accuracies)
    std_precision = np.std(all_precisions)
    std_recall = np.std(all_recalls)
    std_f1 = np.std(all_f1_scores)
    std_specificity = np.std(all_specificities)

    print(f"Accuracy:    {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Precision:   {mean_precision:.4f} ± {std_precision:.4f}")
    print(f"Recall:      {mean_recall:.4f} ± {std_recall:.4f}")
    print(f"F1-Score:    {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Specificity: {mean_specificity:.4f} ± {std_specificity:.4f}")

    # ================================
    # CONFUSION MATRIX ANALYSIS
    # ================================
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX ANALYSIS")
    print("=" * 60)

    # Aggregated Confusion Matrix
    total_cm = np.sum(all_conf_matrices, axis=0)
    total_tn, total_fp, total_fn, total_tp = total_cm.ravel()

    print("Aggregated Confusion Matrix:")
    print(f"  TN: {total_tn:3d}  FP: {total_fp:3d}")
    print(f"  FN: {total_fn:3d}  TP: {total_tp:3d}")

    # Other Metrics
    total_samples = total_tn + total_fp + total_fn + total_tp
    print(f"\nTotal Samples: {total_samples}")
    print(f"True Negatives Rate:  {total_tn / total_samples * 100:.1f}%")
    print(f"False Positives Rate: {total_fp / total_samples * 100:.1f}%")
    print(f"False Negatives Rate: {total_fn / total_samples * 100:.1f}%")
    print(f"True Positives Rate:  {total_tp / total_samples * 100:.1f}%")

    # ================================
    # MODELL-PERFORMANCE RATING
    # ================================
    print("\n" + "=" * 60)
    print("MODELL-PERFORMANCE RATING")
    print("=" * 60)

    if mean_f1 >= 0.8:
        f1_interpretation = "Excellent"
    elif mean_f1 >= 0.6:
        f1_interpretation = "Good"
    elif mean_f1 >= 0.4:
        f1_interpretation = "Acceptable"
    else:
        f1_interpretation = "Bad"

    print(f"F1-score performance: {f1_interpretation} ({mean_f1:.4f})")

    pr_balance = abs(mean_precision - mean_recall)
    print(f"Precision-Recall balance: {pr_balance:.4f}")
    if pr_balance < 0.1:
        print("  >> Balanced precision and recall")
    elif mean_precision > mean_recall:
        print("  >> Higher precision (fewer false positives)")
    else:
        print("  >> Higher recall (fewer false negatives)")

    if mean_recall > 0.7 and mean_precision > 0.7:
        detection_quality = "Very good"
    elif mean_recall > 0.5 and mean_precision > 0.5:
        detection_quality = "Good"
    else:
        detection_quality = "Needs improvement"

    print(f"Anomaly detection quality: {detection_quality}")

    # ================================
    # SUMMARY
    # ================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"Number of folds: {len(all_accuracies)}")
    print(f"Best F1-score:   {max(all_f1_scores):.4f}")
    print(f"Worst F1-score:  {min(all_f1_scores):.4f}")

    print("\nStability (standard deviation):")
    print(f"Accuracy:  {std_accuracy:.4f}")
    print(f"Precision: {std_precision:.4f}")
    print(f"Recall:    {std_recall:.4f}")
    print(f"F1-score:  {std_f1:.4f}")

    if std_f1 < 0.05:
        print("\nVery stable results across all folds!")
    elif std_f1 < 0.1:
        print("\nStable results across all folds!")
    else:
        print("\nVarying results across the folds!")

    print(f"\nModel recommendation: {f1_interpretation} for anomaly detection")

    # Class imbalance check
    total_anomalies = np.sum(
        [y_anom.iloc[val_idx].sum() for _, val_idx in cv.split(X_anom, y_anom)]
    )
    total_samples_cv = np.sum([len(val_idx) for _, val_idx in cv.split(X_anom, y_anom)])
    imbalance_ratio = total_anomalies / total_samples_cv

    print(
        f"\nClass imbalance ratio: {imbalance_ratio:.3f} "
        f"({imbalance_ratio*100:.1f}% anomalies)"
    )
    if imbalance_ratio < 0.1:
        print("Severe class imbalance – interpret results with caution!")
    elif imbalance_ratio < 0.3:
        print("Moderate class imbalance – F1-score is more important than accuracy!")
    else:
        print("Acceptable class balance")
