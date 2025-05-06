#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation script for pre-trained XGBoost model on external test data.
This script loads a trained model and selected features, applies feature selection
to the test data, and evaluates the model's performance with comprehensive metrics.

Author: Tanmoy Sil
Affiliation: UKW, Germany
Email: sil_t@ukw.de
Date: May 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import joblib
from time import time

# Machine learning imports
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score,
    cohen_kappa_score,
    roc_curve,
    auc
)


def load_data_and_model(data_path, model_path, features_path, all_features_path):
    """
    Load test data, trained model, and feature information.

    Parameters:
    -----------
    data_path : str
        Path to the .mat file containing test data
    model_path : str
        Path to the saved XGBoost model
    features_path : str
        Path to the selected feature names file
    all_features_path : str
        Path to all feature names file

    Returns:
    --------
    data_dict : dict
        Dictionary containing loaded data and model
    """
    print("Loading test data and model...")
    start_time = time()

    # Load test data
    mat_file = loadmat(data_path)
    X, y = mat_file['X'], mat_file['y']
    y = y.ravel()  # Flatten label vector

    # Load model
    model = XGBClassifier()
    model.load_model(model_path)

    # Load selected feature names
    selected_feature_names = joblib.load(features_path)

    # Load all feature names
    all_feature_names = np.genfromtxt(all_features_path, dtype=str, delimiter='\n')

    # Create mapping from feature names to indices
    feature_name_to_index = {name: idx for idx, name in enumerate(all_feature_names)}

    # Get indices of selected features
    selected_indices = [feature_name_to_index[name] for name in selected_feature_names]

    # Select features from test data
    X_selected = X[:, selected_indices]

    print(f"Test data shape: {X.shape}")
    print(f"Selected features: {len(selected_indices)} out of {X.shape[1]}")
    print(f"Data and model loading completed in {time() - start_time:.2f} seconds")

    # Create dictionary with all loaded components
    data_dict = {
        'X': X,
        'y': y,
        'X_selected': X_selected,
        'model': model,
        'selected_feature_names': selected_feature_names,
        'all_feature_names': all_feature_names,
        'selected_indices': selected_indices
    }

    return data_dict


def evaluate_model(X_selected, y, model, classification_threshold=0.8):
    """
    Evaluate model on test data with custom classification threshold.

    Parameters:
    -----------
    X_selected : ndarray
        Test data with selected features
    y : ndarray
        True labels
    model : XGBClassifier
        Loaded XGBoost model
    classification_threshold : float
        Probability threshold for binary classification

    Returns:
    --------
    metrics_dict : dict
        Dictionary containing evaluation metrics
    """
    print(f"Evaluating model with threshold {classification_threshold}...")
    start_time = time()

    # Scale the test data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    # Generate predictions
    y_proba = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_proba >= classification_threshold).astype(int)

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y, y_proba)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()

    # Calculate performance metrics
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'f1': f1_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_proba),
        'cohen_kappa': cohen_kappa_score(y, y_pred),
        'confusion_matrix': conf_matrix
    }

    # Calculate ROC AUC with 95% CI using bootstrap
    metrics['roc_auc_mean'], metrics['roc_auc_ci_lower'], metrics['roc_auc_ci_upper'] = auc_ci_bootstrap(
        y, y_proba
    )

    # Print evaluation metrics
    print("\nTest Set Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall (Sensitivity): {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f} ({metrics['roc_auc_ci_lower']:.4f} - {metrics['roc_auc_ci_upper']:.4f})")
    print(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    print(f"Evaluation completed in {time() - start_time:.2f} seconds")

    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {metrics["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # Mark the current threshold on the ROC curve
    threshold_idx = np.argmin(np.abs(thresholds - classification_threshold))
    plt.scatter(fpr[threshold_idx], tpr[threshold_idx], color='red', s=100,
                label=f'Threshold = {classification_threshold}')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig("model_evaluation_roc_curve.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Create metrics dictionary
    metrics_dict = {
        'metrics': metrics,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'roc_curve': {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }
    }

    return metrics_dict


def auc_ci_bootstrap(y_true, y_pred, n_bootstraps=1000):
    """
    Calculate ROC AUC with 95% confidence interval using bootstrap.

    Parameters:
    -----------
    y_true : ndarray
        True binary labels
    y_pred : ndarray
        Predicted probabilities
    n_bootstraps : int
        Number of bootstrap iterations

    Returns:
    --------
    auc_mean, ci_lower, ci_upper : float
        Mean AUC and 95% confidence interval bounds
    """
    print(f"Calculating ROC AUC confidence intervals using {n_bootstraps} bootstrap samples...")
    start_time = time()

    aucs = []

    # Bootstrap sampling
    for i in range(n_bootstraps):
        # Sample with replacement
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        bootstrap_true = y_true[indices]
        bootstrap_pred = y_pred[indices]

        # Skip if only one class is present in the bootstrap sample
        if len(np.unique(bootstrap_true)) < 2:
            continue

        # Calculate ROC AUC for the bootstrap sample
        try:
            fpr, tpr, _ = roc_curve(bootstrap_true, bootstrap_pred)
            bootstrap_auc = auc(fpr, tpr)
            aucs.append(bootstrap_auc)
        except:
            continue

    # Calculate 95% confidence interval
    sorted_aucs = np.array(sorted(aucs))
    ci_lower = sorted_aucs[int(0.025 * len(sorted_aucs))]
    ci_upper = sorted_aucs[int(0.975 * len(sorted_aucs))]

    print(f"Bootstrap completed in {time() - start_time:.2f} seconds")

    return np.mean(aucs), ci_lower, ci_upper


def save_results(metrics_dict, output_dir='./evaluation_results'):
    """
    Save evaluation results to files.

    Parameters:
    -----------
    metrics_dict : dict
        Dictionary containing evaluation metrics
    output_dir : str
        Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': list(metrics_dict['metrics'].keys()),
        'Value': [str(v) if isinstance(v, np.ndarray) else v for v in metrics_dict['metrics'].values()]
    })
    metrics_df.to_csv(f"{output_dir}/evaluation_metrics.csv", index=False)

    # Save detailed results to text file
    with open(f"{output_dir}/evaluation_results.txt", 'w') as f:
        f.write("Model Evaluation Results\n")
        f.write("=======================\n\n")

        f.write("Performance Metrics:\n")
        f.write("-----------------\n")
        metrics = metrics_dict['metrics']
        for metric, value in metrics.items():
            if metric == 'confusion_matrix':
                f.write(f"\nConfusion Matrix:\n{value}\n")
            elif metric == 'roc_auc_ci_lower' or metric == 'roc_auc_ci_upper':
                continue  # These are included with roc_auc
            elif metric == 'roc_auc':
                f.write(
                    f"ROC AUC: {value:.4f} ({metrics['roc_auc_ci_lower']:.4f} - {metrics['roc_auc_ci_upper']:.4f})\n")
            else:
                f.write(f"{metric.replace('_', ' ').title()}: {value:.4f}\n")

    print(f"Evaluation results saved to {output_dir}")


def main():
    """Main function to execute the model evaluation pipeline."""
    # Record execution time
    start_time = time()

    # Define file paths
    data_path = './data/X_y_test.mat'
    model_path = './models/xgboost_model.json'
    features_path = './models/best_model_vars.pkl'
    all_features_path = './data/features_all.txt'

    # Step 1: Load data and model
    data_dict = load_data_and_model(data_path, model_path, features_path, all_features_path)

    # Step 2: Evaluate model
    metrics_dict = evaluate_model(
        data_dict['X_selected'],
        data_dict['y'],
        data_dict['model'],
        classification_threshold=0.8  # Use 0.8 as threshold
    )

    # Step 3: Save results
    save_results(metrics_dict)

    # Report execution time
    total_time = time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print("Model evaluation completed successfully.")


if __name__ == "__main__":
    main()