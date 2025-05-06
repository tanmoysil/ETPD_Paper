#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Binary classification using Random Forest with comprehensive validation.
This implementation incorporates grid search hyperparameter optimization
and robust statistical analysis of model performance with confidence intervals.

Author: Tanmoy Sil
Affiliation: UKW, Germany
Email: sil_t@ukw.de
Date: May 2025
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import joblib
import pandas as pd
from time import time

# Machine learning imports
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RepeatedStratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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
    auc,
    precision_recall_curve,
    average_precision_score
)


def load_and_preprocess_data(filepath):
    """
    Load data from a .mat file and split into train and test sets.

    Parameters:
    -----------
    filepath : str
        Path to the .mat file containing features (X) and labels (y)

    Returns:
    --------
    data_splits : dict
        Dictionary containing all data splits and scaler
    """
    print("Loading and preprocessing data...")

    # Load data from .mat file
    mat_file = loadmat(filepath)
    X, y = mat_file['X'], mat_file['y']
    y = y.ravel()  # Flatten label vector

    # Display dataset statistics
    n_samples, n_features = X.shape
    class_distribution = np.bincount(y)
    print(f"Dataset loaded: {n_samples} samples with {n_features} features")
    print(f"Class distribution: {class_distribution}")

    # Split data into train and test sets (80%/20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Standardize features to zero mean and unit variance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create a dictionary with all data splits and scaler
    data_splits = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': [f'Feature_{i}' for i in range(X.shape[1])]
    }

    return data_splits


def optimize_hyperparameters(X_train, y_train):
    """
    Perform grid search to optimize Random Forest hyperparameters.

    Parameters:
    -----------
    X_train : ndarray
        Training feature matrix
    y_train : ndarray
        Training labels

    Returns:
    --------
    best_model : RandomForestClassifier
        Optimized model
    grid_results : dict
        Grid search results
    """
    print("Optimizing hyperparameters using grid search...")
    start_time = time()

    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Initialize the model
    model = RandomForestClassifier(
        n_jobs=-1,  # Use all available cores
        verbose=0,  # Suppress output messages
        class_weight='balanced'  # Handle class imbalance
    )

    # Setup cross-validation
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5)

    # Initialize and run GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',  # Optimize for ROC AUC
        n_jobs=-1,  # Use all available cores
        verbose=1,
        return_train_score=True
    )

    # Fit GridSearchCV on the training data
    grid_search.fit(X_train, y_train)

    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Compile grid search results
    cv_results = pd.DataFrame(grid_search.cv_results_)

    # Sort results by rank
    cv_results = cv_results.sort_values('rank_test_score')

    # Extract the top 5 parameter combinations
    top_params = cv_results.head(5)[['params', 'mean_test_score', 'std_test_score']]

    # Calculate execution time
    execution_time = time() - start_time

    # Compile results dictionary
    grid_results = {
        'best_params': best_params,
        'best_score': grid_search.best_score_,
        'top_params': top_params,
        'execution_time': execution_time
    }

    print(f"Optimization complete in {execution_time:.2f} seconds")
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation score (ROC AUC): {grid_search.best_score_:.6f}")

    return best_model, grid_results


def evaluate_model_performance(model, data_splits):
    """
    Comprehensive evaluation of model performance on test set.

    Parameters:
    -----------
    model : RandomForestClassifier
        Trained classifier
    data_splits : dict
        Dictionary containing all data splits

    Returns:
    --------
    metrics : dict
        Dictionary of performance metrics
    """
    print("Evaluating model performance...")

    # Extract test data
    X_test, y_test = data_splits['X_test'], data_splits['y_test']

    # Generate predictions for test set
    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)[:, 1]

    # Calculate metrics for test set
    metrics = calculate_metrics(y_test, y_pred_test, y_proba_test)

    # Print evaluation results
    print("\nTest Set Metrics:")
    print_metrics(metrics)

    # Generate ROC curve plot
    plot_roc_curve(y_test, y_proba_test)

    # Generate precision-recall curve plot
    plot_precision_recall_curve(y_test, y_proba_test)

    # Analyze feature importance
    feature_names = data_splits['feature_names']
    plot_feature_importance(model, feature_names)

    return metrics


def calculate_metrics(y_true, y_pred, y_proba):
    """
    Calculate comprehensive performance metrics.

    Parameters:
    -----------
    y_true : ndarray
        True labels
    y_pred : ndarray
        Predicted labels
    y_proba : ndarray
        Predicted probabilities for positive class

    Returns:
    --------
    metrics : dict
        Dictionary of performance metrics
    """
    # Calculate basic classification metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'average_precision': average_precision_score(y_true, y_proba)
    }

    # Calculate confusion matrix and derived metrics
    conf_matrix = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = conf_matrix

    # Extract values from confusion matrix
    tn, fp, fn, tp = conf_matrix.ravel()

    # Calculate specificity
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Calculate ROC AUC with 95% CI using bootstrap
    metrics['roc_auc_mean'], metrics['roc_auc_ci_lower'], metrics['roc_auc_ci_upper'] = auc_ci_bootstrap(y_true,
                                                                                                         y_proba)

    return metrics


def print_metrics(metrics):
    """
    Print metrics in a formatted way.

    Parameters:
    -----------
    metrics : dict
        Dictionary of performance metrics
    """
    print(f"ROC AUC: {metrics['roc_auc']:.4f} ({metrics['roc_auc_ci_lower']:.4f} - {metrics['roc_auc_ci_upper']:.4f})")
    print(f"Average Precision: {metrics['average_precision']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Recall (Sensitivity): {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")


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

    return np.mean(aucs), ci_lower, ci_upper


def plot_roc_curve(y_test, y_proba_test):
    """
    Plot ROC curve for test set.

    Parameters:
    -----------
    y_test : ndarray
        Test set true labels
    y_proba_test : ndarray
        Test set predicted probabilities
    """
    # Calculate ROC curve points
    fpr, tpr, _ = roc_curve(y_test, y_proba_test)

    # Calculate AUC value
    roc_auc = auc(fpr, tpr)

    # Create plot
    plt.figure(figsize=(10, 8))

    # Plot test set ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')

    # Plot diagonal reference line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # Set plot properties
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)

    # Save plot
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_precision_recall_curve(y_test, y_proba_test):
    """
    Plot precision-recall curve for test set.

    Parameters:
    -----------
    y_test : ndarray
        Test set true labels
    y_proba_test : ndarray
        Test set predicted probabilities
    """
    # Calculate precision-recall curve points
    precision, recall, _ = precision_recall_curve(y_test, y_proba_test)

    # Calculate average precision value
    ap = average_precision_score(y_test, y_proba_test)

    # Create plot
    plt.figure(figsize=(10, 8))

    # Plot test set precision-recall curve
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AP = {ap:.3f})')

    # Plot baseline
    plt.plot([0, 1], [np.mean(y_test), np.mean(y_test)], color='navy',
             lw=2, linestyle='--', label=f'Baseline (y_mean = {np.mean(y_test):.3f})')

    # Set plot properties
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)

    # Save plot
    plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plot feature importance.

    Parameters:
    -----------
    model : RandomForestClassifier
        Trained model
    feature_names : list
        List of feature names
    top_n : int
        Number of top features to display
    """
    # Get feature importances from model
    importances = model.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Limit to top_n features
    n_features = min(top_n, len(importances))
    indices = indices[:n_features]

    # Create plot
    plt.figure(figsize=(12, 8))

    # Plot feature importances
    plt.bar(range(n_features), importances[indices], align='center')
    plt.xticks(range(n_features), [feature_names[i] for i in indices], rotation=90)
    plt.xlim([-1, n_features])
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Feature Importance (Random Forest)')
    plt.tight_layout()

    # Save plot
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create detailed feature importance table
    feature_importance_data = {
        'Feature': [feature_names[i] for i in indices],
        'Importance': importances[indices]
    }

    # Save feature importance to CSV
    importance_df = pd.DataFrame(feature_importance_data)
    importance_df.to_csv('feature_importance.csv', index=False)


def save_model(model, data_splits, metrics, grid_results, output_dir='./models'):
    """
    Save the trained model and associated metadata.

    Parameters:
    -----------
    model : RandomForestClassifier
        Trained model
    data_splits : dict
        Dictionary containing all data splits
    metrics : dict
        Performance metrics
    grid_results : dict
        Grid search results
    output_dir : str
        Directory to save model files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the model
    joblib.dump(model, os.path.join(output_dir, 'random_forest_model.joblib'))

    # Save the scaler
    joblib.dump(data_splits['scaler'], os.path.join(output_dir, 'scaler.joblib'))

    # Save model metadata
    metadata = {
        'model_type': 'RandomForestClassifier',
        'n_features': len(data_splits['feature_names']),
        'feature_names': data_splits['feature_names'],
        'hyperparameters': model.get_params(),
        'best_grid_search_params': grid_results['best_params'],
        'grid_search_best_score': grid_results['best_score'],
        'test_metrics': {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
    }

    # Save model report
    with open(os.path.join(output_dir, 'model_report.txt'), 'w') as f:
        f.write("Random Forest Classification Model Report\n")
        f.write("=======================================\n\n")

        f.write("Dataset Information:\n")
        f.write("-----------------\n")
        f.write(f"Number of features: {len(data_splits['feature_names'])}\n")
        f.write(f"Training samples: {len(data_splits['y_train'])}\n")
        f.write(f"Test samples: {len(data_splits['y_test'])}\n\n")

        f.write("Hyperparameter Optimization:\n")
        f.write("-------------------------\n")
        f.write(f"Best parameters: {grid_results['best_params']}\n")
        f.write(f"Best CV score: {grid_results['best_score']:.6f}\n")
        f.write(f"Execution time: {grid_results['execution_time']:.2f} seconds\n\n")

        f.write("Top 5 Parameter Combinations:\n")
        f.write(str(grid_results['top_params']))
        f.write("\n\n")

        f.write("Test Set Metrics:\n")
        f.write("----------------\n")
        for metric, value in metrics.items():
            if metric == 'confusion_matrix':
                f.write(f"Confusion Matrix:\n{value}\n")
            elif metric.startswith('roc_auc_ci'):
                continue
            elif metric == 'roc_auc':
                f.write(f"ROC AUC: {value:.4f} ({metrics['roc_auc_ci_lower']:.4f} - {metrics['roc_auc_ci_upper']:.4f})\n")
            else:
                f.write(f"{metric.replace('_', ' ').title()}: {value:.4f}\n")

    print(f"Model and metadata saved to {output_dir}")


def main():
    """Main function to execute the complete classification pipeline."""
    # File path to dataset
    data_path = './data/X_y.mat'

    # Step 1: Load and preprocess data
    data_splits = load_and_preprocess_data(data_path)

    # Step 2: Optimize hyperparameters using grid search
    best_model, grid_results = optimize_hyperparameters(
        data_splits['X_train'],
        data_splits['y_train']
    )

    # Step 3: Evaluate model performance
    metrics = evaluate_model_performance(best_model, data_splits)

    # Step 4: Save model and results
    save_model(best_model, data_splits, metrics, grid_results)

    print("Random Forest classification pipeline completed successfully.")


if __name__ == "__main__":
    main()