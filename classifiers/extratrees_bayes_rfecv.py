#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A robust machine learning pipeline for binary classification using ExtraTrees.
Features recursive feature elimination with cross-validation (RFECV) and
Bayesian optimization for hyperparameter tuning to maximize model performance.

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

# Machine learning imports
from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
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
from sklearn.feature_selection import RFECV
from bayes_opt import BayesianOptimization


def load_and_preprocess_data(filepath):
    """
    Load data from a .mat file and preprocess it for machine learning.

    Parameters:
    -----------
    filepath : str
        Path to the .mat file containing features (X) and labels (y)

    Returns:
    --------
    X_train, X_test, y_train, y_test : ndarrays
        Preprocessed and scaled training and testing datasets
    scaler : StandardScaler
        Fitted scaler for future transformations
    """
    print("Loading and preprocessing data...")

    # Load data from .mat file
    mat_file = loadmat(filepath)
    X, y = mat_file['X'], mat_file['y']
    y = y.ravel()  # Flatten label vector

    # Split data ensuring class balance preservation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )

    # Standardize features to zero mean and unit variance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")

    return X_train, X_test, y_train, y_test, scaler


def perform_feature_selection(X_train, y_train):
    """
    Perform recursive feature elimination with cross-validation
    to select the optimal feature subset.

    Parameters:
    -----------
    X_train : ndarray
        Training feature matrix
    y_train : ndarray
        Training labels

    Returns:
    --------
    X_train_selected, X_test_selected : ndarrays
        Reduced feature matrices containing only selected features
    rfecv : RFECV
        Fitted feature selector for future transformations
    """
    print("Performing recursive feature elimination with cross-validation...")

    # Initialize base classifier for feature selection
    base_model = ExtraTreesClassifier(random_state=42)

    # Configure RFECV with stratified k-fold CV and ROC AUC scoring
    rfecv = RFECV(
        estimator=base_model,
        step=1,
        cv=StratifiedKFold(n_splits=10, shuffle=True),
        scoring='roc_auc',
        n_jobs=-1  # Use all available cores
    )

    # Fit RFECV to identify optimal feature subset
    rfecv.fit(X_train, y_train)

    # Transform data to include only selected features
    X_train_selected = rfecv.transform(X_train)

    print(f"Feature selection complete: {rfecv.n_features_} of {X_train.shape[1]} features selected")

    # Plot number of features vs. CV score
    plt.figure(figsize=(10, 6))
    plt.xlabel("Number of features selected")
    plt.ylabel("Mean ROC AUC (CV)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.title("Feature Selection: CV Score vs Number of Features")
    plt.savefig("feature_selection_results.png", dpi=300, bbox_inches="tight")

    return X_train_selected, rfecv


def optimize_hyperparameters(X_train_selected, y_train):
    """
    Perform Bayesian optimization to find optimal hyperparameters
    for the ExtraTrees classifier.

    Parameters:
    -----------
    X_train_selected : ndarray
        Feature-selected training data
    y_train : ndarray
        Training labels

    Returns:
    --------
    best_params : dict
        Optimized hyperparameters
    """
    print("Optimizing hyperparameters using Bayesian optimization...")

    # Define hyperparameter search space
    param_bounds = {
        'n_estimators': (50, 300),  # Number of trees
        'max_depth': (3, 20),  # Maximum tree depth
        'min_samples_split': (2, 20),  # Minimum samples to split an internal node
        'min_samples_leaf': (1, 20),  # Minimum samples in a leaf node
        'max_features': (0.1, 1.0)  # Fraction of features to consider for best split
    }

    # Define objective function for Bayesian optimization
    def et_evaluate(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features):
        """Objective function to maximize ROC AUC through cross-validation"""
        params = {
            'n_estimators': int(n_estimators),
            'max_depth': int(max_depth),
            'min_samples_split': int(min_samples_split),
            'min_samples_leaf': int(min_samples_leaf),
            'max_features': max_features,
            'random_state': 42
        }

        # Initialize model with current hyperparameter set
        model = ExtraTreesClassifier(**params)

        # Setup repeated stratified k-fold cross-validation
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
        scores = []

        # Perform cross-validation for current hyperparameter set
        for train_idx, val_idx in cv.split(X_train_selected, y_train):
            X_train_cv, X_val_cv = X_train_selected[train_idx], X_train_selected[val_idx]
            y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]

            # Train model and evaluate performance
            model.fit(X_train_cv, y_train_cv)
            y_val_pred = model.predict_proba(X_val_cv)[:, 1]
            score = roc_auc_score(y_val_cv, y_val_pred)
            scores.append(score)

        return np.mean(scores)

    # Initialize and run Bayesian optimization
    optimizer = BayesianOptimization(
        f=et_evaluate,
        pbounds=param_bounds,
        verbose=2,
        random_state=42
    )

    # Maximize the objective function (ROC AUC)
    optimizer.maximize(init_points=10, n_iter=50)

    # Extract and convert best parameters
    best_params = optimizer.max['params']
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['min_samples_split'] = int(best_params['min_samples_split'])
    best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])
    best_params['random_state'] = 42

    print(f"Optimization complete. Best parameters found: {best_params}")
    print(f"Best CV score: {optimizer.max['target']:.4f}")

    return best_params


def train_and_evaluate_model(X_train_selected, X_test, y_train, y_test, rfecv, best_params):
    """
    Train the final model with optimal parameters and evaluate on test data.

    Parameters:
    -----------
    X_train_selected : ndarray
        Feature-selected training data
    X_test : ndarray
        Test feature matrix (before feature selection)
    y_train : ndarray
        Training labels
    y_test : ndarray
        Test labels
    rfecv : RFECV
        Fitted feature selector
    best_params : dict
        Optimized hyperparameters

    Returns:
    --------
    best_model : ExtraTreesClassifier
        Trained model with optimal parameters
    """
    print("Training final model and evaluating performance...")

    # Apply feature selection to test data
    X_test_selected = rfecv.transform(X_test)

    # Initialize and train the best model
    best_model = ExtraTreesClassifier(**best_params)
    best_model.fit(X_train_selected, y_train)

    # Generate predictions on test set
    y_test_proba = best_model.predict_proba(X_test_selected)[:, 1]

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)

    # Set classification threshold (can be adjusted based on application needs)
    custom_threshold = 0.5
    y_test_pred = (y_test_proba >= custom_threshold).astype(int)

    # Calculate comprehensive evaluation metrics
    test_confusion_matrix = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = test_confusion_matrix.ravel()

    metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1': f1_score(y_test, y_test_pred),
        'roc_auc': roc_auc_score(y_test, y_test_proba),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
        'cohen_kappa': cohen_kappa_score(y_test, y_test_pred),
        'specificity': tn / (tn + fp)
    }

    # Calculate ROC AUC with 95% CI using bootstrap
    auc_mean, auc_ci_lower, auc_ci_upper = auc_ci_bootstrap(y_test, y_test_proba)
    metrics['roc_auc_ci'] = (auc_ci_lower, auc_ci_upper)

    # Print evaluation metrics
    print("\nTest Set Performance Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall (Sensitivity): {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f} ({metrics['roc_auc_ci'][0]:.4f} - {metrics['roc_auc_ci'][1]:.4f})")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
    print(f"Confusion Matrix:\n{test_confusion_matrix}")

    # Visualize ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {metrics["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig("roc_curve.png", dpi=300, bbox_inches="tight")

    # Calculate feature importances
    feature_indices = np.where(rfecv.support_)[0]
    feature_importances = best_model.feature_importances_

    # Plot feature importances (top 15)
    if len(feature_indices) > 0:
        # Sort features by importance
        sorted_idx = np.argsort(feature_importances)[::-1]
        plt.figure(figsize=(10, 8))
        n_features_to_plot = min(15, len(feature_indices))
        plt.barh(range(n_features_to_plot),
                 feature_importances[sorted_idx[:n_features_to_plot]],
                 align='center')
        plt.yticks(range(n_features_to_plot),
                   [f"Feature {feature_indices[i]}" for i in sorted_idx[:n_features_to_plot]])
        plt.xlabel('Feature Importance')
        plt.title('Top Feature Importances')
        plt.tight_layout()
        plt.savefig("feature_importances.png", dpi=300, bbox_inches="tight")

    return best_model, metrics


def auc_ci_bootstrap(y_true, y_pred, n_bootstraps=1000, random_state=42):
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
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    auc_mean, ci_lower, ci_upper : float
        Mean AUC and 95% confidence interval bounds
    """
    np.random.seed(random_state)
    aucs = []

    # Bootstrap sampling
    for _ in range(n_bootstraps):
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


def save_model(model, scaler, rfecv, metrics, output_dir='./models'):
    """
    Save the trained model and associated components for future use.

    Parameters:
    -----------
    model : ExtraTreesClassifier
        Trained classifier
    scaler : StandardScaler
        Fitted data scaler
    rfecv : RFECV
        Fitted feature selector
    metrics : dict
        Performance metrics
    output_dir : str
        Directory to save model files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save model components
    joblib.dump(model, os.path.join(output_dir, 'extra_trees_model.joblib'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
    joblib.dump(rfecv, os.path.join(output_dir, 'feature_selector.joblib'))

    # Save performance metrics to text file
    with open(os.path.join(output_dir, 'model_performance.txt'), 'w') as f:
        f.write("Model Performance Metrics\n")
        f.write("========================\n\n")
        for metric, value in metrics.items():
            if metric == 'roc_auc_ci':
                f.write(f"ROC AUC 95% CI: ({value[0]:.4f} - {value[1]:.4f})\n")
            else:
                f.write(f"{metric.replace('_', ' ').title()}: {value:.4f}\n")

    print(f"Model and components saved to {output_dir}")


def main():
    """Main function to execute the complete classification pipeline."""

    # Define file paths
    data_path = './data/X_y.mat'  # Update this path as needed

    # Step 1: Load and preprocess data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(data_path)

    # Step 2: Perform feature selection
    X_train_selected, rfecv = perform_feature_selection(X_train, y_train)

    # Step 3: Optimize hyperparameters
    best_params = optimize_hyperparameters(X_train_selected, y_train)

    # Step 4: Train and evaluate the final model
    best_model, metrics = train_and_evaluate_model(
        X_train_selected, X_test, y_train, y_test, rfecv, best_params
    )

    # Step 5: Save the model and components
    save_model(best_model, scaler, rfecv, metrics)

    print("Classification pipeline completed successfully.")


if __name__ == "__main__":
    main()