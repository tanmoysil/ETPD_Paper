#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
High-performance binary classification using Random Forest with parallel processing.
This implementation features recursive feature elimination, Bayesian hyperparameter
optimization with parallel cross-validation, and comprehensive performance evaluation.

The parallel processing design enables efficient utilization of multi-core systems,
significantly reducing training and optimization time for large-scale datasets.

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
from joblib import Parallel, delayed
import pandas as pd
from time import time

# Machine learning imports
from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    StratifiedKFold
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
    X_train : ndarray
        Training feature matrix (standardized)
    X_test : ndarray
        Test feature matrix (standardized)
    y_train : ndarray
        Training labels
    y_test : ndarray
        Test labels
    scaler : StandardScaler
        Fitted scaler for future transformations
    """
    print("Loading and preprocessing data...")
    start_time = time()

    # Load data from .mat file
    mat_file = loadmat(filepath)
    X, y = mat_file['X'], mat_file['y']
    y = y.ravel()  # Flatten label vector

    # Display dataset characteristics
    n_samples, n_features = X.shape
    class_distribution = np.bincount(y)
    pos_rate = class_distribution[1] / n_samples if len(class_distribution) > 1 else 0

    print(f"Dataset loaded: {n_samples} samples with {n_features} features")
    print(f"Class distribution: {class_distribution}")
    print(f"Positive class rate: {pos_rate:.2%}")

    # Split data ensuring class balance preservation with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Standardize features to zero mean and unit variance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Data preprocessing completed in {time() - start_time:.2f} seconds")

    return X_train, X_test, y_train, y_test, scaler


def select_features(X_train, y_train):
    """
    Perform recursive feature elimination with cross-validation (RFECV)
    to select the optimal feature subset.

    Parameters:
    -----------
    X_train : ndarray
        Training feature matrix
    y_train : ndarray
        Training labels

    Returns:
    --------
    X_train_selected : ndarray
        Reduced feature matrix containing only selected features
    rfecv : RFECV
        Fitted feature selector for future transformations
    """
    print("Performing recursive feature elimination with cross-validation...")
    start_time = time()

    # Initialize RandomForest classifier for feature selection
    base_model = RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1  # Use all available cores
    )

    # Configure RFECV with stratified k-fold CV and ROC AUC scoring
    rfecv = RFECV(
        estimator=base_model,
        step=1,  # Remove one feature at a time
        cv=StratifiedKFold(10),
        scoring='roc_auc',
        n_jobs=-1,  # Use all available cores
        verbose=1
    )

    # Fit RFECV to identify optimal feature subset
    rfecv.fit(X_train, y_train)

    # Transform data to include only selected features
    X_train_selected = rfecv.transform(X_train)

    # Get indices of selected features
    feature_indices = np.where(rfecv.support_)[0]

    print(f"Feature selection complete: {rfecv.n_features_} of {X_train.shape[1]} features selected")
    print(f"Optimal number of features determined by RFECV: {rfecv.n_features_}")
    print(f"Feature selection completed in {time() - start_time:.2f} seconds")

    # Plot number of features vs. CV score
    plt.figure(figsize=(10, 6))
    plt.xlabel("Number of features selected")
    plt.ylabel("Mean ROC AUC (CV)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.axvline(x=rfecv.n_features_, color='r', linestyle='--',
                label=f'Optimal number: {rfecv.n_features_}')
    plt.title("Feature Selection: CV Score vs Number of Features")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("rf_feature_selection.png", dpi=300, bbox_inches="tight")
    plt.close()

    return X_train_selected, rfecv


def optimize_hyperparameters(X_train_selected, y_train):
    """
    Perform Bayesian optimization to find optimal hyperparameters
    for the Random Forest classifier, using parallel processing for
    cross-validation evaluation.

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
    print("Optimizing hyperparameters using Bayesian optimization with parallel processing...")
    start_time = time()

    # Define hyperparameter search space
    param_bounds = {
        'n_estimators': (50, 300),  # Number of trees
        'max_depth': (3, 20),  # Maximum tree depth
        'min_samples_split': (2, 20),  # Minimum samples to split an internal node
        'min_samples_leaf': (1, 20),  # Minimum samples in a leaf node
        'max_features': (0.1, 1.0)  # Fraction of features to consider for best split
    }

    # Define objective function for Bayesian optimization with parallel CV
    def rf_evaluate(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features):
        """Objective function to maximize ROC AUC through parallel cross-validation"""
        params = {
            'n_estimators': int(n_estimators),
            'max_depth': int(max_depth),
            'min_samples_split': int(min_samples_split),
            'min_samples_leaf': int(min_samples_leaf),
            'max_features': max_features,
            'n_jobs': 1  # Set to 1 for parallel CV to avoid nested parallelism
        }

        # Initialize model with current hyperparameter set
        model = RandomForestClassifier(**params)

        # Setup repeated stratified k-fold cross-validation
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)

        # Define function for parallel execution
        def _fit_and_score(train_idx, val_idx):
            """Train and evaluate model on a single CV fold"""
            X_train_cv, X_val_cv = X_train_selected[train_idx], X_train_selected[val_idx]
            y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]

            # Train model on training fold
            model.fit(X_train_cv, y_train_cv)

            # Evaluate on validation fold
            y_val_pred = model.predict_proba(X_val_cv)[:, 1]
            return roc_auc_score(y_val_cv, y_val_pred)

        # Execute cross-validation in parallel
        scores = Parallel(n_jobs=-1)(
            delayed(_fit_and_score)(train_idx, val_idx)
            for train_idx, val_idx in cv.split(X_train_selected, y_train)
        )

        return np.mean(scores)

    # Initialize and run Bayesian optimization
    optimizer = BayesianOptimization(
        f=rf_evaluate,
        pbounds=param_bounds,
        verbose=2  # More detailed logging
    )

    # Maximize the objective function (ROC AUC)
    optimizer.maximize(init_points=10, n_iter=50)

    # Extract and convert best parameters
    best_params = optimizer.max['params']
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['min_samples_split'] = int(best_params['min_samples_split'])
    best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])
    best_params['n_jobs'] = -1  # Use all cores for final model

    print(f"Optimization complete in {time() - start_time:.2f} seconds")
    print(f"Best parameters found: {best_params}")
    print(f"Best CV score (ROC AUC): {optimizer.max['target']:.6f}")

    # Store optimization results for later analysis
    opt_results = pd.DataFrame(optimizer.res)
    opt_results.to_csv('bayesian_opt_results.csv', index=False)

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
    best_model : RandomForestClassifier
        Trained model with optimal parameters
    metrics : dict
        Performance metrics
    """
    print("Training final model and evaluating performance...")
    start_time = time()

    # Apply feature selection to test data
    X_test_selected = rfecv.transform(X_test)

    # Initialize and train the best model
    best_model = RandomForestClassifier(**best_params)
    best_model.fit(X_train_selected, y_train)

    training_time = time() - start_time
    print(f"Model training completed in {training_time:.2f} seconds")

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
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'confusion_matrix': test_confusion_matrix,
        'training_time': training_time
    }

    # Calculate ROC AUC with 95% CI using bootstrap
    auc_mean, auc_ci_lower, auc_ci_upper = auc_ci_bootstrap(y_test, y_test_proba)
    metrics['roc_auc_ci'] = (auc_ci_lower, auc_ci_upper)

    # Print evaluation metrics
    print("\nTest Set Performance Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall (Sensitivity): {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f} ({auc_ci_lower:.4f} - {auc_ci_upper:.4f})")
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
    plt.savefig("rf_roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Visualize feature importances
    visualize_feature_importance(best_model, rfecv)

    return best_model, metrics


def auc_ci_bootstrap(y_true, y_pred, n_bootstraps=1000):
    """
    Calculate ROC AUC with 95% confidence interval using bootstrap.
    This implementation uses parallel processing for faster computation.

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

    # Define function for parallel bootstrap iterations
    def _bootstrap_auc(i):
        # Sample with replacement
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        bootstrap_true = y_true[indices]
        bootstrap_pred = y_pred[indices]

        # Skip if only one class is present in the bootstrap sample
        if len(np.unique(bootstrap_true)) < 2:
            return None

        # Calculate ROC AUC for the bootstrap sample
        try:
            fpr, tpr, _ = roc_curve(bootstrap_true, bootstrap_pred)
            bootstrap_auc = auc(fpr, tpr)
            return bootstrap_auc
        except:
            return None

    # Execute bootstrap iterations in parallel
    results = Parallel(n_jobs=-1)(
        delayed(_bootstrap_auc)(i) for i in range(n_bootstraps)
    )

    # Filter out None values
    aucs = [result for result in results if result is not None]

    # Calculate 95% confidence interval
    sorted_aucs = np.array(sorted(aucs))
    ci_lower = sorted_aucs[int(0.025 * len(sorted_aucs))]
    ci_upper = sorted_aucs[int(0.975 * len(sorted_aucs))]

    print(f"Bootstrap completed in {time() - start_time:.2f} seconds")

    return np.mean(aucs), ci_lower, ci_upper


def visualize_feature_importance(model, rfecv, top_n=20):
    """
    Visualize feature importance from the trained Random Forest model.

    Parameters:
    -----------
    model : RandomForestClassifier
        Trained model
    rfecv : RFECV
        Fitted feature selector that contains feature support information
    top_n : int
        Number of top features to visualize
    """
    # Get feature importances
    importances = model.feature_importances_

    # Get original feature indices that were selected
    feature_indices = np.where(rfecv.support_)[0]

    # Map importances to original feature indices
    feature_importance = [(idx, imp) for idx, imp in zip(feature_indices, importances)]

    # Sort by importance (descending)
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    # Select top N features
    top_features = feature_importance[:top_n]

    # Create dataframe for plotting
    df = pd.DataFrame(top_features, columns=['Feature Index', 'Importance'])

    # Plot feature importances
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(df)), df['Importance'], align='center')
    plt.yticks(range(len(df)), [f'Feature {idx}' for idx in df['Feature Index']])
    plt.xlabel('Importance')
    plt.title('Top Feature Importances (Random Forest)')
    plt.tight_layout()
    plt.savefig("rf_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Save feature importance to CSV
    df.to_csv('rf_feature_importance.csv', index=False)

    # Print top features
    print("\nTop 10 most important features:")
    for i, (idx, imp) in enumerate(top_features[:10]):
        print(f"{i + 1}. Feature {idx}: {imp:.4f}")


def save_model(model, rfecv, scaler, metrics, best_params, output_dir='./models'):
    """
    Save the trained model and associated components for future use.

    Parameters:
    -----------
    model : RandomForestClassifier
        Trained classifier
    rfecv : RFECV
        Fitted feature selector
    scaler : StandardScaler
        Fitted data scaler
    metrics : dict
        Performance metrics
    best_params : dict
        Best hyperparameters
    output_dir : str
        Directory to save model files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save model components
    joblib.dump(model, os.path.join(output_dir, 'random_forest_model.joblib'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
    joblib.dump(rfecv, os.path.join(output_dir, 'feature_selector.joblib'))

    # Save feature mask
    np.save(os.path.join(output_dir, 'feature_mask.npy'), rfecv.support_)

    # Save model configuration as JSON
    import json
    model_config = {
        'hyperparameters': best_params,
        'feature_count': {
            'original': rfecv.n_features_in_,
            'selected': rfecv.n_features_
        },
        'selected_feature_indices': np.where(rfecv.support_)[0].tolist(),
        'performance': {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)}
    }

    # Handle non-serializable values
    if 'confusion_matrix' in model_config['performance']:
        model_config['performance']['confusion_matrix'] = model_config['performance']['confusion_matrix'].tolist()
    if 'roc_auc_ci' in model_config['performance']:
        model_config['performance']['roc_auc_ci'] = list(model_config['performance']['roc_auc_ci'])

    with open(os.path.join(output_dir, 'model_config.json'), 'w') as f:
        json.dump(model_config, f, indent=4)

    # Save performance metrics to text file
    with open(os.path.join(output_dir, 'model_performance.txt'), 'w') as f:
        f.write("Random Forest Classification Model Performance\n")
        f.write("===========================================\n\n")
        f.write(f"Original features: {rfecv.n_features_in_}\n")
        f.write(f"Selected features: {rfecv.n_features_}\n\n")
        f.write("Best Hyperparameters:\n")
        for param, value in best_params.items():
            f.write(f"  {param}: {value}\n")
        f.write("\nPerformance Metrics:\n")
        for metric, value in metrics.items():
            if metric == 'confusion_matrix':
                f.write(f"\nConfusion Matrix:\n{value}\n")
            elif metric == 'roc_auc_ci':
                f.write(f"ROC AUC 95% CI: ({value[0]:.4f} - {value[1]:.4f})\n")
            elif metric == 'training_time':
                f.write(f"Training Time: {value:.2f} seconds\n")
            else:
                f.write(f"{metric.replace('_', ' ').title()}: {value:.4f}\n")

    print(f"Model and components saved to {output_dir}")


def main():
    """Main function to execute the complete classification pipeline."""
    # Record total execution time
    pipeline_start_time = time()

    # Define file paths
    data_path = './data/X_y.mat'  # Update this path as needed

    # Step 1: Load and preprocess data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(data_path)

    # Step 2: Perform feature selection
    X_train_selected, rfecv = select_features(X_train, y_train)

    # Step 3: Optimize hyperparameters
    best_params = optimize_hyperparameters(X_train_selected, y_train)

    # Step 4: Train and evaluate the final model
    best_model, metrics = train_and_evaluate_model(
        X_train_selected, X_test, y_train, y_test, rfecv, best_params
    )

    # Step 5: Save the model and components
    save_model(best_model, rfecv, scaler, metrics, best_params)

    # Report total execution time
    total_time = time() - pipeline_start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal pipeline execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print("Random Forest classification pipeline completed successfully.")


if __name__ == "__main__":
    main()