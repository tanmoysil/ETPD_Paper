#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced binary classification pipeline using LightGBM with GPU acceleration.
Implements recursive feature elimination, Bayesian hyperparameter optimization,
and comprehensive performance evaluation with confidence intervals.

This implementation leverages GPU acceleration for high-performance training
on large-scale datasets, making it suitable for computationally intensive
classification tasks in biomedical, financial, or other domains requiring
both speed and accuracy.

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
from lightgbm import LGBMClassifier
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

    # Load data from .mat file and extract features and labels
    mat_file = loadmat(filepath)
    X, y = mat_file['X'], mat_file['y']
    y = y.ravel()  # Flatten label vector for compatibility with sklearn

    # Display dataset characteristics
    n_samples, n_features = X.shape
    class_distribution = np.bincount(y)
    print(f"Dataset loaded: {n_samples} samples with {n_features} features")
    print(f"Class distribution: {class_distribution}")

    # Split data ensuring class balance preservation with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Standardize features to zero mean and unit variance
    # This is important for proper functioning of many machine learning algorithms
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def perform_feature_selection(X_train, y_train, use_gpu=True):
    """
    Perform recursive feature elimination with cross-validation (RFECV)
    to select the optimal feature subset.

    Parameters:
    -----------
    X_train : ndarray
        Training feature matrix
    y_train : ndarray
        Training labels
    use_gpu : bool
        Whether to use GPU acceleration for LightGBM

    Returns:
    --------
    X_train_selected : ndarray
        Reduced feature matrix containing only selected features
    rfecv : RFECV
        Fitted feature selector for future transformations
    """
    print("Performing recursive feature elimination with cross-validation...")

    # Initialize base LightGBM classifier for feature selection
    # Using GPU acceleration if available and requested
    base_model = LGBMClassifier(
        device='gpu' if use_gpu else 'cpu',
        objective='binary',
        n_jobs=-1,  # Use all available cores for CPU operations
        verbose=-1  # Suppress LightGBM output messages
    )

    # Configure RFECV with stratified k-fold CV and ROC AUC scoring
    # ROC AUC is robust to class imbalance and provides a good measure of model performance
    rfecv = RFECV(
        estimator=base_model,
        step=1,  # Remove one feature at a time
        cv=StratifiedKFold(n_splits=10, shuffle=True),
        scoring='roc_auc',
        n_jobs=-1,  # Use all available cores
        verbose=1
    )

    # Fit RFECV to identify optimal feature subset
    print("Starting feature selection process (this may take some time)...")
    rfecv.fit(X_train, y_train)

    # Transform training data to include only selected features
    X_train_selected = rfecv.transform(X_train)

    print(f"Feature selection complete: {rfecv.n_features_} of {X_train.shape[1]} features selected")
    print(f"Optimal number of features determined by RFECV: {rfecv.n_features_}")

    # Plot number of features vs. CV score
    plt.figure(figsize=(10, 6))
    plt.xlabel("Number of features selected")
    plt.ylabel("Mean ROC AUC (cross-validation)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.title("Feature Selection: CV Score vs Number of Features")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("lightgbm_feature_selection.png", dpi=300, bbox_inches="tight")

    return X_train_selected, rfecv


def optimize_hyperparameters(X_train_selected, y_train, use_gpu=True):
    """
    Perform Bayesian optimization to find optimal hyperparameters
    for the LightGBM classifier.

    Parameters:
    -----------
    X_train_selected : ndarray
        Feature-selected training data
    y_train : ndarray
        Training labels
    use_gpu : bool
        Whether to use GPU acceleration for LightGBM

    Returns:
    --------
    best_params : dict
        Optimized hyperparameters
    """
    print("Optimizing hyperparameters using Bayesian optimization...")

    # Define hyperparameter search space based on LightGBM documentation recommendations
    # These ranges cover a wide spectrum of potential optimal configurations
    param_bounds = {
        'num_leaves': (20, 100),  # Number of leaves in full tree
        'max_depth': (3, 9),  # Maximum tree depth
        'learning_rate': (0.01, 0.2),  # Boosting learning rate
        'feature_fraction': (0.6, 0.9),  # Fraction of features to use in each iteration
        'bagging_fraction': (0.6, 0.9),  # Fraction of data to use for each tree
        'min_child_samples': (10, 50)  # Minimum number of data per leaf
    }

    # Define objective function for Bayesian optimization
    def lgbm_evaluate(num_leaves, max_depth, learning_rate, feature_fraction, bagging_fraction, min_child_samples):
        """
        Evaluate LightGBM performance with given hyperparameters using cross-validation.
        Returns mean ROC AUC score across CV folds.
        """
        # Configure model with current hyperparameter set
        params = {
            'num_leaves': int(num_leaves),
            'max_depth': int(max_depth),
            'learning_rate': learning_rate,
            'feature_fraction': feature_fraction,
            'bagging_fraction': bagging_fraction,
            'min_child_samples': int(min_child_samples),
            'device': 'gpu' if use_gpu else 'cpu',
            'objective': 'binary',
            'metric': 'auc',
            'verbose': -1,
            'random_state': 42
        }

        # Initialize model with current hyperparameter set
        model = LGBMClassifier(**params)

        # Setup repeated stratified k-fold cross-validation for robust performance estimation
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)
        scores = []

        # Perform cross-validation for current hyperparameter set
        for i, (train_idx, val_idx) in enumerate(cv.split(X_train_selected, y_train)):
            X_train_cv, X_val_cv = X_train_selected[train_idx], X_train_selected[val_idx]
            y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]

            # Train model and evaluate performance
            model.fit(
                X_train_cv,
                y_train_cv,
                eval_set=[(X_val_cv, y_val_cv)],
                early_stopping_rounds=50,
                verbose=False
            )

            # Get validation set predictions (probabilities)
            y_val_pred = model.predict_proba(X_val_cv)[:, 1]

            # Calculate ROC AUC score
            score = roc_auc_score(y_val_cv, y_val_pred)
            scores.append(score)

        # Return mean ROC AUC across all CV iterations
        return np.mean(scores)

    # Initialize and run Bayesian optimization
    optimizer = BayesianOptimization(
        f=lgbm_evaluate,
        pbounds=param_bounds,
        verbose=2  # Set to 2 for detailed output
    )

    # Maximize the objective function (ROC AUC)
    print("Starting Bayesian optimization (this may take some time)...")
    optimizer.maximize(init_points=10, n_iter=50)

    # Extract and convert best parameters to appropriate types
    best_params = optimizer.max['params']
    best_params['num_leaves'] = int(best_params['num_leaves'])
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['min_child_samples'] = int(best_params['min_child_samples'])
    best_params['device'] = 'gpu' if use_gpu else 'cpu'
    best_params['objective'] = 'binary'
    best_params['metric'] = 'auc'
    best_params['verbose'] = -1

    print(f"Optimization complete. Best parameters found: {best_params}")
    print(f"Best CV score (ROC AUC): {optimizer.max['target']:.6f}")

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
    best_model : LGBMClassifier
        Trained model with optimal parameters
    metrics : dict
        Dictionary of performance metrics
    """
    print("Training final model and evaluating performance...")

    # Apply feature selection to test data
    X_test_selected = rfecv.transform(X_test)
    print(f"Test set after feature selection: {X_test_selected.shape[1]} features")

    # Initialize and train the best model
    best_model = LGBMClassifier(**best_params)

    # Train with early stopping using a validation set
    # This prevents overfitting and improves generalization
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train_selected, y_train, test_size=0.2, stratify=y_train
    )

    # Fit the model with early stopping
    print("Training final model with optimal hyperparameters...")
    best_model.fit(
        X_train_fit,
        y_train_fit,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        early_stopping_rounds=50
    )

    # Retrain on full training data with the optimal number of iterations
    print("Retraining on full training set with best iteration...")
    best_iterations = best_model.best_iteration_
    final_params = best_params.copy()
    final_params['n_estimators'] = best_iterations
    best_model = LGBMClassifier(**final_params)
    best_model.fit(X_train_selected, y_train)

    # Generate predictions on test set (probabilities)
    y_test_proba = best_model.predict_proba(X_test_selected)[:, 1]

    # Calculate ROC curve for visualization
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)

    # Determine optimal classification threshold from ROC curve
    # J-statistic = sensitivity + specificity - 1
    J_statistic = tpr - fpr
    optimal_idx = np.argmax(J_statistic)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold based on Youden's J statistic: {optimal_threshold:.4f}")

    # For this evaluation, use standard 0.5 threshold
    # In practice, the threshold should be chosen based on the specific application requirements
    custom_threshold = 0.5
    print(f"Using threshold of {custom_threshold} for evaluation")

    y_test_pred = (y_test_proba >= custom_threshold).astype(int)

    # Calculate comprehensive evaluation metrics
    test_confusion_matrix = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = test_confusion_matrix.ravel()

    # Build dictionary of performance metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),  # Same as sensitivity
        'f1': f1_score(y_test, y_test_pred),
        'roc_auc': roc_auc_score(y_test, y_test_proba),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
        'cohen_kappa': cohen_kappa_score(y_test, y_test_pred),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'confusion_matrix': test_confusion_matrix
    }

    # Calculate ROC AUC with 95% CI using bootstrap
    auc_mean, auc_ci_lower, auc_ci_upper = auc_ci_bootstrap(y_test, y_test_proba)
    metrics['roc_auc_ci'] = (auc_ci_lower, auc_ci_upper)

    # Print evaluation metrics
    print("\nTest Set Performance Metrics:")
    print("-" * 40)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall (Sensitivity): {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f} ({metrics['roc_auc_ci'][0]:.4f} - {metrics['roc_auc_ci'][1]:.4f})")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
    print("\nConfusion Matrix:")
    print(test_confusion_matrix)

    # Visualize ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {metrics["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red',
                label=f'Optimal threshold: {optimal_threshold:.2f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig("lightgbm_roc_curve.png", dpi=300, bbox_inches="tight")

    # Plot feature importances
    feature_importance = best_model.feature_importances_
    features_idx = np.where(rfecv.support_)[0]  # Get indices of selected features

    # Sort features by importance
    sorted_idx = np.argsort(feature_importance)[::-1]

    # Plot top 15 features (or all if less than 15)
    plt.figure(figsize=(10, 8))
    n_features_to_plot = min(15, len(feature_importance))
    plt.barh(range(n_features_to_plot),
             feature_importance[sorted_idx[:n_features_to_plot]],
             align='center')
    plt.yticks(range(n_features_to_plot),
               [f"Feature {features_idx[i]}" for i in sorted_idx[:n_features_to_plot]])
    plt.xlabel('Feature Importance')
    plt.title('Top Feature Importances (LightGBM)')
    plt.tight_layout()
    plt.savefig("lightgbm_feature_importances.png", dpi=300, bbox_inches="tight")

    return best_model, metrics


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


def save_model(model, scaler, rfecv, metrics, output_dir='./models'):
    """
    Save the trained model and associated components for future use.

    Parameters:
    -----------
    model : LGBMClassifier
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

    # Save LightGBM model using its native format
    model_path = os.path.join(output_dir, 'lightgbm_model.txt')
    model.booster_.save_model(model_path)
    print(f"LightGBM model saved to {model_path}")

    # Save other components using joblib
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
    joblib.dump(rfecv, os.path.join(output_dir, 'feature_selector.joblib'))

    # Save model configuration
    model_config = {
        'best_params': model.get_params(),
        'feature_indices': np.where(rfecv.support_)[0].tolist(),
        'n_features_original': rfecv.n_features_in_,
        'n_features_selected': rfecv.n_features_,
        'performance_metrics': {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
    }

    # Save configuration as JSON
    import json
    with open(os.path.join(output_dir, 'model_config.json'), 'w') as f:
        # Convert numpy values to Python native types
        config_serializable = {
            k: (v.item() if hasattr(v, 'item') else
                (v[0].item(), v[1].item()) if k == 'roc_auc_ci' else v)
            for k, v in model_config['performance_metrics'].items()
        }
        model_config['performance_metrics'] = config_serializable
        json.dump(model_config, f, indent=4)

    # Save performance metrics to text file for easy reference
    with open(os.path.join(output_dir, 'model_performance.txt'), 'w') as f:
        f.write("LightGBM Classification Model Performance Metrics\n")
        f.write("==============================================\n\n")
        f.write(f"Original feature count: {rfecv.n_features_in_}\n")
        f.write(f"Selected feature count: {rfecv.n_features_}\n\n")
        f.write("Test Set Performance Metrics:\n")
        f.write("-" * 40 + "\n")
        for metric, value in metrics.items():
            if metric == 'confusion_matrix':
                f.write(f"\nConfusion Matrix:\n{value}\n")
            elif metric == 'roc_auc_ci':
                f.write(f"ROC AUC 95% CI: ({value[0]:.4f} - {value[1]:.4f})\n")
            else:
                f.write(f"{metric.replace('_', ' ').title()}: {value:.4f}\n")

    print(f"All model components and metrics saved to {output_dir}")


def main():
    """Main function to execute the complete classification pipeline."""

    # Define file paths
    data_path = './data/X_y.mat'  # Update this path before running

    # Step 1: Load and preprocess data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(data_path)

    # Step 2: Perform feature selection
    X_train_selected, rfecv = perform_feature_selection(X_train, y_train, use_gpu=True)

    # Step 3: Optimize hyperparameters
    best_params = optimize_hyperparameters(X_train_selected, y_train, use_gpu=True)

    # Step 4: Train and evaluate the final model
    best_model, metrics = train_and_evaluate_model(
        X_train_selected, X_test, y_train, y_test, rfecv, best_params
    )

    # Step 5: Save the model and components
    save_model(best_model, scaler, rfecv, metrics)

    print("LightGBM classification pipeline completed successfully.")


if __name__ == "__main__":
    main()