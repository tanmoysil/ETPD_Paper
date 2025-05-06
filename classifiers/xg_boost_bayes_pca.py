#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Binary classification using XGBoost with dimensionality reduction through PCA.
This implementation includes feature reduction, Bayesian hyperparameter optimization,
and model explainability with SHAP (SHapley Additive exPlanations) analysis.

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
from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization
import shap


def load_and_preprocess_data(filepath, feature_names_path=None):
    """
    Load data from a .mat file and preprocess it for machine learning.

    Parameters:
    -----------
    filepath : str
        Path to the .mat file containing features (X) and labels (y)
    feature_names_path : str, optional
        Path to a file containing feature names

    Returns:
    --------
    data_dict : dict
        Dictionary containing preprocessed data and metadata
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
    print(f"Dataset loaded: {n_samples} samples with {n_features} features")
    print(f"Class distribution: {class_distribution}")

    # Load feature names if provided
    if feature_names_path and os.path.exists(feature_names_path):
        feature_names = np.genfromtxt(feature_names_path, dtype=str, delimiter='\n')
        print(f"Loaded {len(feature_names)} feature names")
    else:
        feature_names = [f"Feature_{i}" for i in range(n_features)]
        print("No feature names file provided, using generic feature names")

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Standardize features to zero mean and unit variance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create dictionary with all data
    data_dict = {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'scaler': scaler
    }

    print(f"Data preprocessing completed in {time() - start_time:.2f} seconds")

    return data_dict


def apply_pca(X_train, X_test, feature_names, variance_threshold=0.95, plot_variance=True):
    """
    Apply Principal Component Analysis (PCA) for dimensionality reduction.

    Parameters:
    -----------
    X_train : ndarray
        Training feature matrix (already scaled)
    X_test : ndarray
        Test feature matrix (already scaled)
    feature_names : list or ndarray
        Original feature names
    variance_threshold : float
        Percentage of variance to retain (default: 0.95)
    plot_variance : bool
        Whether to plot explained variance

    Returns:
    --------
    pca_dict : dict
        Dictionary containing PCA-transformed data and PCA object
    """
    print(f"Applying PCA with {variance_threshold:.0%} variance threshold...")
    start_time = time()

    # Initialize and fit PCA
    pca = PCA(n_components=variance_threshold)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    n_components = pca.n_components_
    total_variance = np.sum(pca.explained_variance_ratio_)

    print(f"Reduced dimensions from {X_train.shape[1]} to {n_components} features")
    print(f"Retained {total_variance:.2%} of total variance")

    # Plot explained variance if requested
    if plot_variance:
        plt.figure(figsize=(10, 6))

        # Cumulative explained variance
        plt.subplot(1, 2, 1)
        plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', markersize=6)
        plt.axhline(y=variance_threshold, color='r', linestyle='--',
                    label=f'{variance_threshold:.0%} Threshold')
        plt.axvline(x=n_components - 0.5, color='g', linestyle='--',
                    label=f'{n_components} Components')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Cumulative Explained Variance')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Individual explained variance
        plt.subplot(1, 2, 2)
        plt.bar(range(1, min(20, len(pca.explained_variance_ratio_)) + 1),
                pca.explained_variance_ratio_[:20])
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance by Component (Top 20)')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('pca_variance_explained.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Create dictionary with PCA results
    pca_dict = {
        'X_train_pca': X_train_pca,
        'X_test_pca': X_test_pca,
        'pca': pca,
        'n_components': n_components,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'total_variance': total_variance
    }

    print(f"PCA transformation completed in {time() - start_time:.2f} seconds")

    return pca_dict


def optimize_xgboost_hyperparameters(X_train_pca, y_train):
    """
    Perform Bayesian optimization to find optimal XGBoost hyperparameters.

    Parameters:
    -----------
    X_train_pca : ndarray
        PCA-transformed training data
    y_train : ndarray
        Training labels

    Returns:
    --------
    best_params : dict
        Optimized hyperparameters
    optimization_results : dict
        Information about the optimization process
    """
    print("Optimizing XGBoost hyperparameters using Bayesian optimization...")
    start_time = time()

    # Define hyperparameter search space
    param_bounds = {
        'n_estimators': (50, 300),  # Number of boosting rounds
        'max_depth': (3, 9),  # Maximum tree depth
        'learning_rate': (0.01, 0.2),  # Step size shrinkage
        'subsample': (0.7, 0.9),  # Subsample ratio of training instances
        'colsample_bytree': (0.6, 0.9),  # Subsample ratio of columns when constructing each tree
        'scale_pos_weight': (0.45, 2)  # Control the balance of positive and negative weights
    }

    # Define objective function for Bayesian optimization
    def xgb_evaluate(n_estimators, max_depth, learning_rate, subsample, colsample_bytree, scale_pos_weight):
        """Objective function to maximize ROC AUC through cross-validation"""
        params = {
            'n_estimators': int(n_estimators),
            'max_depth': int(max_depth),
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'scale_pos_weight': scale_pos_weight,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }

        # Initialize model with current hyperparameter set
        model = XGBClassifier(**params)

        # Setup repeated stratified k-fold cross-validation
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)
        scores = []

        # Perform cross-validation for current hyperparameter set
        for train_idx, val_idx in cv.split(X_train_pca, y_train):
            X_train_cv, X_val_cv = X_train_pca[train_idx], X_train_pca[val_idx]
            y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]

            # Train model and evaluate performance
            model.fit(X_train_cv, y_train_cv)
            y_val_pred = model.predict_proba(X_val_cv)[:, 1]
            score = roc_auc_score(y_val_cv, y_val_pred)
            scores.append(score)

        return np.mean(scores)

    # Initialize and run Bayesian optimization
    optimizer = BayesianOptimization(
        f=xgb_evaluate,
        pbounds=param_bounds,
        verbose=2  # Detailed logs
    )

    # Maximize the objective function (ROC AUC)
    print("Running optimization (this may take some time)...")
    optimizer.maximize(init_points=10, n_iter=50)

    # Extract best parameters
    best_params = optimizer.max['params']
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['eval_metric'] = 'logloss'
    best_params['use_label_encoder'] = False

    # Compile optimization results
    optimization_results = {
        'best_score': optimizer.max['target'],
        'best_params': best_params,
        'all_results': optimizer.res,
        'execution_time': time() - start_time
    }

    # Save optimization history to CSV
    results_df = pd.DataFrame(optimizer.res)
    results_df.to_csv('xgboost_optimization_results.csv', index=False)

    print(f"Optimization complete in {optimization_results['execution_time']:.2f} seconds")
    print(f"Best parameters: {best_params}")
    print(f"Best CV score (ROC AUC): {optimizer.max['target']:.6f}")

    return best_params, optimization_results


def train_and_evaluate_model(X_train_pca, X_test_pca, y_train, y_test, best_params, classification_threshold=0.5):
    """
    Train XGBoost model with optimized parameters and evaluate on test data.

    Parameters:
    -----------
    X_train_pca : ndarray
        PCA-transformed training data
    X_test_pca : ndarray
        PCA-transformed test data
    y_train : ndarray
        Training labels
    y_test : ndarray
        Test labels
    best_params : dict
        Optimized hyperparameters
    classification_threshold : float
        Probability threshold for binary classification

    Returns:
    --------
    model_dict : dict
        Dictionary containing model, predictions, and performance metrics
    """
    print("Training final model and evaluating performance...")
    start_time = time()

    # Initialize and train the model
    model = XGBClassifier(**best_params)
    model.fit(X_train_pca, y_train)

    training_time = time() - start_time
    print(f"Model training completed in {training_time:.2f} seconds")

    # Generate predictions on test set
    y_test_proba = model.predict_proba(X_test_pca)[:, 1]
    y_test_pred = (y_test_proba >= classification_threshold).astype(int)

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = conf_matrix.ravel()

    # Calculate performance metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'f1': f1_score(y_test, y_test_pred),
        'roc_auc': roc_auc_score(y_test, y_test_proba),
        'cohen_kappa': cohen_kappa_score(y_test, y_test_pred),
        'confusion_matrix': conf_matrix,
        'training_time': training_time
    }

    # Calculate ROC AUC with 95% CI using bootstrap
    metrics['roc_auc_mean'], metrics['roc_auc_ci_lower'], metrics['roc_auc_ci_upper'] = auc_ci_bootstrap(
        y_test, y_test_proba
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

    # Plot ROC curve
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
    plt.savefig('xgboost_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Compile model results
    model_dict = {
        'model': model,
        'y_test_pred': y_test_pred,
        'y_test_proba': y_test_proba,
        'metrics': metrics,
        'roc_curve': {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }
    }

    return model_dict


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


def create_shap_visualizations(model, X_train_pca, X_test_pca, feature_names, output_dir='./visualizations'):
    """
    Create SHAP visualizations to explain model predictions.

    Parameters:
    -----------
    model : XGBClassifier
        Trained XGBoost model
    X_train_pca : ndarray
        PCA-transformed training data (for model explanation)
    X_test_pca : ndarray
        PCA-transformed test data (for visualization)
    feature_names : list
        List of feature names (will be truncated to PCA dimensions)
    output_dir : str
        Directory to save visualizations

    Returns:
    --------
    shap_dict : dict
        Dictionary with SHAP information
    """
    print("Creating SHAP visualizations for model interpretability...")
    start_time = time()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Truncate feature_names to match PCA dimensions
    pca_feature_names = [f"PC_{i + 1}" for i in range(X_train_pca.shape[1])]

    # Create SHAP explainer
    explainer = shap.Explainer(model, X_train_pca)

    # Calculate SHAP values for test set
    shap_values = explainer(X_test_pca)

    # Prepare visualization parameters
    plt.rcParams.update({'font.size': 12})

    # Generate summary plot (beeswarm)
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values,
        X_test_pca,
        feature_names=pca_feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_summary_beeswarm.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/shap_summary_beeswarm.svg", format='svg', bbox_inches='tight')
    plt.close()

    # Generate bar plot (feature importance)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_test_pca,
        feature_names=pca_feature_names,
        plot_type="bar",
        max_display=15,
        show=False
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_summary_bar.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/shap_summary_bar.svg", format='svg', bbox_inches='tight')
    plt.close()

    # Generate additional SHAP visualizations as needed
    # Generate dependence plots for top features
    top_indices = np.argsort(-np.abs(shap_values.values.mean(0)))[:5]
    for i, idx in enumerate(top_indices):
        plt.figure(figsize=(10, 7))
        shap.dependence_plot(
            idx,
            shap_values.values,
            X_test_pca,
            feature_names=pca_feature_names,
            show=False
        )
        plt.tight_layout()
        plt.savefig(f"{output_dir}/shap_dependence_{pca_feature_names[idx]}.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Create a compiled SHAP information dictionary
    shap_dict = {
        'explainer': explainer,
        'shap_values': shap_values,
        'execution_time': time() - start_time
    }

    print(f"SHAP visualization completed in {shap_dict['execution_time']:.2f} seconds")

    return shap_dict


def save_model(model, pca, scaler, metrics, best_params, output_dir='./models'):
    """
    Save the trained model and associated components for future use.

    Parameters:
    -----------
    model : XGBClassifier
        Trained classifier
    pca : PCA
        Fitted PCA transformer
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
    model.save_model(os.path.join(output_dir, 'xgboost_model.json'))
    joblib.dump(pca, os.path.join(output_dir, 'pca_transformer.joblib'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))

    # Save model configuration as JSON
    import json
    model_config = {
        'hyperparameters': best_params,
        'pca_components': pca.n_components_,
        'pca_explained_variance': pca.explained_variance_ratio_.tolist(),
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
        f.write("XGBoost with PCA Classification Model Performance\n")
        f.write("===========================================\n\n")
        f.write(f"PCA components: {pca.n_components_}\n")
        f.write(f"PCA explained variance: {sum(pca.explained_variance_ratio_):.4f}\n\n")
        f.write("Best Hyperparameters:\n")
        for param, value in best_params.items():
            f.write(f"  {param}: {value}\n")
        f.write("\nPerformance Metrics:\n")
        for metric, value in metrics.items():
            if metric == 'confusion_matrix':
                f.write(f"\nConfusion Matrix:\n{value}\n")
            elif metric == 'roc_auc_ci_lower':
                continue
            elif metric == 'roc_auc_ci_upper':
                continue
            elif metric == 'roc_auc_mean':
                f.write(
                    f"ROC AUC: {metrics['roc_auc']:.4f} ({metrics['roc_auc_ci_lower']:.4f} - {metrics['roc_auc_ci_upper']:.4f})\n")
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
    data_path = './data/X_y.mat'
    feature_names_path = './data/features_all.txt'

    # Step 1: Load and preprocess data
    data_dict = load_and_preprocess_data(data_path, feature_names_path)

    # Step 2: Apply PCA for dimensionality reduction
    pca_dict = apply_pca(
        data_dict['X_train'],
        data_dict['X_test'],
        data_dict['feature_names'],
        variance_threshold=0.95,
        plot_variance=True
    )

    # Step 3: Optimize hyperparameters
    best_params, optimization_results = optimize_xgboost_hyperparameters(
        pca_dict['X_train_pca'],
        data_dict['y_train']
    )

    # Step 4: Train and evaluate the final model
    model_dict = train_and_evaluate_model(
        pca_dict['X_train_pca'],
        pca_dict['X_test_pca'],
        data_dict['y_train'],
        data_dict['y_test'],
        best_params,
        classification_threshold=0.5
    )

    # Step 5: Create SHAP visualizations
    shap_dict = create_shap_visualizations(
        model_dict['model'],
        pca_dict['X_train_pca'],
        pca_dict['X_test_pca'],
        data_dict['feature_names'],
        output_dir='./visualizations'
    )

    # Step 6: Save the model and components
    save_model(
        model_dict['model'],
        pca_dict['pca'],
        data_dict['scaler'],
        model_dict['metrics'],
        best_params,
        output_dir='./models'
    )

    # Report total execution time
    total_time = time() - pipeline_start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal pipeline execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print("XGBoost with PCA classification pipeline completed successfully.")


if __name__ == "__main__":
    main()