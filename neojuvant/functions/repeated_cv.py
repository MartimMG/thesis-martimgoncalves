"""
Repeated CV for the binary-classification ensemble models (RSF, GBM, SVM, ...).

Uses StratifiedGroupKFold so a patient's rows never land in both train and val
(grouped by PERSON_ID). Metrics: AUC, PR-AUC, accuracy, precision, recall, F1,
Brier score. Everything is kept at fold/repeat/hyperparameter-combination
granularity rather than collapsed to an aggregate up front.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import warnings
from itertools import product
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, precision_score, recall_score, f1_score, brier_score_loss


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_repeated_stratified_cv(n_splits=5, n_repeats=10, base_random_state=42):
    """Build n_repeats StratifiedKFold objects, each seeded with base_random_state + i.
    Returns a list of (repeat_idx, random_state, cv) tuples."""
    cv_list = []
    for repeat_idx in range(n_repeats):
        random_state = base_random_state + repeat_idx
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_list.append((repeat_idx, random_state, cv))
    return cv_list


def generate_cv_splits(X, y, groups=None, n_splits=5, n_repeats=10, base_random_state=42):
    """Materialize every (repeat, fold) split up front so the same indices can be
    reused across a whole grid search instead of re-splitting per combination.

    Falls back to plain StratifiedKFold when groups is None; otherwise uses
    StratifiedGroupKFold (e.g. grouped by PERSON_ID) so one patient never ends up
    in both train and val. Returns a list of dicts: repeat_idx, fold_idx,
    random_state, train_idx, val_idx.
    """
    all_splits = []
    
    for repeat_idx in range(n_repeats):
        random_state = base_random_state + repeat_idx
        
        if groups is not None:
            # Use StratifiedGroupKFold to keep same patient together
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            splits = cv.split(X, y, groups=groups)
        else:
            # Regular stratified K-fold
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            splits = cv.split(X, y)
        
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            all_splits.append({
                'repeat_idx': repeat_idx,
                'fold_idx': fold_idx,
                'random_state': random_state,
                'train_idx': train_idx,
                'val_idx': val_idx
            })
    
    return all_splits

# =============================================================================
# ENSEMBLE MODEL FUNCTIONS - REPEATED CV
# =============================================================================

def ensemble_repeated_cross_validation(model_class, X, y, model_params, metrics,
                                       scale_data=False, scaler_type='standard',
                                       continuous_features=None,
                                       random_state=None, n_jobs=None,
                                       groups=None,
                                       cat_features=None,
                                       n_splits=5, n_repeats=10, base_random_state=42,
                                       verbose=False):
    """Run one full repeated-CV pass (all repeats x all folds) for a single fixed
    model_params, scoring every metric in `metrics` on each fold.

    random_state / n_jobs / cat_features are only injected into model_params if the
    model class actually accepts them (checked via inspect.signature), so this works
    across model classes with different constructor signatures. A fit or predict
    failure on a fold yields NaN for all metrics on that fold instead of raising,
    and gets recorded in error_flags rather than crashing the whole search.

    metrics can include 'auc', 'pr-auc', 'accuracy', 'precision', 'recall', 'f1',
    'brier'; the binary ones threshold the model's probability output at 0.5.
    scale_data/scaler_type/continuous_features control optional per-fold scaling
    (fit on train, applied to val) - continuous_features restricts scaling to those
    columns and leaves categoricals alone. cat_features, when passed as column
    names, gets converted to positional indices per fold (for CatBoost).

    Returns (results_df, error_flags).
    """
    import inspect
    
    # Generate all splits
    all_splits = generate_cv_splits(X, y, groups, n_splits, n_repeats, base_random_state)
    
    results = {metric: [] for metric in metrics}
    error_flags = {metric: [] for metric in metrics}
    result_rows = []
    
    for split in all_splits:
        repeat_idx = split['repeat_idx']
        fold_idx = split['fold_idx']
        train_idx = split['train_idx']
        val_idx = split['val_idx']
        split_random_state = split['random_state']
        
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_test = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx], y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]
        
        # Scale if requested
        if scale_data:
            if continuous_features is not None:
                # Only scale continuous features
                if scaler_type == 'standard':
                    scaler = ColumnTransformer([
                        ('continuous', StandardScaler(), continuous_features)
                    ], remainder='passthrough')
                else:
                    scaler = ColumnTransformer([
                        ('continuous', MinMaxScaler(), continuous_features)
                    ], remainder='passthrough')
                
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Get feature names after transformation
                feature_names = continuous_features + [col for col in X_train.columns if col not in continuous_features]
                
                X_train = pd.DataFrame(
                    X_train_scaled,
                    columns=feature_names,
                    index=X_train.index
                )
                X_test = pd.DataFrame(
                    X_test_scaled,
                    columns=feature_names,
                    index=X_test.index
                )
            else:
                # Scale all features
                if scaler_type == 'standard':
                    scaler = StandardScaler()
                else:
                    scaler = MinMaxScaler()
                
                X_train = pd.DataFrame(
                    scaler.fit_transform(X_train),
                    columns=X_train.columns,
                    index=X_train.index
                )
                X_test = pd.DataFrame(
                    scaler.transform(X_test),
                    columns=X_test.columns,
                    index=X_test.index
                )
        
        # Initialize and fit model
        model_params_copy = model_params.copy()
        if random_state is not None:
            if 'random_state' in inspect.signature(model_class.__init__).parameters:
                model_params_copy['random_state'] = random_state
        if n_jobs is not None:
            if 'n_jobs' in inspect.signature(model_class.__init__).parameters:
                model_params_copy['n_jobs'] = n_jobs
        
        # Handle cat_features separately (for CatBoost compatibility)
        # Convert column names to indices based on current fold's X_train
        if cat_features is not None:
            if isinstance(cat_features, list) and len(cat_features) > 0:
                # Check if first element is a string (column name)
                if isinstance(cat_features[0], str):
                    # Convert column names to indices
                    model_params_copy['cat_features'] = [X_train.columns.get_loc(col) for col in cat_features]
                else:
                    # Already indices
                    model_params_copy['cat_features'] = cat_features
        
        try:
            model = model_class(**model_params_copy)
            model.fit(X_train, y_train)
        except Exception as e:
            if verbose:
                print(f"  Warning: Model fitting failed: {str(e)}")
            # Store NaN for all metrics
            result_row = {
                'repeat_idx': repeat_idx,
                'fold_idx': fold_idx,
            }
            for metric in metrics:
                result_row[metric] = np.nan
                error_flags[metric].append(True)
            result_rows.append(result_row)
            continue
        
        result_row = {
            'repeat_idx': repeat_idx,
            'fold_idx': fold_idx,
        }
        
        # Get predictions
        try:
            # Get probability predictions for positive class
            if hasattr(model, 'predict_proba'):
                pred_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'predict'):
                # Some models only have predict - use as probability
                pred_proba = model.predict(X_test)
            else:
                raise ValueError("Model does not have predict_proba or predict method")
            
            # Create binary predictions using 0.5 threshold
            pred_binary = (pred_proba >= 0.5).astype(int)
            
        except Exception as e:
            if verbose:
                print(f"    Warning: Prediction failed: {str(e)}")
            for metric in metrics:
                result_row[metric] = np.nan
                error_flags[metric].append(True)
            result_rows.append(result_row)
            continue
        
        # Evaluate on each metric
        for metric in metrics:
            error_occurred = False
            
            if metric == 'auc':
                try:
                    score = roc_auc_score(y_test, pred_proba)
                    result_row[metric] = score
                except Exception as e:
                    if verbose:
                        print(f"    Warning: AUC calculation failed: {str(e)}")
                    result_row[metric] = np.nan
                    error_occurred = True
                    
            elif metric == 'pr-auc':
                try:
                    precision_vals, recall_vals, _ = precision_recall_curve(y_test, pred_proba)
                    score = auc(recall_vals, precision_vals)
                    result_row[metric] = score
                except Exception as e:
                    if verbose:
                        print(f"    Warning: PR-AUC calculation failed: {str(e)}")
                    result_row[metric] = np.nan
                    error_occurred = True
                    
            elif metric == 'accuracy':
                try:
                    score = accuracy_score(y_test, pred_binary)
                    result_row[metric] = score
                except Exception as e:
                    if verbose:
                        print(f"    Warning: Accuracy calculation failed: {str(e)}")
                    result_row[metric] = np.nan
                    error_occurred = True
                    
            elif metric == 'precision':
                try:
                    score = precision_score(y_test, pred_binary, zero_division=0)
                    result_row[metric] = score
                except Exception as e:
                    if verbose:
                        print(f"    Warning: Precision calculation failed: {str(e)}")
                    result_row[metric] = np.nan
                    error_occurred = True
                    
            elif metric == 'recall':
                try:
                    score = recall_score(y_test, pred_binary, zero_division=0)
                    result_row[metric] = score
                except Exception as e:
                    if verbose:
                        print(f"    Warning: Recall calculation failed: {str(e)}")
                    result_row[metric] = np.nan
                    error_occurred = True
                    
            elif metric == 'f1':
                try:
                    score = f1_score(y_test, pred_binary, zero_division=0)
                    result_row[metric] = score
                except Exception as e:
                    if verbose:
                        print(f"    Warning: F1 score calculation failed: {str(e)}")
                    result_row[metric] = np.nan
                    error_occurred = True
                    
            elif metric == 'brier':
                try:
                    score = brier_score_loss(y_test, pred_proba)
                    result_row[metric] = score
                except Exception as e:
                    if verbose:
                        print(f"    Warning: Brier score calculation failed: {str(e)}")
                    result_row[metric] = np.nan
                    error_occurred = True
            
            error_flags[metric].append(error_occurred)
        
        result_rows.append(result_row)
    
    results_df = pd.DataFrame(result_rows)
    return results_df, error_flags


def ensemble_repeated_grid_search(model_class, X, y, param_grid, metrics,
                                  model_name='Model',
                                  scale_data=False, scaler_type='standard',
                                  continuous_features=None,
                                  random_state=None, n_jobs=None,
                                  groups=None,
                                  cat_features=None,
                                  n_splits=5, n_repeats=10, base_random_state=42,
                                  verbose=True):
    """Grid search over param_grid, built on top of ensemble_repeated_cross_validation.

    Splits are generated once and reused for every combination, so combinations are
    compared on identical folds. param_grid should NOT include 'cat_features' - pass
    that separately (same handling as in ensemble_repeated_cross_validation: column
    names get converted to positional indices per fold, for CatBoost).

    Best params per metric are chosen by max mean, except metrics with 'brier' in
    the name where lower is better. summary_df includes not just the mean/std per
    combination but a full breakdown per repeat and per fold-within-repeat, which
    makes it a wide table but useful later for eyeballing variance.

    Returns (results_df, summary_df, best_params). Prints progress/ETA when verbose.
    """
    start_time = time.time()
    
    # Generate all splits once
    all_splits = generate_cv_splits(X, y, groups, n_splits, n_repeats, base_random_state)
    
    # Generate parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    total_combinations = len(param_combinations)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"{model_name.upper()} REPEATED GRID SEARCH")
        print(f"{'='*80}")
        print(f"Scaling: {'Yes (' + scaler_type + ')' if scale_data else 'No'}")
        print(f"Grouping: {'Yes (StratifiedGroupKFold)' if groups is not None else 'No (StratifiedKFold)'}")
        print(f"Metrics: {metrics}")
        print(f"Parameter Grid:")
        for param, values in param_grid.items():
            print(f"  - {param}: {values}")
        print(f"Total hyperparameter combinations: {total_combinations}")
        print(f"Folds per repetition: {n_splits}")
        print(f"Repetitions: {n_repeats}")
        print(f"Total evaluations: {total_combinations * n_splits * n_repeats}")
        print(f"{'='*80}\n")
    
    all_results = []
    
    for comb_idx, param_values_tuple in enumerate(param_combinations):
        params = dict(zip(param_names, param_values_tuple))
        
        if verbose:
            elapsed = time.time() - start_time
            print(f"[{comb_idx + 1}/{total_combinations}] Testing params: {params}")
            if comb_idx > 0:
                avg_time = elapsed / comb_idx
                remaining = avg_time * (total_combinations - comb_idx)
                print(f"  Elapsed: {elapsed/60:.1f}min, Est. remaining: {remaining/60:.1f}min")
        
        # Run repeated CV for this parameter combination using the pre-generated splits
        try:
            cv_results, error_flags = ensemble_repeated_cross_validation(
                model_class=model_class,
                X=X,
                y=y,
                model_params=params,
                metrics=metrics,
                scale_data=scale_data,
                scaler_type=scaler_type,
                continuous_features=continuous_features,
                random_state=random_state,
                n_jobs=n_jobs,
                groups=groups,
                cat_features=cat_features,
                n_splits=n_splits,
                n_repeats=n_repeats,
                base_random_state=base_random_state,
                verbose=False
            )
            
            # Add parameter columns to results
            for idx, row in cv_results.iterrows():
                result_row = params.copy()
                result_row['repeat_idx'] = row['repeat_idx']
                result_row['fold_idx'] = row['fold_idx']
                for metric in metrics:
                    result_row[metric] = row[metric]
                all_results.append(result_row)
            
            if verbose:
                for metric in metrics:
                    score = cv_results[metric].mean()
                    std = cv_results[metric].std()
                    print(f"  {metric}: {score:.4f} ± {std:.4f}")
                    
        except Exception as e:
            if verbose:
                print(f"  ⚠️ Error: {str(e)}")
            
            # Store NaN for failed combinations
            for split in all_splits:
                result_row = params.copy()
                result_row['repeat_idx'] = split['repeat_idx']
                result_row['fold_idx'] = split['fold_idx']
                for metric in metrics:
                    result_row[metric] = np.nan
                all_results.append(result_row)
    
    # Create detailed results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Create summary DataFrame
    summary_rows = []
    for comb_idx, param_values_tuple in enumerate(param_combinations):
        params = dict(zip(param_names, param_values_tuple))
        
        # Build mask for this parameter combination.
        # Use null-aware matching so grid values like None are matched correctly.
        mask = pd.Series(True, index=results_df.index)
        for param_name, param_value in params.items():
            if pd.isna(param_value):
                mask = mask & results_df[param_name].isna()
            else:
                mask = mask & results_df[param_name].eq(param_value)
        
        subset = results_df[mask]
        
        row = params.copy()
            
        for metric in metrics:
            row[f'{metric}_mean'] = subset[metric].mean()
            row[f'{metric}_std'] = subset[metric].std()
            
            # Store per-repeat means
            for repeat_idx in range(n_repeats):
                repeat_mask = subset['repeat_idx'] == repeat_idx
                row[f'{metric}_repeat_{repeat_idx+1}_mean'] = subset[repeat_mask][metric].mean()
            
            # Store per-fold-per-repeat values
            for repeat_idx in range(n_repeats):
                for fold_idx in range(n_splits):
                    fold_repeat_mask = (subset['repeat_idx'] == repeat_idx) & (subset['fold_idx'] == fold_idx)
                    if fold_repeat_mask.any():
                        row[f'{metric}_repeat_{repeat_idx+1}_fold_{fold_idx+1}'] = subset[fold_repeat_mask][metric].values[0]
                    else:
                        row[f'{metric}_repeat_{repeat_idx+1}_fold_{fold_idx+1}'] = np.nan
        
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Find best parameters for each metric
    best_params = {}
    for metric in metrics:
        mean_col = f'{metric}_mean'
        valid_results = summary_df[~summary_df[mean_col].isna()].copy()
        
        if len(valid_results) == 0:
            if verbose:
                print(f"\n⚠️ Warning: No valid results for {metric}")
            best_params[metric] = None
            continue
        
        # For Brier score, lower is better
        if 'brier' in metric.lower():
            best_idx = valid_results[mean_col].idxmin()
        else:
            best_idx = valid_results[mean_col].idxmax()
        
        best_row = summary_df.loc[best_idx]
        
        # Extract best parameters
        best_param_dict = {}
        for param in param_names:
            best_param_dict[param] = best_row[param]
        
        best_params[metric] = {
            'params': best_param_dict,
            'mean': best_row[f'{metric}_mean'],
            'std': best_row[f'{metric}_std']
        }
    
    elapsed_time = time.time() - start_time
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"REPEATED GRID SEARCH COMPLETE")
        print(f"{'='*80}")
        print(f"Total time: {elapsed_time/60:.1f} minutes")
        print(f"\nBest parameters for each metric:")
        print(f"{'-'*80}")
        for metric, info in best_params.items():
            if info is not None:
                print(f"\n{metric}:")
                print(f"  Score: {info['mean']:.4f} ± {info['std']:.4f}")
                print(f"  Parameters:")
                for param, value in info['params'].items():
                    print(f"    {param}: {value}")
        print(f"\n{'='*80}\n")
    
    return results_df, summary_df, best_params