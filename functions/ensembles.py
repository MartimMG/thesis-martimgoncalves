import matplotlib.pyplot as plt
import numpy as np
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sksurv.metrics import concordance_index_censored
from sksurv.metrics import concordance_index_ipcw
from sksurv.metrics import integrated_brier_score, brier_score
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from functions.repeated_cv import sample_case_control


def generic_grid_search(model_class, X, y, cv_folds, param_grid, 
                       metrics=['c-index', 'c-index-ipcw', 'int-brier-score', 'cumulative-dynamic-auc'],
                       model_name='Model',
                       scale_data=False, scaler_type='standard',
                       verbose=True, random_state=None, n_jobs=None,
                       n_controls_per_case=None, dataset=None, feature_columns=None,
                       event_col='Regional LN recurrence', time_col='Survival days'):
    """
    Manual grid search over param_grid for any sksurv-style model (RSF, GBM, SVM...).

    Scaling (if scale_data=True) happens inside each fold, fit on train only, so no
    leakage from the test split. Use scale_data=True for Cox/SVM, False for RSF/GBM --
    trees don't care about feature scale.

    X/y are the unscaled training data (X a DataFrame, y a structured array with
    'event'/'time'). cv_folds is a train/test index iterator. n_controls_per_case,
    dataset and feature_columns are only relevant if you want case-control subsampling
    inside each fold; dataset must then contain event_col/time_col.

    Returns (results_df, best_params, summary): results_df has one row per parameter
    combo with mean/std/fold-level values for each metric; best_params and summary
    give the winning config per metric (lower-is-better for anything with "brier"
    in the name, higher otherwise).
    """
    import time
    from itertools import product
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    start_time = time.time()
    
    # materialize cv_folds so we can loop over it once per param combo
    if not isinstance(cv_folds, list):
        cv_folds = list(cv_folds)

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    total_combinations = len(param_combinations)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"{model_name.upper()} GRID SEARCH (Manual Implementation)")
        print(f"{'='*80}")
        print(f"Model: {model_class.__name__}")
        print(f"Scaling: {'Yes (' + scaler_type + ') - INSIDE each fold' if scale_data else 'No'}")
        print(f"Subsampling: {'Yes (' + str(n_controls_per_case) + ' controls per case)' if n_controls_per_case else 'No'}")
        print(f"Metrics: {metrics}")
        print(f"Parameter Grid:")
        for param, values in param_grid.items():
            print(f"  - {param}: {values}")
        print(f"Total combinations: {total_combinations}")
        print(f"{'='*80}\n")
    
    all_results = []

    for comb_idx, param_values_tuple in enumerate(param_combinations):
        params = dict(zip(param_names, param_values_tuple))

        # n_controls_per_case can itself be swept as a grid param
        current_n_controls = params.pop('n_controls_per_case', n_controls_per_case)
        
        if verbose:
            elapsed = time.time() - start_time
            if current_n_controls is not None:
                print(f"[{comb_idx + 1}/{total_combinations}] Testing params: {params}, n_controls_per_case={current_n_controls}")
            else:
                print(f"[{comb_idx + 1}/{total_combinations}] Testing params: {params}")
            if comb_idx > 0:
                avg_time = elapsed / comb_idx
                remaining = avg_time * (total_combinations - comb_idx)
                print(f"  Elapsed: {elapsed/60:.1f}min, Est. remaining: {remaining/60:.1f}min")
        
        try:
            cv_results, error_flags = ensemble_cross_validation(
                model_class=model_class,
                X=X,
                y=y,
                cv_folds=cv_folds,
                model_params=params,
                metrics=metrics,
                scale_data=scale_data,
                scaler_type=scaler_type,
                random_state=random_state,
                n_jobs=n_jobs,
                n_controls_per_case=current_n_controls,
                dataset=dataset,
                feature_columns=feature_columns,
                event_col=event_col,
                time_col=time_col,
                verbose=False  # Don't print errors for each fold in grid search
            )
            
            result_dict = params.copy()
            if current_n_controls is not None:
                result_dict['n_controls_per_case'] = current_n_controls

            for metric in metrics:
                result_dict[f'{metric}_mean'] = cv_results.loc['Mean', metric]
                result_dict[f'{metric}_std'] = cv_results.loc['Std', metric]

            # keep the per-fold values too, useful for spotting one bad fold later
            fold_rows = [idx for idx in cv_results.index if idx.startswith('Fold_')]
            for metric in metrics:
                for fold_idx, fold_name in enumerate(fold_rows):
                    fold_value = cv_results.loc[fold_name, metric]
                    result_dict[f'{metric}_fold_{fold_idx+1}'] = fold_value

            for metric in metrics:
                has_error = any(error_flags.get(metric, []))
                result_dict[f'{metric}_has_error'] = has_error
            
            all_results.append(result_dict)
            
            if verbose:
                for metric in metrics:
                    score = cv_results.loc['Mean', metric]
                    std = cv_results.loc['Std', metric]
                    print(f"  {metric}: {score:.4f} ± {std:.4f}")
        
        except Exception as e:
            if verbose:
                print(f"  Error: {str(e)}")
            # whole combo blew up (not just one fold) -- fill everything with NaN
            # and flag it so it doesn't silently win a metric later
            result_dict = params.copy()
            if current_n_controls is not None:
                result_dict['n_controls_per_case'] = current_n_controls

            for metric in metrics:
                result_dict[f'{metric}_mean'] = np.nan
                result_dict[f'{metric}_std'] = np.nan

            num_folds = len(cv_folds)
            for metric in metrics:
                for fold_idx in range(num_folds):
                    result_dict[f'{metric}_fold_{fold_idx+1}'] = np.nan
                result_dict[f'{metric}_has_error'] = True

            all_results.append(result_dict)

    results_df = pd.DataFrame(all_results)

    best_params = {}
    summary = {}
    
    for metric in metrics:
        mean_col = f'{metric}_mean'
        std_col = f'{metric}_std'
        
        # Remove NaN values
        valid_results = results_df[~results_df[mean_col].isna()].copy()
        
        if len(valid_results) == 0:
            if verbose:
                print(f"\nWarning: no valid results for {metric}")
            best_params[metric] = None
            summary[metric] = None
            continue
        
        # For Brier score, lower is better; for others, higher is better
        if 'brier' in metric.lower():
            best_idx = valid_results[mean_col].idxmin()
        else:
            best_idx = valid_results[mean_col].idxmax()
        
        best_row = results_df.loc[best_idx]

        best_param_dict = {}
        for param in param_names:
            if param != 'n_controls_per_case':
                best_param_dict[param] = best_row[param]
        if 'n_controls_per_case' in best_row.index:
            best_param_dict['n_controls_per_case'] = best_row['n_controls_per_case']
        
        best_params[metric] = best_param_dict
        summary[metric] = {
            'best_score': best_row[mean_col],
            'std': best_row[std_col],
            'best_params': best_param_dict
        }
    
    elapsed_time = time.time() - start_time
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"GRID SEARCH COMPLETE")
        print(f"{'='*80}")
        print(f"Total time: {elapsed_time/60:.1f} minutes")
        print(f"\nBest parameters for each metric:")
        print(f"{'-'*80}")
        for metric, info in summary.items():
            if info is not None:
                print(f"\n{metric}:")
                print(f"  Score: {info['best_score']:.4f} ± {info['std']:.4f}")
                print(f"  Parameters:")
                for param, value in info['best_params'].items():
                    print(f"    {param}: {value}")
        print(f"\n{'='*80}\n")
    
    return results_df, best_params, summary


def ensemble_cross_validation(model_class, X, y, cv_folds, model_params,
                              metrics, scale_data=False, scaler_type='standard',
                              random_state=None, n_jobs=None, 
                              n_controls_per_case=None, dataset=None, feature_columns=None,
                              event_col='Regional LN recurrence', time_col='Survival days',
                              verbose=False):
    """
    Cross-validate model_class(**model_params) over cv_folds, one fold at a time.

    Same subsampling/scaling behaviour as generic_grid_search (scaling fit on train
    only, subsampling drawn from dataset via sample_case_control when
    n_controls_per_case is given). random_state/n_jobs get injected into
    model_params only if the model's __init__ actually accepts them.

    Returns results_df (rows: Fold_1..Fold_n, Mean, Std; columns: metrics) plus
    error_flags, a dict of per-metric booleans marking which folds failed.
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, integrated_brier_score
    
    if not isinstance(cv_folds, list):
        cv_folds = list(cv_folds)

    results = {metric: [] for metric in metrics}
    error_flags = {metric: [] for metric in metrics}
    fold_num = 0

    for train_idx, test_idx in cv_folds:
        fold_num += 1

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if n_controls_per_case is not None:
            if dataset is None or feature_columns is None:
                raise ValueError("dataset and feature_columns must be provided when using subsampling")

            # subsample controls from the full training fold, not the whole dataset
            train_fold = dataset.iloc[train_idx]

            X_train, y_train, _ = sample_case_control(
                train_fold=train_fold,
                event_col=event_col,
                time_col=time_col,
                n_controls_per_case=n_controls_per_case,
                feature_columns=feature_columns,
                random_state=random_state if random_state is not None else 12
            )
        
        # Scale if requested
        if scale_data:
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
        
        # only pass random_state/n_jobs through if the model actually takes them
        model_params_copy = model_params.copy()
        if random_state is not None:
            import inspect
            if 'random_state' in inspect.signature(model_class.__init__).parameters:
                model_params_copy['random_state'] = random_state
        if n_jobs is not None:
            import inspect
            if 'n_jobs' in inspect.signature(model_class.__init__).parameters:
                model_params_copy['n_jobs'] = n_jobs
        
        model = model_class(**model_params_copy)
        model.fit(X_train, y_train)
        
        for metric in metrics:
            error_occurred = False

            if metric == 'c-index':
                try:
                    pred = model.predict(X_test)
                    c_index = concordance_index_censored(
                        y_test['event'],
                        y_test['time'],
                        pred
                    )[0]
                    results[metric].append(c_index)
                except Exception as e:
                    if verbose:
                        print(f"    Warning: C-index calculation failed: {str(e)}")
                    results[metric].append(np.nan)
                    error_occurred = True
                
            elif metric == 'c-index-ipcw':
                try:
                    pred = model.predict(X_test)
                    tau = min(y_train['time'].max(), y_test['time'].max()) * 0.9
                    c_index_ipcw = concordance_index_ipcw(
                        y_train, y_test, pred, tau=tau
                    )[0]
                    results[metric].append(c_index_ipcw)
                except Exception as e:
                    if verbose:
                        print(f"    Warning: C-index IPCW calculation failed: {str(e)}")
                    results[metric].append(np.nan)
                    error_occurred = True
                    
            elif metric == 'int-brier-score':
                # IBS over the 5th-90th percentile of observed times (percentile rule)
                try:
                    if hasattr(model, 'predict_survival_function'):
                        surv_funcs = model.predict_survival_function(X_test)
                        max_time = min(y_train['time'].max(), y_test['time'].max()) * 0.9

                        times = np.percentile(y_train['time'], np.linspace(5, 90, 10))
                        times = np.unique(times)
                        times = times[(times > 0) & (times <= max_time)]

                        if len(times) > 1:  # need at least 2 points to integrate over
                            predicted_survival_probabilities = np.asarray(
                                [[fn(t) for t in times] for fn in surv_funcs]
                            )
                            ibs = integrated_brier_score(y_train, y_test, predicted_survival_probabilities, times)
                            results[metric].append(ibs)
                        else:
                            results[metric].append(np.nan)
                            error_occurred = True
                    else:
                        results[metric].append(np.nan)
                        error_occurred = True
                except Exception as e:
                    if verbose:
                        print(f"    Warning: Brier score calculation failed: {str(e)}")
                    results[metric].append(np.nan)
                    error_occurred = True

            elif metric.startswith('brier-'):
                # single-time-point Brier scores, e.g. 'brier-5y'
                try:
                    if hasattr(model, 'predict_survival_function'):
                        surv_funcs = model.predict_survival_function(X_test)
                        max_time = min(y_train['time'].max(), y_test['time'].max()) * 0.9

                        time_points = {
                            'brier-6m': 0.5 * 365,
                            'brier-1y': 1 * 365,
                            'brier-2y': 2 * 365,
                            'brier-5y': 5 * 365,
                            'brier-10y': 10 * 365,
                            'brier-20y': 20 * 365
                        }
                        
                        if metric in time_points:
                            time_point = time_points[metric]
                            if time_point <= max_time:
                                # single time point -> brier_score, not integrated_brier_score
                                predicted_survival_probabilities = np.asarray(
                                    [fn(time_point) for fn in surv_funcs]
                                )
                                score = float(brier_score(y_train, y_test, predicted_survival_probabilities, time_point)[1][0])
                                results[metric].append(score)
                            else:
                                results[metric].append(np.nan)
                                error_occurred = True
                        else:
                            results[metric].append(np.nan)
                            error_occurred = True
                    else:
                        results[metric].append(np.nan)
                        error_occurred = True
                except Exception as e:
                    if verbose:
                        print(f"    Warning: {metric} calculation failed: {str(e)}")
                    results[metric].append(np.nan)
                    error_occurred = True
            
            elif metric == 'cumulative-dynamic-auc':
                try:
                    pred = model.predict(X_test)
                    max_time = min(y_train['time'].max(), y_test['time'].max()) * 0.9
                    times = np.percentile(y_train['time'], np.linspace(10, 90, 10))
                    times = np.unique(times)
                    times = times[(times > 0) & (times <= max_time)]

                    if len(times) >= 2:
                        auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, pred, times)
                        results[metric].append(mean_auc)
                    else:
                        results[metric].append(np.nan)
                        error_occurred = True
                except Exception as e:
                    if verbose:
                        print(f"    Warning: Cumulative Dynamic AUC calculation failed: {str(e)}")
                    results[metric].append(np.nan)
                    error_occurred = True
            
            error_flags[metric].append(error_occurred)

    results_df = pd.DataFrame(results)
    results_df.index = [f'Fold_{i+1}' for i in range(fold_num)]

    results_df.loc['Mean'] = results_df.mean()
    results_df.loc['Std'] = results_df.std()

    return results_df, error_flags


def analyze_fold_failures(results_df, metrics=None, threshold_nan_pct=10.0):
    """
    Scan a generic_grid_search results_df for NaN folds (metrics can't fail silently
    if some param combo blows up the model or a fold has too few valid times).

    metrics=None auto-detects from the *_mean columns. threshold_nan_pct sets how
    many failed folds (as a %) a config needs before it's flagged as problematic.

    Returns a dict with 'failed_configs' (list of flagged rows), 'nan_summary'
    (per-metric counts), and 'fold_statistics' (mean/std/min/max per fold column).
    """
    if metrics is None:
        mean_cols = [col for col in results_df.columns if col.endswith('_mean')]
        metrics = [col.replace('_mean', '') for col in mean_cols]
    
    analysis = {
        'metrics': metrics,
        'failed_configs': [],
        'nan_summary': {},
        'fold_statistics': {}
    }
    
    for metric in metrics:
        fold_cols = [col for col in results_df.columns if col.startswith(f'{metric}_fold_')]

        if not fold_cols:
            continue

        nan_counts = results_df[fold_cols].isna().sum(axis=1)
        total_folds = len(fold_cols)
        nan_pct = (nan_counts / total_folds * 100)

        problematic_configs = results_df[nan_pct >= threshold_nan_pct].copy()
        
        if len(problematic_configs) > 0:
            analysis['nan_summary'][metric] = {
                'total_configs': len(results_df),
                'configs_with_nans': len(problematic_configs),
                'percentage': len(problematic_configs) / len(results_df) * 100,
                'max_nan_count': nan_counts.max(),
                'total_folds': total_folds
            }
            
            for idx, row in problematic_configs.iterrows():
                # crude filter to keep just the hyperparams, not the score columns
                config_params = {k: v for k, v in row.items()
                               if not k.endswith(('_mean', '_std', '_fold_1', '_fold_2', '_fold_3', '_fold_4', '_fold_5'))}
                
                fold_values = {col: row[col] for col in fold_cols}
                failed_folds = [col for col, val in fold_values.items() if pd.isna(val)]
                
                analysis['failed_configs'].append({
                    'index': idx,
                    'metric': metric,
                    'params': config_params,
                    'nan_count': nan_counts[idx],
                    'nan_percentage': nan_pct[idx],
                    'failed_folds': failed_folds,
                    'fold_values': fold_values,
                    'mean_value': row.get(f'{metric}_mean', np.nan),
                    'std_value': row.get(f'{metric}_std', np.nan)
                })
        
        fold_stats = {}
        for fold_col in fold_cols:
            fold_data = results_df[fold_col].dropna()
            if len(fold_data) > 0:
                fold_stats[fold_col] = {
                    'mean': fold_data.mean(),
                    'std': fold_data.std(),
                    'min': fold_data.min(),
                    'max': fold_data.max(),
                    'nan_count': results_df[fold_col].isna().sum(),
                    'valid_count': len(fold_data)
                }
        
        analysis['fold_statistics'][metric] = fold_stats
    
    return analysis


def print_fold_analysis(analysis, top_n=10):
    """Pretty-print the dict from analyze_fold_failures; top_n caps how many failed
    configs get listed in detail."""
    print("=" * 100)
    print("FOLD-LEVEL FAILURE ANALYSIS")
    print("=" * 100)
    
    if not analysis['nan_summary']:
        print("\nNo fold failures detected -- all configurations completed successfully.")
        return
    
    print("\nSUMMARY BY METRIC:")
    print("-" * 100)
    for metric, summary in analysis['nan_summary'].items():
        print(f"\n{metric}:")
        print(f"  Configurations with failures: {summary['configs_with_nans']}/{summary['total_configs']} "
              f"({summary['percentage']:.1f}%)")
        print(f"  Maximum NaN count: {summary['max_nan_count']}/{summary['total_folds']} folds")
    
    if analysis['failed_configs']:
        print(f"\n\nTOP {top_n} FAILED CONFIGURATIONS:")
        print("-" * 100)

        sorted_failures = sorted(analysis['failed_configs'],
                               key=lambda x: x['nan_count'], 
                               reverse=True)[:top_n]
        
        for i, config in enumerate(sorted_failures, 1):
            print(f"\n{i}. Configuration Index: {config['index']}")
            print(f"   Metric: {config['metric']}")
            print(f"   Failed folds: {config['nan_count']} ({config['nan_percentage']:.1f}%)")
            print(f"   Parameters: {config['params']}")
            print(f"   Mean: {config['mean_value']:.4f if not pd.isna(config['mean_value']) else 'NaN'}, "
                  f"Std: {config['std_value']:.4f if not pd.isna(config['std_value']) else 'NaN'}")
            print(f"   Failed fold names: {config['failed_folds']}")
    
    print("\n\nFOLD-LEVEL STATISTICS:")
    print("-" * 100)
    for metric, fold_stats in analysis['fold_statistics'].items():
        print(f"\n{metric}:")
        for fold_name, stats in fold_stats.items():
            print(f"  {fold_name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                  f"valid={stats['valid_count']}, nan={stats['nan_count']}")
    
    print("\n" + "=" * 100)
