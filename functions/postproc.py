from xml.parsers.expat import model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from scipy.special import expit, logit
from scipy.optimize import minimize
from sksurv.util import Surv
from sksurv.metrics import brier_score
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.utils import compute_class_weight

# Among the top N hyperparameter configs by mean performance, pick the most stable one
# (lowest std/mean) rather than just taking the single best mean - avoids picking a
# config that got lucky on a couple of folds.
def find_lowest_cv_config(df, metric='c-index-ipcw', top_n=10):
    """Rank configs by mean `metric`, then within the top_n pick the one with lowest CV (std/mean).

    Returns (best_idx, best_cv, best_mean, best_std, best_rank), where best_rank is the
    config's position (1 = best) in the full mean-sorted list, not just among the top_n.
    """
    mean_col = f'{metric}_mean'
    std_col = f'{metric}_std'

    top_configs = df.nlargest(top_n, mean_col)
    cv_values = top_configs[std_col] / top_configs[mean_col]
    best_idx = cv_values.idxmin()

    # rank by mean in the full (unfiltered) dataframe, not just among the top_n
    df_sorted = df.sort_values(mean_col, ascending=False).reset_index(drop=False)
    best_rank = df_sorted[df_sorted['index' if 'index' in df_sorted.columns else df_sorted.columns[0]] == best_idx].index[0] + 1
    
    return (best_idx, 
            cv_values[best_idx], 
            df.loc[best_idx, mean_col], 
            df.loc[best_idx, std_col],
            best_rank)

# ---- drop grid-search rows with gaps (a fold that errored out / never finished) ----
def clean_cox_data(df):
    """Drop rows with any NaN in the c-index or c-index-ipcw fold columns, so only
    configs where every fold actually completed survive."""
    cindex_fold_cols = [col for col in df.columns if col.startswith('c-index_repeat_') and '_fold_' in col]
    ipcw_fold_cols = [col for col in df.columns if col.startswith('c-index-ipcw_repeat_') and '_fold_' in col]
    all_fold_cols = cindex_fold_cols + ipcw_fold_cols

    rows_before = len(df)
    df_clean = df.dropna(subset=all_fold_cols, how='any')
    rows_after = len(df_clean)
    rows_removed = rows_before - rows_after
    
    print(f"  Checked {len(cindex_fold_cols)} c-index fold columns and {len(ipcw_fold_cols)} c-index-ipcw fold columns")
    print(f"  Removed {rows_removed} rows with missing fold values ({rows_before} -> {rows_after})")
    
    return df_clean


# ---------------------------------------------------------------------------
# calibration curves
# ---------------------------------------------------------------------------

def _fit_beta_calibration(pred_probs, true_labels):
    """Fit a beta calibration model: p_cal = sigmoid(a + b*logit(p) + c*logit(p)^2), a
    3-parameter logistic regression on the logit of the raw prediction (Kull et al. 2017).
    Falls back to plain Platt scaling (c fixed at 0) if the BFGS fit doesn't converge.

    Returns dict {'a', 'b', 'c'}.
    """
    eps = 1e-15
    pred_probs_clipped = np.clip(pred_probs, eps, 1 - eps)
    logit_preds = logit(pred_probs_clipped)

    def neg_log_likelihood(params):
        a, b, c = params
        z = a + b * logit_preds + c * (logit_preds ** 2)
        calibrated_probs = expit(z)
        calibrated_probs = np.clip(calibrated_probs, eps, 1 - eps)
        loss = -np.mean(
            true_labels * np.log(calibrated_probs) +
            (1 - true_labels) * np.log(1 - calibrated_probs)
        )
        return loss

    initial_params = [0.0, 1.0, 0.0]  # a=0, b=1, c=0 -> identity (no recalibration)
    result = minimize(neg_log_likelihood, initial_params, method='BFGS')

    if not result.success:
        result = minimize(lambda params: neg_log_likelihood([params[0], params[1], 0.0]),
                         [0.0, 1.0], method='BFGS')
        a, b = result.x
        c = 0.0
    else:
        a, b, c = result.x

    return {'a': a, 'b': b, 'c': c}


def _apply_beta_calibration(pred_prob, params):
    """Apply the fitted beta-calibration params to a single raw probability."""
    eps = 1e-15
    pred_prob_clipped = np.clip(pred_prob, eps, 1 - eps)
    
    logit_pred = logit(pred_prob_clipped)
    
    a, b, c = params['a'], params['b'], params['c']
    z = a + b * logit_pred + c * (logit_pred ** 2)
    
    calibrated_prob = expit(z)
    return calibrated_prob


def collect_cv_predictions_for_calibration(model_class, X, y, model_params, 
                                          time_points,
                                          scale_data=False, scaler_type='standard',
                                          continuous_features=None,
                                          random_state=None, n_jobs=None,
                                          n_controls_per_case=None, dataset=None,
                                          feature_columns=None,
                                          event_col='Regional LN recurrence',
                                          time_col='Survival days',
                                          n_splits=5, n_repeats=10, base_random_state=42,
                                          recalibrate=False, recalibration_type='platt',
                                          sample_weight=None,
                                          calculate_threshold=False,
                                          verbose=True):
    """Run repeated stratified CV, refit model_class on each training fold, and stash the
    out-of-fold survival predictions at each time point in time_points. This is the workhorse
    behind the calibration plots below - out-of-fold predictions are what make the calibration
    curves honest instead of optimistic.

    Most args are the usual CV plumbing (scale_data/scaler_type/continuous_features for
    feature scaling, n_controls_per_case/dataset/feature_columns for case-control
    subsampling via sample_case_control, n_splits/n_repeats/base_random_state for the CV
    itself). A few are worth calling out:
    - recalibrate/recalibration_type: if recalibrate=True, fits a secondary calibrator
      ('platt', 'beta', or 'isotonic') on the training fold's predictions per time point
      and applies it to the validation predictions before returning them.
    - sample_weight: pass True to have class weights split per fold automatically, or an
      array-like matching len(X) to split it yourself.
    - calculate_threshold: when True, also finds one risk threshold per (fold, time point)
      on the training data that minimizes false negatives (ties broken by lower FP, then
      higher threshold), applies it on validation, and returns a second dataframe of
      fold-level confusion metrics.

    Returns predictions_df (repeat_idx, fold_idx, sample_idx, event, time, plus one
    pred_surv_XXXd column per time point), and also threshold_metrics_df if
    calculate_threshold=True.
    """
    from functions.repeated_cv import sample_case_control
    
    if verbose:
        print(f"Collecting predictions for calibration...")
        print(f"Time points: {time_points}")
        print(f"CV: {n_repeats} repeats x {n_splits} folds = {n_repeats * n_splits} models")
        if recalibrate:
            print(f"Recalibration: {recalibration_type}")
        if calculate_threshold:
            print("Threshold mode: ON (optimize per-fold, per-timepoint thresholds)")
    
    all_predictions = []
    all_threshold_metrics = []
    
    for repeat_idx in range(n_repeats):
        random_state_cv = base_random_state + repeat_idx
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state_cv)
        
        if verbose:
            print(f"\nRepeat {repeat_idx + 1}/{n_repeats}")
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y['event'])):
            if verbose:
                print(f"  Fold {fold_idx + 1}/{n_splits}", end='... ')
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Split sample_weight if provided
            if sample_weight is True:
                sample_weight_train = compute_class_weight(class_weight="balanced", classes=[np.unique(y_train['event'])], y=y_train)
            else:
                sample_weight_train = None
            
            # Handle subsampling
            if n_controls_per_case is not None and n_controls_per_case != 'Full':
                train_dataset = dataset.iloc[train_idx].copy()
                # sample_case_control returns data_x, data_y, sub_dataset
                X_train, y_train, _ = sample_case_control(
                    train_dataset, 
                    event_col=event_col, 
                    time_col=time_col,
                    n_controls_per_case=n_controls_per_case,
                    feature_columns=feature_columns,
                    random_state=random_state_cv + fold_idx
                )
            
            # Scale data if needed
            if scale_data:
                if continuous_features is not None:
                    # Only scale continuous features
                    scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
                    X_train_scaled = X_train.copy()
                    X_val_scaled = X_val.copy()
                    X_train_scaled[continuous_features] = scaler.fit_transform(X_train[continuous_features])
                    X_val_scaled[continuous_features] = scaler.transform(X_val[continuous_features])
                    X_train, X_val = X_train_scaled, X_val_scaled
                else:
                    # Scale all features
                    scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
                    X_train = pd.DataFrame(scaler.fit_transform(X_train), 
                                          columns=X_train.columns, index=X_train.index)
                    X_val = pd.DataFrame(scaler.transform(X_val), 
                                        columns=X_val.columns, index=X_val.index)
            
            # Train model - handle different parameter requirements for different model types
            try:
                import inspect
                model_signature = inspect.signature(model_class.__init__)
                model_init_params = model_params.copy()
                
                # Add random_state if the model accepts it
                if 'random_state' in model_signature.parameters and random_state is not None:
                    model_init_params['random_state'] = random_state
                
                # Add n_jobs if the model accepts it
                if 'n_jobs' in model_signature.parameters and n_jobs is not None:
                    model_init_params['n_jobs'] = n_jobs
                
                if verbose:
                    print(f"Creating model with params: {model_init_params}")
                
                model = model_class(**model_init_params)
                
                # Fit model with or without sample_weight
                if sample_weight_train is not None:
                    model.fit(X_train, y_train, sample_weight=sample_weight_train)
                else:
                    model.fit(X_train, y_train)
                
                # Get survival functions for validation set
                surv_funcs_val = model.predict_survival_function(X_val)
                surv_funcs_train = None
                if recalibrate or calculate_threshold:
                    surv_funcs_train = model.predict_survival_function(X_train)
                
                # fit a calibrator per time point on the training fold's own predictions,
                # then apply it to validation below - keeps the calibrator itself out-of-fold too
                calibrators = {}
                if recalibrate:
                    for t in time_points:
                        train_preds = []
                        for i in range(len(y_train)):
                            try:
                                if callable(surv_funcs_train[i]):
                                    train_preds.append(surv_funcs_train[i](t))
                                else:
                                    train_preds.append(surv_funcs_train[i](t))
                            except:
                                train_preds.append(np.nan)

                        train_preds = np.array(train_preds)

                        # observed = 1 if survived past t, 0 if event occurred at/before t
                        train_outcomes = ((y_train['time'] > t) |
                                        ((y_train['time'] <= t) & (y_train['event'] == 0))).astype(int)

                        valid_mask = ~np.isnan(train_preds)
                        train_preds_valid = train_preds[valid_mask]
                        train_outcomes_valid = train_outcomes[valid_mask]

                        if len(train_preds_valid) > 0 and len(np.unique(train_outcomes_valid)) > 1:
                            if recalibration_type == 'platt':
                                calibrator = LogisticRegression(solver='lbfgs', max_iter=1000)
                                calibrator.fit(train_preds_valid.reshape(-1, 1), train_outcomes_valid)
                            elif recalibration_type == 'beta':
                                calibrator = _fit_beta_calibration(train_preds_valid, train_outcomes_valid)
                            elif recalibration_type == 'isotonic':
                                calibrator = IsotonicRegression(out_of_bounds='clip')
                                calibrator.fit(train_preds_valid, train_outcomes_valid)
                            else:
                                raise ValueError(f"Unknown recalibration_type: {recalibration_type}")
                            
                            calibrators[t] = calibrator
                        else:
                            calibrators[t] = None

                # Optional threshold optimization on the training fold
                thresholds_by_time = {}
                if calculate_threshold:
                    for t in time_points:
                        train_preds = []
                        for i in range(len(y_train)):
                            try:
                                if callable(surv_funcs_train[i]):
                                    train_preds.append(surv_funcs_train[i](t))
                                else:
                                    train_preds.append(surv_funcs_train[i](t))
                            except Exception:
                                train_preds.append(np.nan)

                        train_preds = np.array(train_preds)

                        # Apply recalibration to training predictions if enabled
                        if recalibrate and t in calibrators and calibrators[t] is not None:
                            for i in range(len(train_preds)):
                                if np.isnan(train_preds[i]):
                                    continue
                                if recalibration_type == 'platt':
                                    train_preds[i] = calibrators[t].predict_proba([[train_preds[i]]])[0, 1]
                                elif recalibration_type == 'beta':
                                    train_preds[i] = _apply_beta_calibration(train_preds[i], calibrators[t])
                                elif recalibration_type == 'isotonic':
                                    train_preds[i] = calibrators[t].predict([train_preds[i]])[0]
                                    train_preds[i] = np.clip(train_preds[i], 0.0, 1.0)

                        valid_mask = ~np.isnan(train_preds)
                        if valid_mask.sum() == 0:
                            thresholds_by_time[t] = 0.5
                            continue

                        # Event-positive target at horizon t
                        y_true_train = ((y_train['time'] <= t) & (y_train['event'] == 1)).astype(int)
                        y_true_train = y_true_train[valid_mask]
                        risk_train = 1.0 - train_preds[valid_mask]

                        # Candidates include extremes to allow all-positive or all-negative predictions
                        candidates = np.unique(risk_train)
                        if len(candidates) == 0:
                            thresholds_by_time[t] = 0.5
                            continue

                        candidates = np.concatenate(([risk_train.min() - 1e-12], candidates, [risk_train.max() + 1e-12]))

                        best_score = None
                        best_threshold = 0.5

                        for thr in candidates:
                            pred_event_train = (risk_train >= thr).astype(int)

                            tp_train = np.sum((pred_event_train == 1) & (y_true_train == 1))
                            fp_train = np.sum((pred_event_train == 1) & (y_true_train == 0))
                            fn_train = np.sum((pred_event_train == 0) & (y_true_train == 1))

                            # Priority: minimize FN, then FP, then prefer higher threshold
                            score = (fn_train, fp_train, -thr)
                            if best_score is None or score < best_score:
                                best_score = score
                                best_threshold = thr

                        thresholds_by_time[t] = float(best_threshold)
                
                # Extract predictions at each time point
                fold_eval_data = {
                    t: {'pred_surv': [], 'event': [], 'time': []}
                    for t in time_points
                }

                for i, sample_idx in enumerate(val_idx):
                    pred_dict = {
                        'repeat_idx': repeat_idx,
                        'fold_idx': fold_idx,
                        'sample_idx': sample_idx,
                        'event': y_val['event'][i],
                        'time': y_val['time'][i]
                    }
                    
                    for t in time_points:
                        try:
                            # surv_funcs_val[i] is callable either way (step function); the
                            # branch is a leftover from handling an older model interface
                            if callable(surv_funcs_val[i]):
                                pred_prob = surv_funcs_val[i](t)
                            else:
                                pred_prob = surv_funcs_val[i](t)

                            if recalibrate and t in calibrators and calibrators[t] is not None:
                                if recalibration_type == 'platt':
                                    pred_prob = calibrators[t].predict_proba([[pred_prob]])[0, 1]
                                elif recalibration_type == 'beta':
                                    pred_prob = _apply_beta_calibration(pred_prob, calibrators[t])
                                elif recalibration_type == 'isotonic':
                                    pred_prob = calibrators[t].predict([pred_prob])[0]
                                    pred_prob = np.clip(pred_prob, 0.0, 1.0)  # isotonic can slightly overshoot [0,1]
                            
                            pred_dict[f'pred_surv_{int(t)}d'] = pred_prob

                            if calculate_threshold:
                                fold_eval_data[t]['pred_surv'].append(pred_prob)
                                fold_eval_data[t]['event'].append(y_val['event'][i])
                                fold_eval_data[t]['time'].append(y_val['time'][i])
                        except Exception as e_pred:
                            if verbose:
                                print(f"Warning: Could not predict at time {t} for sample {i}: {e_pred}")
                            pred_dict[f'pred_surv_{int(t)}d'] = np.nan

                            if calculate_threshold:
                                fold_eval_data[t]['pred_surv'].append(np.nan)
                                fold_eval_data[t]['event'].append(y_val['event'][i])
                                fold_eval_data[t]['time'].append(y_val['time'][i])
                    
                    all_predictions.append(pred_dict)

                if calculate_threshold:
                    for t in time_points:
                        pred_surv_val = np.array(fold_eval_data[t]['pred_surv'])
                        event_val = np.array(fold_eval_data[t]['event'])
                        time_val = np.array(fold_eval_data[t]['time'])

                        valid_mask = ~np.isnan(pred_surv_val)
                        if valid_mask.sum() == 0:
                            all_threshold_metrics.append({
                                'repeat_idx': repeat_idx,
                                'fold_idx': fold_idx,
                                'time_days': int(t),
                                'time_years': float(t / 365),
                                'threshold': thresholds_by_time.get(t, 0.5),
                                'n_train_samples': int(len(y_train)),
                                'n_val_samples': 0,
                                'n_train_events': int(np.sum((y_train['time'] <= t) & (y_train['event'] == 1))),
                                'n_val_events': 0,
                                'TP': 0,
                                'FP': 0,
                                'TN': 0,
                                'FN': 0,
                                'sensitivity': np.nan,
                                'specificity': np.nan,
                                'PPV': np.nan,
                                'NPV': np.nan,
                                'accuracy': np.nan,
                                'F1_score': np.nan
                            })
                            continue

                        pred_surv_val = pred_surv_val[valid_mask]
                        event_val = event_val[valid_mask]
                        time_val = time_val[valid_mask]

                        y_true_val = ((time_val <= t) & (event_val == 1)).astype(int)
                        risk_val = 1.0 - pred_surv_val
                        threshold = thresholds_by_time.get(t, 0.5)
                        pred_event_val = (risk_val >= threshold).astype(int)

                        tp = np.sum((pred_event_val == 1) & (y_true_val == 1))
                        fp = np.sum((pred_event_val == 1) & (y_true_val == 0))
                        tn = np.sum((pred_event_val == 0) & (y_true_val == 0))
                        fn = np.sum((pred_event_val == 0) & (y_true_val == 1))

                        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
                        accuracy = (tp + tn) / len(y_true_val) if len(y_true_val) > 0 else 0.0
                        f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0.0

                        all_threshold_metrics.append({
                            'repeat_idx': repeat_idx,
                            'fold_idx': fold_idx,
                            'time_days': int(t),
                            'time_years': float(t / 365),
                            'threshold': float(threshold),
                            'n_train_samples': int(len(y_train)),
                            'n_val_samples': int(len(y_true_val)),
                            'n_train_events': int(np.sum((y_train['time'] <= t) & (y_train['event'] == 1))),
                            'n_val_events': int(np.sum(y_true_val == 1)),
                            'TP': int(tp),
                            'FP': int(fp),
                            'TN': int(tn),
                            'FN': int(fn),
                            'sensitivity': float(sensitivity),
                            'specificity': float(specificity),
                            'PPV': float(ppv),
                            'NPV': float(npv),
                            'accuracy': float(accuracy),
                            'F1_score': float(f1)
                        })
                
                if verbose:
                    print("Done")
                    
            except Exception as e:
                if verbose:
                    print(f"Error in fold {fold_idx + 1}: {e}")
                    import traceback
                    traceback.print_exc()
    
    # Check if any predictions were collected
    if len(all_predictions) == 0:
        raise ValueError(
            "No predictions were collected! All folds failed. "
            "Check the error messages above for details. "
            "Common issues: incorrect model parameters, incompatible data types, or model fitting failures."
        )
    
    predictions_df = pd.DataFrame(all_predictions)
    
    if verbose:
        print(f"\nTotal predictions collected: {len(predictions_df)}")
        print(f"Unique samples: {predictions_df['sample_idx'].nunique()}")

    if calculate_threshold:
        threshold_metrics_df = pd.DataFrame(all_threshold_metrics)
        if verbose:
            print(f"Total threshold-metric rows: {len(threshold_metrics_df)}")
        return predictions_df, threshold_metrics_df
    
    return predictions_df


def calculate_calibration_curve(predictions_df, time_point, n_bins=10, strategy='quantile'):
    """Bin predicted survival probabilities at time_point and compare against the observed
    proportion in each bin (the classic calibration-curve computation). strategy='quantile'
    gives equal-count bins, 'uniform' gives equal-width bins.

    Returns a DataFrame with bin, pred_mean, count, obs_mean, obs_se, obs_ci_lower/upper
    (95% CI via normal approximation).
    """
    col_name = f'pred_surv_{int(time_point)}d'

    if col_name not in predictions_df.columns:
        raise ValueError(f"Time point {time_point} not in predictions")

    df = predictions_df[['sample_idx', 'event', 'time', col_name]].dropna()

    # observed = 1 if survived past time_point, 0 if event occurred at/before it
    df['observed'] = ((df['time'] > time_point) | ((df['time'] <= time_point) & (df['event'] == 0))).astype(int)

    if strategy == 'quantile':
        df['bin'] = pd.qcut(df[col_name], q=n_bins, labels=False, duplicates='drop')
    else:  # uniform
        df['bin'] = pd.cut(df[col_name], bins=n_bins, labels=False)

    calibration = df.groupby('bin').agg({
        col_name: ['mean', 'count'],
        'observed': 'mean'
    }).reset_index()

    calibration.columns = ['bin', 'pred_mean', 'count', 'obs_mean']

    # binomial proportion SE, then a plain normal-approx 95% CI
    calibration['obs_se'] = np.sqrt(
        calibration['obs_mean'] * (1 - calibration['obs_mean']) / calibration['count']
    )

    calibration['obs_ci_lower'] = calibration['obs_mean'] - 1.96 * calibration['obs_se']
    calibration['obs_ci_upper'] = calibration['obs_mean'] + 1.96 * calibration['obs_se']

    calibration['obs_ci_lower'] = calibration['obs_ci_lower'].clip(0, 1)
    calibration['obs_ci_upper'] = calibration['obs_ci_upper'].clip(0, 1)
    
    return calibration


def plot_calibration_curves(predictions_df, time_points, n_bins=10, strategy='quantile',
                           figsize=(15, 5), save_path=None, xlim=None, ylim=None,
                           show_error_bars=True, error_bar_type='ci'):
    """Grid of calibration plots, one panel per time point, each built via
    calculate_calibration_curve. error_bar_type controls whether the bars show 'se' or
    the 95% 'ci' (default). xlim/ylim default to [0, 1] if not given.

    Returns fig, axes, and a dict of {time_point: calibration DataFrame}.
    """
    n_plots = len(time_points)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    calibration_results = {}
    
    for idx, t in enumerate(time_points):
        ax = axes[idx]
        
        try:
            cal_data = calculate_calibration_curve(predictions_df, t, n_bins, strategy)
            calibration_results[t] = cal_data
            
            # Plot calibration curve with error bars
            if show_error_bars:
                if error_bar_type == 'ci':
                    # Plot with 95% confidence intervals
                    yerr_lower = cal_data['obs_mean'] - cal_data['obs_ci_lower']
                    yerr_upper = cal_data['obs_ci_upper'] - cal_data['obs_mean']
                    yerr = [yerr_lower, yerr_upper]
                else:  # 'se'
                    # Plot with standard error
                    yerr = cal_data['obs_se']
                
                ax.errorbar(cal_data['pred_mean'], cal_data['obs_mean'], 
                           yerr=yerr, fmt='o-', 
                           label='Calibration curve', linewidth=2, markersize=8,
                           capsize=5, capthick=2)
            else:
                # Plot without error bars
                ax.plot(cal_data['pred_mean'], cal_data['obs_mean'], 'o-', 
                       label='Calibration curve', linewidth=2, markersize=8)
            
            # Plot perfect calibration line
            min_val = min(cal_data['pred_mean'].min(), cal_data['obs_mean'].min())
            max_val = max(cal_data['pred_mean'].max(), cal_data['obs_mean'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', 
                   label='Perfect calibration', alpha=0.5)
            
            # Add sample size info
            for _, row in cal_data.iterrows():
                ax.text(row['pred_mean'], row['obs_mean'], 
                       f"n={int(row['count'])}", 
                       fontsize=8, ha='center', va='bottom')
            
            ax.set_xlabel('Predicted Survival Probability', fontsize=12)
            ax.set_ylabel('Observed Survival Proportion', fontsize=12)
            ax.set_title(f'Calibration at {int(t)} days ({t/365:.1f} years)', fontsize=13)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(xlim if xlim is not None else [0, 1])
            ax.set_ylim(ylim if ylim is not None else [0, 1])
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
            ax.set_title(f'Time point: {t} days')
    
    # Hide extra subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Calibration plot saved to {save_path}")
    
    return fig, axes, calibration_results


def calculate_loess_calibration(predictions_df, time_point, frac=0.3, n_points=100):
    """Same idea as calculate_calibration_curve but smoothed with lowess instead of binned -
    avoids the arbitrary bin-edge artifacts, at the cost of losing per-bin CI. frac controls
    the smoothing window (statsmodels' lowess fraction); n_points isn't actually used since
    lowess picks its own output grid from the input x values (kept for API symmetry with the
    binned version).

    Returns (loess_data, raw_data): the smoothed curve (pred_prob, smooth_obs_prob) and the
    underlying (pred_prob, observed) points it was fit on.
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess

    col_name = f'pred_surv_{int(time_point)}d'

    if col_name not in predictions_df.columns:
        raise ValueError(f"Time point {time_point} not in predictions")

    df = predictions_df[['sample_idx', 'event', 'time', col_name]].dropna()

    # observed = 1 if survived past time_point, 0 if event occurred at/before it
    df['observed'] = ((df['time'] > time_point) | ((df['time'] <= time_point) & (df['event'] == 0))).astype(int)

    x = df[col_name].values
    y = df['observed'].values

    smoothed = lowess(y, x, frac=frac, return_sorted=True)  # returns [x, smoothed_y] pairs

    loess_data = pd.DataFrame({
        'pred_prob': smoothed[:, 0],
        'smooth_obs_prob': smoothed[:, 1]
    })
    
    raw_data = df[[col_name, 'observed']].rename(columns={col_name: 'pred_prob'})
    
    return loess_data, raw_data


def plot_loess_calibration_curves(predictions_df, time_points, frac=0.3, 
                                  show_raw=True, raw_alpha=0.1, raw_sample=None,
                                  figsize=(15, 5), save_path=None, xlim=None, ylim=None):
    """Grid version of calculate_loess_calibration, one panel per time point. Set
    raw_sample if the scatter of raw points gets too cluttered for large datasets;
    frac ~0.5 smooths more, ~0.2 hugs the data more closely.
    """
    n_plots = len(time_points)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    loess_results = {}
    
    for idx, t in enumerate(time_points):
        ax = axes[idx]
        
        try:
            loess_data, raw_data = calculate_loess_calibration(predictions_df, t, frac=frac)
            loess_results[t] = {'loess': loess_data, 'raw': raw_data}
            
            # Plot raw data points if requested
            if show_raw:
                raw_to_plot = raw_data
                if raw_sample is not None and len(raw_data) > raw_sample:
                    raw_to_plot = raw_data.sample(n=raw_sample, random_state=42)
                
                ax.scatter(raw_to_plot['pred_prob'], raw_to_plot['observed'], 
                          alpha=raw_alpha, s=20, c='gray', label='Observations')
            
            # Plot Loess-smoothed calibration curve
            ax.plot(loess_data['pred_prob'], loess_data['smooth_obs_prob'], 
                   linewidth=3, color='blue', label='Loess calibration curve')
            
            # Plot perfect calibration line
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', alpha=0.5)
            
            # Add statistics
            n_samples = len(raw_data)
            ax.text(0.05, 0.95, f'n = {n_samples}', 
                   transform=ax.transAxes, fontsize=10, 
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            ax.set_xlabel('Predicted Survival Probability', fontsize=12)
            ax.set_ylabel('Observed Survival Proportion', fontsize=12)
            ax.set_title(f'Loess Calibration at {int(t)} days ({t/365:.1f} years)', fontsize=13)
            ax.legend(fontsize=9, loc='lower right')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(xlim if xlim is not None else [0, 1])
            ax.set_ylim(ylim if ylim is not None else [0, 1])
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
            ax.set_title(f'Time point: {t} days')
    
    # Hide extra subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loess calibration plot saved to {save_path}")
    
    return fig, axes, loess_results


def plot_jittered_risk_scores(predictions_df, time_points, 
                              jitter_amount=0.3, alpha=0.6,
                              event_color='red', no_event_color='blue', censored_color='gray',
                              show_legend=True, show_stats=True,
                              figsize=(15, 5), save_path=None):
    """Jittered strip plot of predicted risk at each time point, split into event /
    no-event / censored groups (colored separately) - a quick visual check of how well
    the model separates the groups, backed up by Cohen's d per time point in the
    returned stats dict.
    """
    n_plots = len(time_points)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    stats_dict = {}
    
    for idx, t in enumerate(time_points):
        ax = axes[idx]
        col_name = f'pred_surv_{int(t)}d'
        
        try:
            if col_name not in predictions_df.columns:
                raise ValueError(f"Time point {t} not in predictions")
            
            # Get data for this time point
            df = predictions_df[['sample_idx', 'event', 'time', col_name]].dropna()
            
            # Determine observed outcome at this time point
            # Event occurred = patient had event at or before time point
            df['had_event_at_t'] = (df['event'] == 1) & (df['time'] <= t)
            # Censored at or before t = no observed event status beyond this horizon
            df['censored_at_or_before_t'] = (df['event'] == 0) & (df['time'] <= t)
            
            # Separate into three groups
            event_data = df[df['had_event_at_t']]
            censored_data = df[df['censored_at_or_before_t']]
            no_event_data = df[(~df['had_event_at_t']) & (~df['censored_at_or_before_t'])]
            
            # Generate jittered y-positions
            np.random.seed(42)
            event_y = np.random.uniform(-jitter_amount, jitter_amount, len(event_data))
            censored_y = np.random.uniform(-jitter_amount, jitter_amount, len(censored_data))
            no_event_y = np.random.uniform(-jitter_amount, jitter_amount, len(no_event_data))
            
            # Plot points (event-free first, censored second, event on top)
            if len(no_event_data) > 0:
                ax.scatter(no_event_data[col_name], no_event_y, 
                          c=no_event_color, alpha=alpha, s=30,
                          label=f'No event by t (observed) (n={len(no_event_data)})', edgecolors='none',
                          zorder=2)

            if len(censored_data) > 0:
                ax.scatter(censored_data[col_name], censored_y,
                          c=censored_color, alpha=alpha, s=30,
                          label=f'Censored by t (n={len(censored_data)})', edgecolors='none',
                          zorder=2.5)
            
            if len(event_data) > 0:
                ax.scatter(event_data[col_name], event_y, 
                          c=event_color, alpha=alpha, s=30, 
                          label=f'Event (n={len(event_data)})', edgecolors='none',
                          zorder=3)
            
            # Calculate separation statistics
            if len(event_data) > 0 and len(no_event_data) > 0:
                event_mean = event_data[col_name].mean()
                no_event_mean = no_event_data[col_name].mean()
                event_median = event_data[col_name].median()
                no_event_median = no_event_data[col_name].median()
                
                # Calculate effect size (Cohen's d)
                pooled_std = np.sqrt(
                    (event_data[col_name].std()**2 + no_event_data[col_name].std()**2) / 2
                )
                cohens_d = (no_event_mean - event_mean) / pooled_std if pooled_std > 0 else 0
                
                stats_dict[t] = {
                    'n_events': len(event_data),
                    'n_censored': len(censored_data),
                    'n_no_events': len(no_event_data),
                    'event_mean': event_mean,
                    'no_event_mean': no_event_mean,
                    'event_median': event_median,
                    'no_event_median': no_event_median,
                    'cohens_d': cohens_d
                }
                
                # Add vertical lines for means
                ax.axvline(event_mean, color=event_color, linestyle='--', 
                          alpha=0.7, linewidth=2, label='Event mean')
                ax.axvline(no_event_mean, color=no_event_color, linestyle='--', 
                          alpha=0.7, linewidth=2, label='No event mean')
                
                # Add statistics text box
                if show_stats:
                    stats_text = (
                        f'Mean difference: {no_event_mean - event_mean:.3f}\n'
                        f"Cohen's d: {cohens_d:.2f}"
                    )
                    ax.text(0.05, 0.95, stats_text, 
                           transform=ax.transAxes, fontsize=9,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Formatting
            ax.set_xlabel('Predicted Survival Probability', fontsize=12)
            ax.set_ylabel('', fontsize=12)
            ax.set_title(f'Risk Score Distribution at {int(t)} days ({t/365:.1f} years)', 
                        fontsize=13)
            ax.set_yticks([])  # Hide y-axis ticks since they're just jitter
            ax.set_xlim([0, 1])
            ax.grid(True, alpha=0.3, axis='x')
            
            if show_legend and idx == 0:  # Only show legend on first plot
                ax.legend(fontsize=8, loc='upper left')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
            ax.set_title(f'Time point: {t} days')
    
    # Hide extra subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Jittered risk score plot saved to {save_path}")
    
    return fig, axes, stats_dict


def plot_individual_survival_curves(model_class, X, y, model_params,
                                    patient_indices=None, 
                                    n_patients=10, select_events_only=True,
                                    scale_data=False, scaler_type='standard',
                                    continuous_features=None,
                                    n_controls_per_case=None, dataset=None,
                                    feature_columns=None,
                                    event_col='Regional LN recurrence',
                                    time_col='Survival days',
                                    n_splits=4, n_repeats=1, random_state=42,
                                    time_range=None, n_time_points=100,
                                    figsize=(15, 10), save_path=None,
                                    show_event_markers=True, show_censoring_markers=True,
                                    event_marker_size=100, censoring_marker_size=80,
                                    colormap='tab10', alpha=0.8,
                                    show_legend=True, legend_outside=True,
                                    show_confidence_interval=True, ci_alpha=0.2,
                                    compute_weights=False,
                                    verbose=True):
    """Plot per-patient survival curves using out-of-fold CV predictions (each patient's
    curve comes from a model that never saw them in training, so this isn't the usual
    overfit-flattered training-set plot).

    Set n_repeats > 1 to get a spread of predictions per patient and draw 95% CI bands
    (needs several repeats to be meaningful - 5-10 is reasonable). compute_weights lets
    you upweight events during fitting: True derives the ratio automatically from
    censored/event counts in the training fold, or pass a number to set the event weight
    directly (censored stays at 1.0). n_controls_per_case/dataset/feature_columns are for
    case-control subsampling on the training folds, same as elsewhere in this module.

    patient_indices lets you pick specific patients; otherwise n_patients are chosen
    randomly (events only, by default, since those are usually the interesting cases).

    Returns fig, ax, a patient_info DataFrame, and a dict of per-patient survival curve
    data (time points, mean survival, all repeat predictions, and CI bounds if computed).
    """
    if verbose:
        if n_repeats > 1:
            print(f"Collecting survival curves using {n_repeats} repeats x {n_splits}-fold CV...")
            print(f"  Total models: {n_repeats * n_splits}")
        else:
            print(f"Collecting survival curves using {n_splits}-fold CV...")
    
    # Select patients to plot
    if patient_indices is None:
        if select_events_only:
            # Get indices of patients with events
            event_indices = np.where(y['event'] == 1)[0]
            if len(event_indices) == 0:
                raise ValueError("No patients with events found in the data")
            # Select random subset if needed
            if len(event_indices) > n_patients:
                np.random.seed(42)
                patient_indices = np.random.choice(event_indices, size=n_patients, replace=False)
            else:
                patient_indices = event_indices[:n_patients]
        else:
            # Select random patients
            if len(y) > n_patients:
                np.random.seed(42)
                patient_indices = np.random.choice(len(y), size=n_patients, replace=False)
            else:
                patient_indices = np.arange(len(y))
    
    patient_indices = np.array(patient_indices)
    
    # Determine time range
    if time_range is None:
        max_time = y['time'].max()
        time_range = (0, max_time)
    
    time_points = np.linspace(time_range[0], time_range[1], n_time_points)
    
    # Perform repeated cross-validation to get survival functions for all patients
    # Store multiple predictions per patient for confidence interval calculation
    survival_predictions = {}  # Will store {patient_idx: [list of survival functions]}
    
    for repeat_idx in range(n_repeats):
        if verbose and n_repeats > 1:
            print(f"\n  Repeat {repeat_idx + 1}/{n_repeats}")
        
        cv_random_state = random_state + repeat_idx
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cv_random_state)
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y['event'])):
            if verbose:
                if n_repeats > 1:
                    print(f"    Fold {fold_idx + 1}/{n_splits}...", end=' ')
                else:
                    print(f"  Processing fold {fold_idx + 1}/{n_splits}...")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Handle subsampling
            if n_controls_per_case is not None and n_controls_per_case != 'Full':
                from functions.repeated_cv import sample_case_control
                train_dataset = dataset.iloc[train_idx].copy()
                # sample_case_control returns data_x, data_y, sub_dataset
                X_train, y_train, _ = sample_case_control(
                    train_dataset, 
                    event_col=event_col, 
                    time_col=time_col,
                    n_controls_per_case=n_controls_per_case,
                    feature_columns=feature_columns,
                    random_state=cv_random_state + fold_idx
                )
            
            # Scale data if needed
            if scale_data:
                if continuous_features is not None:
                    scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
                    X_train_scaled = X_train.copy()
                    X_val_scaled = X_val.copy()
                    X_train_scaled[continuous_features] = scaler.fit_transform(X_train[continuous_features])
                    X_val_scaled[continuous_features] = scaler.transform(X_val[continuous_features])
                    X_train, X_val = X_train_scaled, X_val_scaled
                else:
                    scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
                    X_train = pd.DataFrame(scaler.fit_transform(X_train), 
                                          columns=X_train.columns, index=X_train.index)
                    X_val = pd.DataFrame(scaler.transform(X_val), 
                                        columns=X_val.columns, index=X_val.index)
            
            # optionally upweight events relative to censored cases in the training fold
            sample_weight_train = None
            if compute_weights is not False:
                num_events = y_train['event'].sum()
                num_censored = len(y_train) - num_events

                if num_events > 0:
                    # True -> derive the ratio automatically, else use the number given
                    if compute_weights is True:
                        weight_event = num_censored / num_events
                    else:
                        weight_event = float(compute_weights)

                    weight_censored = 1.0

                    sample_weight_train = np.full(len(y_train), weight_censored)
                    event_indicator = y_train['event'].astype(bool)
                    sample_weight_train[event_indicator] = weight_event

                    # rescale so weights sum to n (keeps the effective sample size unchanged)
                    sample_weight_train = sample_weight_train * (len(sample_weight_train) / sample_weight_train.sum())
                    
                    if verbose and fold_idx == 0 and repeat_idx == 0:
                        weight_type = "auto-computed" if compute_weights is True else "user-specified"
                        print(f"    Computed weights ({weight_type}): {num_events} events (weight={weight_event:.2f}), "
                              f"{num_censored} censored (weight={weight_censored:.2f})")
            
            # Train model
            import inspect
            model_signature = inspect.signature(model_class.__init__)
            model_init_params = model_params.copy()
            
            # Add random_state if model accepts it
            if 'random_state' in model_signature.parameters:
                model_init_params['random_state'] = cv_random_state + fold_idx
            
            # Add n_jobs if model accepts it
            if 'n_jobs' in model_signature.parameters:
                model_init_params['n_jobs'] = -1
            
            model = model_class(**model_init_params)
            
            # Fit model with or without sample weights
            if sample_weight_train is not None:
                model.fit(X_train, y_train, sample_weight=sample_weight_train)
            else:
                model.fit(X_train, y_train)
            
            # Get survival functions for validation patients
            surv_funcs = model.predict_survival_function(X_val)
            
            # Store survival functions for patients in this validation fold
            for i, patient_idx in enumerate(val_idx):
                if patient_idx not in survival_predictions:
                    survival_predictions[patient_idx] = []
                survival_predictions[patient_idx].append(surv_funcs[i])
            
            if verbose and n_repeats > 1:
                print("Done")
    
    if verbose:
        print(f"\n  Collected survival curves for {len(survival_predictions)} patients")
        if n_repeats > 1:
            pred_counts = [len(preds) for preds in survival_predictions.values()]
            print(f"  Predictions per patient: min={min(pred_counts)}, max={max(pred_counts)}, mean={np.mean(pred_counts):.1f}")
    
    # Now plot the selected patients
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get colormap
    cmap = plt.get_cmap(colormap)
    colors = [cmap(i % cmap.N) for i in range(len(patient_indices))]
    
    # Store patient information and survival curves
    patient_info_list = []
    survival_curves_data = {}
    
    # Plot each patient's survival curve
    for i, (idx, color) in enumerate(zip(patient_indices, colors)):
        if idx not in survival_predictions:
            if verbose:
                print(f"Warning: Patient {idx} not in CV results, skipping")
            continue
        
        patient_event = y['event'][idx]
        patient_time = y['time'][idx]
        
        # Evaluate survival functions at time points for all predictions of this patient
        try:
            all_surv_probs = []
            for surv_func in survival_predictions[idx]:
                if callable(surv_func):
                    surv_probs = np.array([surv_func(t) for t in time_points])
                else:
                    surv_probs = np.array([surv_func(t) for t in time_points])
                all_surv_probs.append(surv_probs)
            
            all_surv_probs = np.array(all_surv_probs)
            
            # Calculate mean and confidence intervals
            mean_survival = np.mean(all_surv_probs, axis=0)
            
            if n_repeats > 1 and show_confidence_interval:
                # Calculate 95% confidence intervals (2.5th and 97.5th percentiles)
                lower_ci = np.percentile(all_surv_probs, 2.5, axis=0)
                upper_ci = np.percentile(all_surv_probs, 97.5, axis=0)
            
            # Store survival curve data
            survival_curves_data[idx] = {
                'time_points': time_points,
                'mean_survival': mean_survival,
                'all_predictions': all_surv_probs
            }
            if n_repeats > 1 and show_confidence_interval:
                survival_curves_data[idx]['lower_ci'] = lower_ci
                survival_curves_data[idx]['upper_ci'] = upper_ci
                
        except Exception as e:
            if verbose:
                print(f"Warning: Could not evaluate survival function for patient {idx}: {e}")
            continue
        
        # Create label
        event_status = 'Event' if patient_event == 1 else 'Censored'
        if n_repeats > 1:
            label = f'Patient {idx} ({event_status}, t={patient_time:.0f}, n={len(all_surv_probs)})'
        else:
            label = f'Patient {idx} ({event_status}, t={patient_time:.0f})'
        
        # Plot confidence interval first (if available)
        if n_repeats > 1 and show_confidence_interval:
            ax.fill_between(time_points, lower_ci, upper_ci, 
                           color=color, alpha=ci_alpha, linewidth=0)
        
        # Plot mean survival curve
        ax.plot(time_points, mean_survival, color=color, linewidth=2, 
               alpha=alpha, label=label)
        
        # Add marker at actual event/censoring time
        # Use the mean survival function for marker placement
        if patient_event == 1 and show_event_markers:
            # Find predicted survival at event time
            try:
                time_idx = np.argmin(np.abs(time_points - patient_time))
                surv_at_event = mean_survival[time_idx]
                ax.scatter(patient_time, surv_at_event, color=color, 
                          marker='X', s=event_marker_size, zorder=5,
                          edgecolors='black', linewidths=1.5)
            except:
                pass
        
        if patient_event == 0 and show_censoring_markers:
            # Censored patient
            try:
                time_idx = np.argmin(np.abs(time_points - patient_time))
                surv_at_censor = mean_survival[time_idx]
                ax.scatter(patient_time, surv_at_censor, color=color, 
                          marker='o', s=censoring_marker_size, zorder=5,
                          edgecolors='black', linewidths=1.5, facecolors='none')
            except:
                pass
        
        # Store patient info
        patient_info_list.append({
            'patient_idx': idx,
            'event': patient_event,
            'time': patient_time,
            'event_status': event_status,
            'n_predictions': len(all_surv_probs)
        })
    
    # Formatting
    ax.set_xlabel('Time (days)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Predicted Survival Probability', fontsize=14, fontweight='bold')
    
    # Set title based on whether CI is shown
    if n_repeats > 1 and show_confidence_interval:
        title = f'Individual Patient Survival Curves (CV Predictions with 95% CI, {n_repeats} repeats)'
    else:
        title = 'Individual Patient Survival Curves (CV Predictions)'
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    ax.set_ylim([0, 1])
    ax.set_xlim(time_range)
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at 0.5
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Legend
    if show_legend:
        if legend_outside:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                     fontsize=9, framealpha=0.9)
        else:
            ax.legend(loc='best', fontsize=9, framealpha=0.9)
    
    # Add custom markers to legend explanation
    from matplotlib.lines import Line2D
    if show_event_markers or show_censoring_markers:
        custom_lines = []
        custom_labels = []
        
        if show_event_markers:
            custom_lines.append(Line2D([0], [0], marker='X', color='gray', 
                                      linestyle='None', markersize=10, 
                                      markeredgecolor='black', markeredgewidth=1.5))
            custom_labels.append('Event occurred')
        
        if show_censoring_markers:
            custom_lines.append(Line2D([0], [0], marker='o', color='white', 
                                      linestyle='None', markersize=8, 
                                      markeredgecolor='black', markeredgewidth=1.5))
            custom_labels.append('Censored')
        
        # Add second legend for markers
        if legend_outside:
            ax.legend(custom_lines, custom_labels, bbox_to_anchor=(1.05, 0), 
                     loc='lower left', fontsize=9, framealpha=0.9, title='Markers')
        else:
            # Combine with main legend
            handles, labels = ax.get_legend_handles_labels()
            all_handles = handles + custom_lines
            all_labels = labels + custom_labels
            ax.legend(all_handles, all_labels, loc='best', fontsize=8, 
                     framealpha=0.9, ncol=1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Individual survival curves saved to {save_path}")
    
    # Create patient info DataFrame
    patient_info_df = pd.DataFrame(patient_info_list)
    
    return fig, ax, patient_info_df, survival_curves_data


def plot_individual_survival_curves_from_model(model, X, y, patient_indices=None, 
                                    n_patients=10, select_events_only=True,
                                    time_range=None, n_time_points=100,
                                    figsize=(15, 10), save_path=None,
                                    show_event_markers=True, show_censoring_markers=True,
                                    event_marker_size=100, censoring_marker_size=80,
                                    colormap='tab10', alpha=0.8,
                                    show_legend=True, legend_outside=True):
    """Same plot as plot_individual_survival_curves, but takes an already-fitted model
    directly instead of refitting via CV - so unlike the CV version, these predictions
    aren't guaranteed to be out-of-fold for the plotted patients (that's on the caller).
    Handy for a quick look at a final model's behavior on specific patients (events by
    default) without rerunning CV just to plot.
    """
    # Select patients to plot
    if patient_indices is None:
        if select_events_only:
            # Get indices of patients with events
            event_indices = np.where(y['event'] == 1)[0]
            if len(event_indices) == 0:
                raise ValueError("No patients with events found in the data")
            # Select random subset if needed
            if len(event_indices) > n_patients:
                np.random.seed(42)
                patient_indices = np.random.choice(event_indices, size=n_patients, replace=False)
            else:
                patient_indices = event_indices[:n_patients]
        else:
            # Select random patients
            if len(y) > n_patients:
                np.random.seed(42)
                patient_indices = np.random.choice(len(y), size=n_patients, replace=False)
            else:
                patient_indices = np.arange(len(y))
    
    patient_indices = np.array(patient_indices)
    
    # Validate model
    if not hasattr(model, 'predict_survival_function'):
        raise AttributeError(
            f"Model of type {type(model).__name__} does not have 'predict_survival_function' method. "
            "Make sure you pass a fitted survival model instance, not the class itself."
        )
    
    # Check if model is fitted
    if not hasattr(model, 'n_features_in_') and not hasattr(model, 'feature_names_in_'):
        raise ValueError(
            "Model appears to be unfitted. Please fit the model before calling this function. "
            "Example: model.fit(X_train, y_train)"
        )
    
    # Determine time range
    if time_range is None:
        max_time = y['time'].max()
        time_range = (0, max_time)
    
    time_points = np.linspace(time_range[0], time_range[1], n_time_points)
    
    # Get predictions for selected patients
    X_selected = X.iloc[patient_indices] if hasattr(X, 'iloc') else X[patient_indices]
    
    try:
        surv_funcs = model.predict_survival_function(X_selected)
    except Exception as e:
        raise RuntimeError(
            f"Error calling predict_survival_function: {str(e)}\n"
            f"Model type: {type(model).__name__}\n"
            f"X_selected shape: {X_selected.shape if hasattr(X_selected, 'shape') else 'unknown'}\n"
            "Make sure you pass a fitted model instance."
        ) from e
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get colormap
    cmap = plt.get_cmap(colormap)
    colors = [cmap(i % cmap.N) for i in range(len(patient_indices))]
    
    # Store patient information
    patient_info_list = []
    
    # Plot each patient's survival curve
    for i, (idx, color) in enumerate(zip(patient_indices, colors)):
        patient_event = y['event'][idx]
        patient_time = y['time'][idx]
        
        # Evaluate survival function at time points
        try:
            if callable(surv_funcs[i]):
                survival_probs = [surv_funcs[i](t) for t in time_points]
            else:
                survival_probs = [surv_funcs[i](t) for t in time_points]
        except Exception as e:
            print(f"Warning: Could not evaluate survival function for patient {idx}: {e}")
            continue
        
        # Create label
        event_status = 'Event' if patient_event == 1 else 'Censored'
        label = f'Patient {idx} ({event_status}, t={patient_time:.0f})'
        
        # Plot survival curve
        ax.plot(time_points, survival_probs, color=color, linewidth=2, 
               alpha=alpha, label=label)
        
        # Add marker at actual event/censoring time
        if patient_event == 1 and show_event_markers:
            # Find predicted survival at event time
            try:
                if callable(surv_funcs[i]):
                    surv_at_event = surv_funcs[i](patient_time)
                else:
                    surv_at_event = surv_funcs[i](patient_time)
                ax.scatter(patient_time, surv_at_event, color=color, 
                          marker='X', s=event_marker_size, zorder=5,
                          edgecolors='black', linewidths=1.5)
            except:
                pass
        
        if patient_event == 0 and show_censoring_markers:
            # Censored patient
            try:
                if callable(surv_funcs[i]):
                    surv_at_censor = surv_funcs[i](patient_time)
                else:
                    surv_at_censor = surv_funcs[i](patient_time)
                ax.scatter(patient_time, surv_at_censor, color=color, 
                          marker='o', s=censoring_marker_size, zorder=5,
                          edgecolors='black', linewidths=1.5, facecolors='none')
            except:
                pass
        
        # Store patient info
        patient_info_list.append({
            'patient_idx': idx,
            'event': patient_event,
            'time': patient_time,
            'event_status': event_status
        })
    
    # Formatting
    ax.set_xlabel('Time (days)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Predicted Survival Probability', fontsize=14, fontweight='bold')
    ax.set_title('Individual Patient Survival Curves', fontsize=16, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.set_xlim(time_range)
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at 0.5
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Legend
    if show_legend:
        if legend_outside:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                     fontsize=9, framealpha=0.9)
        else:
            ax.legend(loc='best', fontsize=9, framealpha=0.9)
    
    # Add custom markers to legend explanation
    from matplotlib.lines import Line2D
    if show_event_markers or show_censoring_markers:
        custom_lines = []
        custom_labels = []
        
        if show_event_markers:
            custom_lines.append(Line2D([0], [0], marker='X', color='gray', 
                                      linestyle='None', markersize=10, 
                                      markeredgecolor='black', markeredgewidth=1.5))
            custom_labels.append('Event occurred')
        
        if show_censoring_markers:
            custom_lines.append(Line2D([0], [0], marker='o', color='white', 
                                      linestyle='None', markersize=8, 
                                      markeredgecolor='black', markeredgewidth=1.5))
            custom_labels.append('Censored')
        
        # Add second legend for markers
        if legend_outside:
            ax.legend(custom_lines, custom_labels, bbox_to_anchor=(1.05, 0), 
                     loc='lower left', fontsize=9, framealpha=0.9, title='Markers')
        else:
            # Combine with main legend
            handles, labels = ax.get_legend_handles_labels()
            all_handles = handles + custom_lines
            all_labels = labels + custom_labels
            ax.legend(all_handles, all_labels, loc='best', fontsize=8, 
                     framealpha=0.9, ncol=1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Individual survival curves saved to {save_path}")
    
    # Create patient info DataFrame
    patient_info_df = pd.DataFrame(patient_info_list)
    
    return fig, ax, patient_info_df


def compare_survival_curves_by_outcome(model_class, X, y, model_params,
                                       n_patients_per_group=5,
                                       scale_data=False, scaler_type='standard',
                                       continuous_features=None,
                                       n_splits=4, random_state=42,
                                       time_range=None, n_time_points=100,
                                       figsize=(16, 6), save_path=None,
                                       alpha=0.6, show_markers=True,
                                       verbose=True):
    """Side-by-side panels of predicted curves for a sample of event patients vs. a
    sample of censored patients, out-of-fold via single-repeat CV - a quick eyeball
    check of discrimination (do the event curves actually drop faster?). Returns fig,
    axes, and a stats dict with per-group summary numbers (mean end-survival, median
    survival time, mean observed event/censoring time).
    """
    if verbose:
        print(f"Collecting survival curves using {n_splits}-fold CV...")
    
    # Separate patients by outcome
    event_indices = np.where(y['event'] == 1)[0]
    censored_indices = np.where(y['event'] == 0)[0]
    
    # Sample patients
    np.random.seed(42)
    if len(event_indices) > n_patients_per_group:
        event_sample = np.random.choice(event_indices, size=n_patients_per_group, replace=False)
    else:
        event_sample = event_indices
    
    if len(censored_indices) > n_patients_per_group:
        censored_sample = np.random.choice(censored_indices, size=n_patients_per_group, replace=False)
    else:
        censored_sample = censored_indices
    
    # Determine time range
    if time_range is None:
        max_time = y['time'].max()
        time_range = (0, max_time)
    
    time_points = np.linspace(time_range[0], time_range[1], n_time_points)
    
    # Perform cross-validation to get survival functions for all patients
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    survival_functions = {}  # Will store {patient_idx: survival_function}
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y['event'])):
        if verbose:
            print(f"  Processing fold {fold_idx + 1}/{n_splits}...")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Scale data if needed
        if scale_data:
            if continuous_features is not None:
                scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
                X_train_scaled = X_train.copy()
                X_val_scaled = X_val.copy()
                X_train_scaled[continuous_features] = scaler.fit_transform(X_train[continuous_features])
                X_val_scaled[continuous_features] = scaler.transform(X_val[continuous_features])
                X_train, X_val = X_train_scaled, X_val_scaled
            else:
                scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
                X_train = pd.DataFrame(scaler.fit_transform(X_train), 
                                      columns=X_train.columns, index=X_train.index)
                X_val = pd.DataFrame(scaler.transform(X_val), 
                                    columns=X_val.columns, index=X_val.index)
        
        # Train model
        import inspect
        model_signature = inspect.signature(model_class.__init__)
        model_init_params = model_params.copy()
        
        # Add random_state if model accepts it
        if 'random_state' in model_signature.parameters:
            model_init_params['random_state'] = random_state
        
        # Add n_jobs if model accepts it
        if 'n_jobs' in model_signature.parameters:
            model_init_params['n_jobs'] = -1
        
        model = model_class(**model_init_params)
        model.fit(X_train, y_train)
        
        # Get survival functions for validation patients
        surv_funcs = model.predict_survival_function(X_val)
        
        # Store survival functions for patients in this validation fold
        for i, patient_idx in enumerate(val_idx):
            survival_functions[patient_idx] = surv_funcs[i]
    
    if verbose:
        print(f"Collected survival curves for {len(survival_functions)} patients")
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Process each group
    groups = [
        ('Patients with Events', event_sample, 'red', axes[0]),
        ('Censored Patients', censored_sample, 'blue', axes[1])
    ]
    
    stats_dict = {}
    
    for group_name, indices, base_color, ax in groups:
        cmap = plt.get_cmap('Reds' if base_color == 'red' else 'Blues')
        colors = [cmap(0.4 + 0.6 * i / len(indices)) for i in range(len(indices))]
        
        mean_survival = []
        
        # Plot each patient
        for i, (idx, color) in enumerate(zip(indices, colors)):
            if idx not in survival_functions:
                if verbose:
                    print(f"Warning: Patient {idx} not in CV results, skipping")
                continue
            
            patient_time = y['time'][idx]
            patient_event = y['event'][idx]
            surv_func = survival_functions[idx]
            
            try:
                if callable(surv_func):
                    survival_probs = [surv_func(t) for t in time_points]
                else:
                    survival_probs = [surv_func(t) for t in time_points]
                
                mean_survival.append(survival_probs)
                
                label = f'Patient {idx} (t={patient_time:.0f})'
                ax.plot(time_points, survival_probs, color=color, 
                       linewidth=1.5, alpha=alpha, label=label)
                
                # Add marker
                if show_markers:
                    if callable(surv_func):
                        surv_at_time = surv_func(patient_time)
                    else:
                        surv_at_time = surv_func(patient_time)
                    
                    marker = 'X' if patient_event == 1 else 'o'
                    facecolor = color if patient_event == 1 else 'none'
                    ax.scatter(patient_time, surv_at_time, color=color,
                             marker=marker, s=80, zorder=5,
                             edgecolors='black', linewidths=1.5,
                             facecolors=facecolor)
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not plot patient {idx}: {e}")
                continue
        
        # Plot mean curve
        if len(mean_survival) > 0:
            mean_curve = np.mean(mean_survival, axis=0)
            ax.plot(time_points, mean_curve, color='black', 
                   linewidth=3, linestyle='--', label='Mean', alpha=0.9)
        
        # Formatting
        ax.set_xlabel('Time (days)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Survival Probability', fontsize=12, fontweight='bold')
        ax.set_title(f'{group_name}\n(n={len(indices)})', fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.set_xlim(time_range)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.legend(fontsize=8, loc='best', framealpha=0.9)
        
        # Calculate stats
        if len(mean_survival) > 0:
            mean_curve = np.mean(mean_survival, axis=0)
            median_time_idx = np.argmin(np.abs(mean_curve - 0.5))
            median_time = time_points[median_time_idx]
            
            stats_dict[group_name] = {
                'n_patients': len(indices),
                'mean_survival_at_end': mean_curve[-1],
                'median_survival_time': median_time,
                'mean_event_time': y['time'][indices].mean()
            }
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    return fig, axes, stats_dict

def corrected_resampled_t_confidence_interval(
    mean_diff,
    std_diff,
    n_train,
    n_test,
    n_evaluations,
    alpha=0.05,
):
    """Corrected resampled t-test CI (Nadeau & Bengio, 2003) for a mean performance
    difference (model A - model B) across CV folds/repeats. The correction term
    (1/n_evaluations + n_test/n_train) accounts for the fact that CV folds aren't
    independent, which plain paired-t CIs ignore and which makes them too narrow.

    mean_diff/std_diff are the mean and sample std of the per-evaluation differences;
    n_evaluations is folds*repeats; n_train/n_test are the split sizes per fold.
    """
    if n_evaluations <= 1:
        raise ValueError("n_evaluations must be > 1.")
    if n_train <= 0 or n_test <= 0:
        raise ValueError("n_train and n_test must be positive.")
    if std_diff < 0:
        raise ValueError("std_diff must be non-negative.")

    correction = (1.0 / n_evaluations) + (n_test / n_train)
    se_corrected = sqrt(correction) * std_diff
    dof = n_evaluations - 1
    t_crit = student_t.ppf(1 - alpha / 2, dof)

    ci_low = mean_diff - t_crit * se_corrected
    ci_high = mean_diff + t_crit * se_corrected

    return {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "standard_error_corrected": se_corrected,
        "degrees_freedom": dof,
        "t_critical": t_crit,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "alpha": alpha,
        "confidence_level": 1 - alpha,
    }