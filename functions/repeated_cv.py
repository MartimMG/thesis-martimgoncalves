"""
Repeated (stratified) CV for the survival models used in the thesis: plain Cox and
Cox-with-subsampling, plus RSF/GBM/SVM ensembles (also with optional subsampling).

Everything here logs per-fold, per-hyperparameter-combo, per-repetition results rather
than just a summary -- makes it possible to go back and check e.g. whether a bad mean
was driven by one unlucky fold.
"""

import matplotlib.pyplot as plt
from sklearn.base import clone
import numpy as np
import pandas as pd
import time
import warnings
import concurrent.futures
from joblib.externals.loky import get_reusable_executor
from itertools import product
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import precision_recall_curve, auc as sklearn_auc, log_loss
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, integrated_brier_score, brier_score, cumulative_dynamic_auc
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv

import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import clone

def plot_coefficients(coefs):
    """Plot coefficient paths for different alpha values"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    alphas = np.array(coefs.columns)  
    for feature in coefs.index:
        row_values = np.array(coefs.loc[feature])  
        ax.semilogx(alphas, row_values, ".-", label=feature)
    
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Coefficient Value')
    ax.set_title('Coefficient Values vs. Alpha')
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.show()


from sksurv.nonparametric import kaplan_meier_estimator
import numpy as np

def get_ipcw_weights(y_train, y_test, eval_time, max_weight=20.0):
    """IPCW weights for ROC-AUC-style evaluation at a fixed eval_time. Capped at max_weight
    since G(t) can get tiny near the tail and blow the weights up otherwise."""
    # Estimate censoring distribution G(t)
    cens_time, cens_surv = kaplan_meier_estimator(
        ~y_train['event'], y_train['time']
    )
    
    weights = np.zeros(len(y_test))
    valid_mask = np.zeros(len(y_test), dtype=bool)
    y_binary = np.zeros(len(y_test), dtype=int)
    
    for i, (event, time) in enumerate(zip(y_test['event'], y_test['time'])):
        if time <= eval_time and event:
            # Case 1: Event before or at eval_time
            valid_mask[i] = True
            y_binary[i] = 1
            idx = np.searchsorted(cens_time, time) - 1
            g_t = cens_surv[idx] if idx >= 0 else 1.0
            # Apply safety cap to prevent exploding weights
            weights[i] = min(1.0 / g_t, max_weight) if g_t > 0 else 0.0
            
        elif time > eval_time:
            # Case 2: Survived past eval_time
            valid_mask[i] = True
            y_binary[i] = 0
            idx = np.searchsorted(cens_time, eval_time) - 1
            g_t = cens_surv[idx] if idx >= 0 else 1.0
            # Apply safety cap
            weights[i] = min(1.0 / g_t, max_weight) if g_t > 0 else 0.0
            
    return valid_mask, y_binary, weights


def get_ipcw_weights_for_prauc(y_train, y_test, eval_time, max_weight=20.0):
    """Same idea as get_ipcw_weights but for PR-AUC, which needs different handling.

    The main difference: subjects censored before eval_time are dropped entirely here
    rather than downweighted, because we genuinely don't know if they'd have had the
    event by eval_time and including them (even at low weight) biases the PR curve.
    Events keep weight 1/G(Ti) at their own event time; survivors past eval_time get
    1/G(eval_time).
    """
    # Estimate censoring distribution G(t) = P(C > t)
    cens_time, cens_surv = kaplan_meier_estimator(
        ~y_train['event'],  # Censoring indicators (True if censored)
        y_train['time']
    )
    
    weights = np.zeros(len(y_test))
    valid_mask = np.zeros(len(y_test), dtype=bool)
    y_binary = np.zeros(len(y_test), dtype=int)
    
    for i, (event, time) in enumerate(zip(y_test['event'], y_test['time'])):
        if event and time <= eval_time:
            # true positive, weight by 1/G at their own event time
            valid_mask[i] = True
            y_binary[i] = 1

            idx = np.searchsorted(cens_time, time) - 1
            g_ti = cens_surv[idx] if idx >= 0 else 1.0
            weights[i] = min(1.0 / g_ti, max_weight) if g_ti > 0 else 0.0

        elif time > eval_time:
            # true negative (censored or not, doesn't matter -- still alive at eval_time)
            valid_mask[i] = True
            y_binary[i] = 0

            idx = np.searchsorted(cens_time, eval_time) - 1
            g_eval = cens_surv[idx] if idx >= 0 else 1.0
            weights[i] = min(1.0 / g_eval, max_weight) if g_eval > 0 else 0.0

        # else: censored before eval_time -> dropped, valid_mask stays False (unknown outcome)

    return valid_mask, y_binary, weights

def _coxnet_fit_worker(X_train, y_train, alpha, l1_ratio, sample_weight):
    """Fits one CoxnetSurvivalAnalysis. Runs inside a worker process (see _fit_with_timeout
    below) so it can't touch any caller state beyond its plain picklable args."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model = CoxnetSurvivalAnalysis(alphas=[alpha], l1_ratio=l1_ratio, fit_baseline_model=True, max_iter=200000)
        if sample_weight is not None:
            model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train)
    return model


def _fit_with_timeout(X_train, y_train, alpha, l1_ratio, sample_weight, timeout_seconds):
    """
    Fit a Coxnet model with a wall-clock timeout, raising TimeoutError if it's
    exceeded.

    Runs the fit in a separate worker *process* (via joblib's reusable loky
    executor), not a thread. A thread-based timeout can only stop *waiting* on
    a hung fit — the underlying C/Cython coordinate-descent loop keeps running
    in the background forever, since it can't be killed from Python and holds
    the GIL while it runs. With enough slow fits (e.g. from extreme sample
    weights causing poor convergence), those threads pile up, starve the
    GIL/CPU for everything else, and the whole grid search grinds to a halt —
    a process can be terminated outright, so a fit that times out is genuinely
    stopped and its CPU is freed immediately.
    """
    executor = get_reusable_executor(max_workers=1)
    future = executor.submit(_coxnet_fit_worker, X_train, y_train, alpha, l1_ratio, sample_weight)
    try:
        return future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError:
        executor.shutdown(kill_workers=True)
        raise TimeoutError(f"Model fit exceeded {timeout_seconds}s timeout")


def subsampling_model_evaluation(X_train_sub, y_train_sub, X_val, y_val, alpha, l1_ratio, metrics,
                                 scaler=None, y_train_full=None, sample_weight=None, fit_timeout=None,
                                 return_model=False):
    """
    Trains a Cox model on the (subsampled) fold and scores it on every metric in `metrics`.

    A couple of args are worth flagging. `y_train_full`, if given, is the un-subsampled
    fold and is what the KM null model / IPCW weights for R2-IBS and R2-Brier are built
    from -- otherwise case-control subsampling would distort the censoring distribution
    those rely on. `fit_timeout` treats a fit that runs past it as failed (all metrics NaN)
    rather than letting the process hang. `return_model=True` also hands back the fitted
    estimator, or None on any early-exit path.
    """
    def _nan_result():
        result = {metric: np.nan for metric in metrics}
        return (result, None) if return_model else result

    # Use full training data for KM/IPCW if provided, otherwise fall back to subsampled
    y_for_km = y_train_full if y_train_full is not None else y_train_sub
    # Define common time points once
    TIME_POINTS = {
        '6m': 0.5 * 365,
        '1y': 1 * 365,
        '1.5y': 1.5 * 365,
        '2y': 2 * 365,
        '2.5y': 2.5 * 365,
        '3y': 3 * 365,
        '3.5y': 3.5 * 365,
        '4y': 4 * 365,
        '4.5y': 4.5 * 365,
        '5y': 5 * 365,
        '10y': 10 * 365,
        '14y': 14 * 365,
        '20y': 20 * 365
    }

    # First, convert to numpy arrays if needed
    if isinstance(X_train_sub, pd.DataFrame):
        X_train_sub_values = X_train_sub.values
    else:
        X_train_sub_values = X_train_sub
        
    if isinstance(X_val, pd.DataFrame):
        X_val_values = X_val.values
    else:
        X_val_values = X_val

    # Cap extreme values to avoid overflow
    EXTREME_CAP = 1e6
    X_train_sub_values = np.clip(X_train_sub_values, -EXTREME_CAP, EXTREME_CAP)
    X_val_values = np.clip(X_val_values, -EXTREME_CAP, EXTREME_CAP)

    # Check for NaN or inf in ORIGINAL data before scaling
    if np.any(np.isnan(X_train_sub_values)) or np.any(np.isinf(X_train_sub_values)):
        print(f"Warning: Original training data contains NaN or inf values BEFORE scaling. Skipping this fold.")
        return _nan_result()
        
    if np.any(np.isnan(X_val_values)) or np.any(np.isinf(X_val_values)):
        print(f"Warning: Original validation data contains NaN or inf values BEFORE scaling. Skipping this fold.")
        return _nan_result()

    # Apply scaling based on the scaler parameter
    try:
        if scaler is not None:
            scaler_instance = clone(scaler)
            if isinstance(scaler_instance, StandardScaler):
                feature_std = np.std(X_train_sub_values, axis=0)
                constant_features_mask = feature_std == 0
                constant_feature_indices = np.where(constant_features_mask)[0]
                if len(constant_feature_indices) > 0:
                    non_constant_mask = ~constant_features_mask
                    X_train_sub_values = X_train_sub_values[:, non_constant_mask]
                    X_val_values = X_val_values[:, non_constant_mask]
            
            X_train_sub_scaled = scaler_instance.fit_transform(X_train_sub_values)
            X_val_scaled = scaler_instance.transform(X_val_values)
        else:
            X_train_sub_scaled = X_train_sub_values
            X_val_scaled = X_val_values
    except Exception as e:
        print(f"Warning: Scaling/transformation failed. Error: {e}")
        return _nan_result()

    # Check for NaN or inf in scaled data
    if np.any(np.isnan(X_train_sub_scaled)) or np.any(np.isinf(X_train_sub_scaled)):
        print(f"Warning: Scaled training data contains NaN or inf values AFTER scaling. Skipping this fold.")
        return _nan_result()
    if np.any(np.isnan(X_val_scaled)) or np.any(np.isinf(X_val_scaled)):
        print(f"Warning: Scaled validation data contains NaN or inf values AFTER scaling. Skipping this fold.")
        return _nan_result()

    # Train the model on scaled data
    try:
        if fit_timeout is not None:
            model = _fit_with_timeout(
                X_train_sub_scaled, y_train_sub, alpha, l1_ratio, sample_weight, fit_timeout
            )
        else:
            model = _coxnet_fit_worker(X_train_sub_scaled, y_train_sub, alpha, l1_ratio, sample_weight)
    except Exception as e:
        print(f"Warning: Model fitting failed with error: {e}")
        return _nan_result()

    # Predict risk scores ONCE
    predicted_risk_scores = model.predict(X_val_scaled)
    if np.any(np.isnan(predicted_risk_scores)):
        print(f"Warning: Model produced NaN predictions. Skipping this fold.")
        return _nan_result()

    # Predict survival functions ONCE
    surv_funcs = None
    if hasattr(model, 'predict_survival_function'):
        try:
            surv_funcs = model.predict_survival_function(X_val_scaled)
        except Exception:
            pass

    results = {}
    for metric in metrics:
        if metric == 'c-index':
            try:
                score = concordance_index_censored(y_val['event'], y_val['time'], predicted_risk_scores)[0]
                results[metric] = score
            except Exception as e:
                print(f"Warning: C-index calculation failed: {str(e)}")
                results[metric] = np.nan
                
        elif metric == 'c-index-ipcw':
            try:
                max_time = min(y_train_sub['time'].max(), y_val['time'].max()) * 0.9
                score = concordance_index_ipcw(y_train_sub, y_val, predicted_risk_scores, tau=max_time)[0]
                results[metric] = score
            except Exception as e:
                print(f"Warning: C-index IPCW calculation failed: {str(e)}")
                results[metric] = np.nan
                
        elif metric == 'int-brier-score':
            try:
                if surv_funcs is not None:
                    max_time = min(y_train_sub['time'].max(), y_val['time'].max()) * 0.9
                    times = np.percentile(y_train_sub['time'], np.linspace(5, 90, 10))
                    times = np.unique(times[(times > 0) & (times <= max_time)])
                    
                    if len(times) > 1:
                        predicted_survival_probabilities = np.asarray([[fn(t) for t in times] for fn in surv_funcs])
                        score = integrated_brier_score(y_train_sub, y_val, predicted_survival_probabilities, times)
                        results[metric] = score
                    else:
                        results[metric] = np.nan
                else:
                    results[metric] = np.nan
            except Exception as e:
                print(f"Warning: Brier score calculation failed: {str(e)}")
                results[metric] = np.nan
                
        elif metric.startswith('brier-'):
            try:
                time_key = metric.replace('brier-', '')
                if surv_funcs is not None and time_key in TIME_POINTS:
                    time_point = TIME_POINTS[time_key]
                    max_time = min(y_train_sub['time'].max(), y_val['time'].max())
                    
                    if time_point <= max_time:
                        predicted_survival_probabilities = np.asarray([fn(time_point) for fn in surv_funcs])
                        score = float(brier_score(y_train_sub, y_val, predicted_survival_probabilities, time_point)[1][0])
                        results[metric] = score
                    else:
                        results[metric] = np.nan
                else:
                    results[metric] = np.nan
            except Exception as e:
                print(f"Warning: {metric} calculation failed: {str(e)}")
                results[metric] = np.nan
                
        elif metric == 'deviance':
            try:
                risk_scores = predicted_risk_scores
                sort_idx = np.argsort(y_val['time'])
                sorted_time = y_val['time'][sort_idx]
                sorted_event = y_val['event'][sort_idx]
                sorted_risk = risk_scores[sort_idx]
                
                log_likelihood = 0.0
                for i in range(len(sorted_time)):
                    if sorted_event[i]:
                        risk_set_mask = sorted_time >= sorted_time[i]
                        risk_set_scores = sorted_risk[risk_set_mask]
                        log_likelihood += sorted_risk[i] - np.log(np.sum(np.exp(risk_set_scores)))
                
                deviance_score = -2 * log_likelihood
                results[metric] = deviance_score
            except Exception as e:
                print(f"Warning: Deviance calculation failed: {str(e)}")
                results[metric] = np.nan
        
        elif metric == 'r2-ibs':
            try:
                if surv_funcs is not None:
                    max_time = min(y_for_km['time'].max(), y_val['time'].max()) * 0.9
                    times = np.percentile(y_for_km['time'], np.linspace(5, 90, 10))
                    times = np.unique(times[(times > 0) & (times <= max_time)])
                    
                    if len(times) > 1:
                        model_surv_probs = np.asarray([[fn(t) for t in times] for fn in surv_funcs])
                        ibs_model = integrated_brier_score(y_for_km, y_val, model_surv_probs, times)
                        
                        km_time, km_surv = kaplan_meier_estimator(y_for_km['event'], y_for_km['time'])
                        km_surv_at_times = np.interp(times, km_time, km_surv, left=1.0, right=km_surv[-1])
                        km_surv_probs = np.tile(km_surv_at_times, (len(y_val), 1))
                        ibs_null = integrated_brier_score(y_for_km, y_val, km_surv_probs, times)
                        
                        if ibs_null > 0:
                            results[metric] = 1.0 - (ibs_model / ibs_null)
                        else:
                            results[metric] = np.nan
                    else:
                        results[metric] = np.nan
                else:
                    results[metric] = np.nan
            except Exception as e:
                print(f"Warning: R²_IBS calculation failed: {str(e)}")
                results[metric] = np.nan
        
        elif metric.startswith('r2-brier-'):
            # R² Brier score at specific time points
            # Measures improvement over Kaplan-Meier null model at a given time
            try:
                time_key = metric.replace('r2-brier-', '')
                if surv_funcs is not None and time_key in TIME_POINTS:
                    time_point = TIME_POINTS[time_key]
                    max_time = min(y_for_km['time'].max(), y_val['time'].max())
                    
                    if time_point <= max_time:
                        # Model Brier score at this time point
                        model_surv_probs = np.asarray([fn(time_point) for fn in surv_funcs])
                        bs_model = float(brier_score(y_for_km, y_val, model_surv_probs, time_point)[1][0])
                        
                        # Null model (Kaplan-Meier) Brier score at this time point
                        km_time, km_surv = kaplan_meier_estimator(y_for_km['event'], y_for_km['time'])
                        km_surv_at_time = np.interp(time_point, km_time, km_surv, left=1.0, right=km_surv[-1])
                        km_surv_probs = np.full(len(y_val), km_surv_at_time)
                        bs_null = float(brier_score(y_for_km, y_val, km_surv_probs, time_point)[1][0])
                        
                        # R² = 1 - (BS_model / BS_null)
                        results[metric] = 1.0 - (bs_model / bs_null) if bs_null > 0 else np.nan
                    else:
                        results[metric] = np.nan
                else:
                    results[metric] = np.nan
            except Exception as e:
                print(f"Warning: {metric} calculation failed: {str(e)}")
                results[metric] = np.nan
                
        elif metric == 'cumulative-dynamic-auc':
            try:
                max_time = min(y_train_sub['time'].max(), y_val['time'].max()) * 0.9
                times = np.percentile(y_train_sub['time'], np.linspace(10, 90, 10))
                times = np.unique(times[(times > 0) & (times <= max_time)])
                
                if len(times) >= 2:
                    auc, mean_auc = cumulative_dynamic_auc(y_train_sub, y_val, predicted_risk_scores, times)
                    results[metric] = mean_auc
                else:
                    results[metric] = np.nan
            except Exception as e:
                print(f"Warning: Cumulative Dynamic AUC calculation failed: {str(e)}")
                results[metric] = np.nan
        
        elif metric.startswith('auc-'):
            # AUC at specific time points
            try:
                time_key = metric.replace('auc-', '')
                if time_key in TIME_POINTS:
                    time_point = TIME_POINTS[time_key]
                    max_time = min(y_train_sub['time'].max(), y_val['time'].max())
                    
                    if time_point <= max_time:
                        auc_values, _ = cumulative_dynamic_auc(
                            y_train_sub, y_val, predicted_risk_scores, [time_point]
                        )
                        results[metric] = auc_values[0]
                    else:
                        results[metric] = np.nan
                else:
                    results[metric] = np.nan
            except Exception as e:
                print(f"Warning: {metric} calculation failed: {str(e)}")
                results[metric] = np.nan
        
        elif metric == 'cumulative-dynamic-prauc':
            try:
                max_time = min(y_train_sub['time'].max(), y_val['time'].max()) * 0.95
                times = np.percentile(y_train_sub['time'], np.linspace(10, 90, 10))
                times = np.unique(times[(times > 0) & (times <= max_time)])
                
                if len(times) >= 2:
                    prauc_scores = []
                    for t in times:
                        valid_mask, y_binary, weights = get_ipcw_weights_for_prauc(y_train_sub, y_val, t)
                        if valid_mask.sum() > 0 and y_binary[valid_mask].sum() > 0:
                            precision_vals, recall_vals, _ = precision_recall_curve(
                                y_binary[valid_mask], predicted_risk_scores[valid_mask],
                                sample_weight=weights[valid_mask]
                            )
                            prauc_scores.append(sklearn_auc(recall_vals, precision_vals))
                    
                    if len(prauc_scores) > 0:
                        results[metric] = np.mean(prauc_scores)
                    else:
                        results[metric] = np.nan
                else:
                    results[metric] = np.nan
            except Exception as e:
                print(f"Warning: Cumulative Dynamic PR-AUC calculation failed: {str(e)}")
                results[metric] = np.nan
        
        elif metric.startswith('prauc-'):
            try:
                time_key = metric.replace('prauc-', '')
                if time_key in TIME_POINTS:
                    time_point = TIME_POINTS[time_key]
                    max_time = min(y_train_sub['time'].max(), y_val['time'].max())
                    
                    if time_point <= max_time:
                        valid_mask, y_binary, weights = get_ipcw_weights_for_prauc(y_train_sub, y_val, time_point)
                        if valid_mask.sum() > 0 and y_binary[valid_mask].sum() > 0:
                            precision_vals, recall_vals, _ = precision_recall_curve(
                                y_binary[valid_mask], predicted_risk_scores[valid_mask],
                                sample_weight=weights[valid_mask]
                            )
                            results[metric] = sklearn_auc(recall_vals, precision_vals)
                        else:
                            results[metric] = np.nan
                    else:
                        results[metric] = np.nan
                else:
                    results[metric] = np.nan
            except Exception as e:
                print(f"Warning: {metric} calculation failed: {str(e)}")
                results[metric] = np.nan
        
        elif metric == 'partial-likelihood':
            # Cox partial log-likelihood (higher is better)
            try:
                risk_scores = predicted_risk_scores
                sort_idx = np.argsort(y_val['time'])
                sorted_time = y_val['time'][sort_idx]
                sorted_event = y_val['event'][sort_idx]
                sorted_risk = risk_scores[sort_idx]
                
                log_likelihood = 0.0
                for i in range(len(sorted_time)):
                    if sorted_event[i]:
                        risk_set_mask = sorted_time >= sorted_time[i]
                        risk_set_scores = sorted_risk[risk_set_mask]
                        log_likelihood += sorted_risk[i] - np.log(np.sum(np.exp(risk_set_scores)))
                
                results[metric] = log_likelihood
            except Exception as e:
                print(f"Warning: Partial likelihood calculation failed: {str(e)}")
                results[metric] = np.nan
        
        elif metric == 'log-loss':
            # Integrated log loss across multiple time points
            try:
                if surv_funcs is not None:
                    max_time = min(y_train_sub['time'].max(), y_val['time'].max()) * 0.9
                    times = np.percentile(y_train_sub['time'], np.linspace(10, 90, 10))
                    times = np.unique(times[(times > 0) & (times <= max_time)])
                    
                    if len(times) >= 2:
                        log_losses = []
                        for t in times:
                            y_binary = ((y_val['time'] <= t) & y_val['event']).astype(int)
                            pred_proba = 1.0 - np.array([fn(t) for fn in surv_funcs])
                            pred_proba = np.clip(pred_proba, 1e-15, 1 - 1e-15)
                            
                            if len(np.unique(y_binary)) >= 2:
                                log_losses.append(log_loss(y_binary, pred_proba))
                        
                        if len(log_losses) > 0:
                            results[metric] = np.mean(log_losses)
                        else:
                            results[metric] = np.nan
                    else:
                        results[metric] = np.nan
                else:
                    results[metric] = np.nan
            except Exception as e:
                print(f"Warning: Log loss calculation failed: {str(e)}")
                results[metric] = np.nan
        
        elif metric.startswith('log-loss-'):
            # Log loss at specific time points
            try:
                time_key = metric.replace('log-loss-', '')
                if surv_funcs is not None and time_key in TIME_POINTS:
                    time_point = TIME_POINTS[time_key]
                    max_time = min(y_train_sub['time'].max(), y_val['time'].max())
                    
                    if time_point <= max_time:
                        y_binary = ((y_val['time'] <= time_point) & y_val['event']).astype(int)
                        pred_proba = 1.0 - np.array([fn(time_point) for fn in surv_funcs])
                        pred_proba = np.clip(pred_proba, 1e-15, 1 - 1e-15)
                        
                        if len(np.unique(y_binary)) >= 2:
                            results[metric] = log_loss(y_binary, pred_proba)
                        else:
                            results[metric] = np.nan
                    else:
                        results[metric] = np.nan
                else:
                    results[metric] = np.nan
            except Exception as e:
                print(f"Warning: {metric} calculation failed: {str(e)}")
                results[metric] = np.nan
        
        else:
            print(f"Warning: Unknown metric '{metric}', returning NaN")
            results[metric] = np.nan

    return (results, model) if return_model else results


def subsampling_cross_validation(dataset, X, y, cv_folds, alpha, l1_ratio, metrics, 
                                n_controls_per_case, feature_columns,
                                event_col='Regional LN recurrence', time_col='Survival days',
                                scaler=None):
    """CV for a Cox model with case-control subsampling, at one fixed alpha. Returns a
    DataFrame indexed by fold (plus Mean/Std rows at the bottom), one column per metric."""
    n_folds = cv_folds.get_n_splits()

    results = {metric: [] for metric in metrics}

    for fold_idx, (train_idx, val_idx) in enumerate(cv_folds.split(X, y['event'])):
        train_fold = dataset.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y[val_idx]

        # subsample only the training side, validation stays untouched
        X_train_sub, y_train_sub, sub_dataset = sample_case_control(
            train_fold, event_col, time_col, n_controls_per_case, feature_columns
        )

        fold_results = subsampling_model_evaluation(
            X_train_sub, y_train_sub, X_val_fold, y_val_fold,
            alpha, l1_ratio, metrics, scaler=scaler,
            y_train_full=y[train_idx]
        )

        for metric in metrics:
            results[metric].append(fold_results.get(metric, np.nan))

    results_df = pd.DataFrame(results)
    results_df.index = [f'Fold_{i+1}' for i in range(n_folds)]

    mean_row = results_df.mean()
    std_row = results_df.std()

    mean_row.name = 'Mean'
    std_row.name = 'Std'

    results_df = pd.concat([results_df, mean_row.to_frame().T, std_row.to_frame().T])

    return results_df

def sample_case_control(train_fold: pd.DataFrame,
                        event_col: str = 'Regional LN recurrence',
                        time_col: str = 'Survival days', 
                        n_controls_per_case: int = 5,
                        feature_columns: list = None,
                        random_state: int = 12):
    """For each event (case), sample n_controls_per_case controls from the risk set at
    that event's time (i.e. anyone censored who was still at risk then), without reusing
    a control already assigned to a previous case. Returns (X, y, sub_dataset) for the
    resulting subsample."""
    train_df = train_fold.copy()

    events_df = train_df[train_df[event_col] == 1].copy()
    censored_df = train_df[train_df[event_col] == 0].copy()

    patients_ids = []
    added_indexes = set()

    for i, row in events_df.iterrows():
        t_i = row[time_col]
        event_idx = row.name

        # eligible controls = still at risk at t_i, and not already used
        risk_set = censored_df[censored_df[time_col] >= t_i].copy()
        risk_set = risk_set[~risk_set.index.isin(added_indexes)]

        if len(risk_set) >= n_controls_per_case:
            sampled_controls = risk_set.sample(n=n_controls_per_case, replace=False, random_state=random_state)
        else:
            sampled_controls = risk_set

        if event_idx not in added_indexes:
            patients_ids.append(event_idx)
            added_indexes.add(event_idx)

        for ctrl_idx in sampled_controls.index:
            if ctrl_idx not in added_indexes:
                patients_ids.append(ctrl_idx)
                added_indexes.add(ctrl_idx)

    sub_dataset = train_df.loc[patients_ids].copy()

    data_x = sub_dataset[feature_columns]

    data_y = Surv.from_arrays(
        event=sub_dataset[event_col],
        time=sub_dataset[time_col]
    )
    return data_x, data_y, sub_dataset


def compute_case_control_weights(sampled_df, event_col="Regional LN recurrence"):
    """Inverse-probability-of-selection-style weights for a case-control subsample: events
    get upweighted by (controls/events) so the weighted sample roughly reflects the
    original case:control ratio, then everything is rescaled to sum to n."""
    num_events = sampled_df[event_col].sum()
    num_censored = len(sampled_df) - num_events

    if num_events == 0:
        raise ValueError("No events in sampled dataset — cannot compute weights.")

    weight_event = num_censored / num_events
    weight_censored = 1.0

    sample_weights = np.full(len(sampled_df), weight_censored)
    event_indicator = sampled_df[event_col].values.astype(bool)

    sample_weights[event_indicator] = weight_event
    sample_weights = sample_weights * (len(sample_weights) / sample_weights.sum())

    return sample_weights


# Create summary DataFrames for each metric
def create_alpha_summary(results_dict, metric):
    """Create summary DataFrame for a specific metric across all alphas"""
    summary_data = []
    for alpha, df in results_dict.items():
        summary_data.append({
            'Alpha': alpha,
            'Mean': df.loc['Mean', metric],
            'Std': df.loc['Std', metric]
        })
    return pd.DataFrame(summary_data)


# Find the best alpha for each metric
def find_best_alpha(summary_df, metric_name, higher_is_better=True):
    """Find the best alpha value for a given metric"""
    if higher_is_better:
        best_row = summary_df.loc[summary_df['Mean'].idxmax()]
    else:
        best_row = summary_df.loc[summary_df['Mean'].idxmin()]
    
    print(f"Best alpha for {metric_name}: {best_row['Alpha']:.4f}")
    print(f"Score: {best_row['Mean']:.4f} ± {best_row['Std']:.4f}")
    return best_row['Alpha']


def plot_subsampling_results(results_df, metric, title_prefix=""):
    """
    Plot cross-validation results for a specific metric from the results DataFrame.
    """
    # Extract fold results (excluding Mean and Std rows)
    fold_data = results_df.loc[results_df.index.str.startswith('Fold_')]
    
    plt.figure(figsize=(10, 6))
    
    # Get mean and std from the DataFrame
    mean_value = results_df.loc['Mean', metric]
    std_value = results_df.loc['Std', metric]
    
    # Plot individual fold results
    fold_values = fold_data[metric]
    fold_numbers = range(1, len(fold_values) + 1)
    
    plt.scatter(fold_numbers, fold_values, alpha=0.6, s=50, label=f'{metric} (individual folds)')
    plt.axhline(y=mean_value, color='red', linestyle='--', label=f'Mean: {mean_value:.3f}')
    plt.fill_between([0.5, len(fold_values) + 0.5], 
                     mean_value - std_value, mean_value + std_value, 
                     alpha=0.2, color='red', label=f'±1 Std: {std_value:.3f}')
    
    plt.xlabel('Fold')
    plt.ylabel(f'{metric}')
    plt.title(f'{title_prefix}{metric} Across Folds (with Subsampling)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_cindex_subsampling(df):
    """Plot C-index results for subsampling approach"""
    plt.figure(figsize=(10, 6))

    best_idx = df['Mean'].idxmax()
    
    plt.errorbar(df['Alpha'], df['Mean'], yerr=df['Std'], fmt='o', capsize=5, 
                label='C-index (mean)', color='tab:cyan')
    
    plt.errorbar(df['Alpha'].iloc[best_idx], df['Mean'].iloc[best_idx], 
                yerr=df['Std'].iloc[best_idx], 
                fmt='o', capsize=5, label='Best C-index', color='tab:pink', markersize=10)

    plt.xscale('log')  
    plt.ylabel('C-index (Mean)')
    plt.xlabel('Alpha (log scale)')
    plt.title('C-index (Mean) for Different Alphas - Subsampling Approach')
    plt.xticks(rotation=45, ha='right')  
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(False)  

    plt.legend()  
    plt.tight_layout()  
        
    plt.show()


def plot_ipcw_subsampling(df):
    """Plot IPCW C-index results for subsampling approach"""
    plt.figure(figsize=(10, 6))

    best_idx = df['Mean'].idxmax()
    
    plt.errorbar(df['Alpha'], df['Mean'], yerr=df['Std'], fmt='o', capsize=5, 
                label='IPCW C-index (mean)', color='tab:cyan')
    
    plt.errorbar(df['Alpha'].iloc[best_idx], df['Mean'].iloc[best_idx], 
                yerr=df['Std'].iloc[best_idx], 
                fmt='o', capsize=5, label='Best IPCW C-index', color='tab:pink', markersize=10)

    plt.xscale('log')  
    plt.ylabel('IPCW C-index (Mean)')
    plt.xlabel('Alpha (log scale)')
    plt.title('IPCW C-index (Mean) for Different Alphas - Subsampling Approach')
    plt.xticks(rotation=45, ha='right')  
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(False)  

    plt.legend()  
    plt.tight_layout()  
        
    plt.show()


def plot_IBS_subsampling(df):
    """Plot Integrated Brier Score results for subsampling approach"""
    plt.figure(figsize=(10, 6))
    
    plt.errorbar(df['Alpha'], df['Mean'], yerr=df['Std'], fmt='o', capsize=5, 
                label='IBS (mean)')

    plt.xscale('log') 
    plt.ylabel('IBS (Mean)')
    plt.xlabel('Alpha (log scale)')
    plt.title('IBS (Mean) for Different Alphas - Subsampling Approach')
    plt.xticks(rotation=45, ha='right')  
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(False)  

    plt.tight_layout() 
        
    plt.show()


def subsampling_grid_search(dataset, X, y, cv_folds, 
                           alpha_range, l1_ratio_range, controls_per_case_range,
                           metrics, feature_columns,
                           event_col='Regional LN recurrence', time_col='Survival days',
                           scaler=None, verbose=True):
    """
    Grid search over (alpha, l1_ratio, n_controls_per_case) for the subsampled Cox model.
    Returns the raw nested results {n_controls: {l1_ratio: {alpha: results_df}}}, the
    best params per metric, and a flat summary_df for plotting.
    """
    from itertools import product
    import time

    results = {}
    all_results = []

    total_iterations = len(controls_per_case_range) * len(l1_ratio_range) * len(alpha_range)
    current_iteration = 0
    
    start_time = time.time()
    
    if verbose:
        print(f"{'='*80}")
        print(f"SUBSAMPLING GRID SEARCH")
        print(f"{'='*80}")
        print(f"Controls per case: {controls_per_case_range}")
        print(f"L1 ratios: {l1_ratio_range}")
        print(f"Alphas: {len(alpha_range)} values from {min(alpha_range):.2e} to {max(alpha_range):.2e}")
        print(f"Total combinations: {total_iterations}")
        print(f"{'='*80}\n")
    
    # Grid search over all combinations
    for n_controls in controls_per_case_range:
        results[n_controls] = {}
        
        for l1_ratio in l1_ratio_range:
            results[n_controls][l1_ratio] = {}
            
            for alpha in alpha_range:
                current_iteration += 1
                
                if verbose:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / current_iteration if current_iteration > 0 else 0
                    remaining = avg_time * (total_iterations - current_iteration)
                print(f"[{current_iteration}/{total_iterations}] "
                      f"controls={n_controls}, l1={l1_ratio:.3f}, alpha={alpha:.2e} "
                      f"(elapsed: {elapsed/60:.1f}min, est. remaining: {remaining/60:.1f}min)")
                
                try:
                    results_df = subsampling_cross_validation(
                        dataset, X, y, cv_folds, alpha, l1_ratio, metrics,
                        n_controls, feature_columns, event_col, time_col, scaler
                    )

                    results[n_controls][l1_ratio][alpha] = results_df

                    for metric in metrics:
                        all_results.append({
                            'n_controls': n_controls,
                            'l1_ratio': l1_ratio,
                            'alpha': alpha,
                            'metric': metric,
                            'mean': results_df.loc['Mean', metric],
                            'std': results_df.loc['Std', metric]
                        })

                except Exception as e:
                    if verbose:
                        print(f"  Configuration failed: {str(e)[:80]}")

                    # fill in NaNs so this combo still shows up in the summary/plots
                    fold_names = [f'Fold_{i+1}' for i in range(cv_folds.get_n_splits())]
                    nan_data = {metric: [np.nan] * len(fold_names) for metric in metrics}
                    results_df = pd.DataFrame(nan_data, index=fold_names)

                    mean_row = pd.Series({metric: np.nan for metric in metrics}, name='Mean')
                    std_row = pd.Series({metric: np.nan for metric in metrics}, name='Std')
                    results_df = pd.concat([results_df, mean_row.to_frame().T, std_row.to_frame().T])

                    results[n_controls][l1_ratio][alpha] = results_df

                    for metric in metrics:
                        all_results.append({
                            'n_controls': n_controls,
                            'l1_ratio': l1_ratio,
                            'alpha': alpha,
                            'metric': metric,
                            'mean': np.nan,
                            'std': np.nan
                        })
    summary_df = pd.DataFrame(all_results)

    best_params = {}

    for metric in metrics:
        metric_data = summary_df[summary_df['metric'] == metric].copy()

        # brier/deviance: lower is better; everything else: higher is better
        if 'brier' in metric.lower() or 'deviance' in metric.lower():
            best_idx = metric_data['mean'].idxmin()
        else:
            best_idx = metric_data['mean'].idxmax()
        
        best_row = metric_data.loc[best_idx]
        best_params[metric] = {
            'n_controls': int(best_row['n_controls']),
            'l1_ratio': best_row['l1_ratio'],
            'alpha': best_row['alpha'],
            'mean': best_row['mean'],
            'std': best_row['std']
        }
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"GRID SEARCH COMPLETE")
        print(f"{'='*80}")
        print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
        print(f"\nBest parameters for each metric:")
        print(f"{'-'*80}")
        for metric, params in best_params.items():
            print(f"\n{metric}:")
            print(f"  n_controls_per_case: {params['n_controls']}")
            print(f"  l1_ratio: {params['l1_ratio']:.4f}")
            print(f"  alpha: {params['alpha']:.4e}")
            print(f"  Score: {params['mean']:.4f} ± {params['std']:.4f}")
        print(f"\n{'='*80}\n")
    
    return results, best_params, summary_df


def plot_grid_search_results(summary_df, metric='c-index'):
    """Heatmap (l1_ratio x alpha) plus a line plot, one panel per n_controls value, for a
    given metric from subsampling_grid_search's summary_df."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    metric_data = summary_df[summary_df['metric'] == metric].copy()

    n_controls_values = sorted(metric_data['n_controls'].unique())
    n_plots = len(n_controls_values)
    
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    for idx, n_controls in enumerate(n_controls_values):
        ax = axes[idx]
        data = metric_data[metric_data['n_controls'] == n_controls]
        
        # Pivot for heatmap: l1_ratio vs alpha
        pivot_data = data.pivot_table(
            values='mean', 
            index='l1_ratio', 
            columns='alpha', 
            aggfunc='first'
        )
        
        im = ax.imshow(pivot_data, aspect='auto', cmap='viridis')
        ax.set_xlabel('Alpha')
        ax.set_ylabel('L1 Ratio')
        ax.set_title(f'{metric} (n_controls={n_controls})')
        
        # Set tick labels
        ax.set_xticks(range(len(pivot_data.columns)))
        ax.set_xticklabels([f'{a:.2e}' for a in pivot_data.columns], rotation=45, ha='right')
        ax.set_yticks(range(len(pivot_data.index)))
        ax.set_yticklabels([f'{r:.2f}' for r in pivot_data.index])
        
        plt.colorbar(im, ax=ax, label='Mean Score')
    
    plt.tight_layout()
    plt.show()
    
    # Also create a line plot for each n_controls
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    for idx, n_controls in enumerate(n_controls_values):
        ax = axes[idx]
        data = metric_data[metric_data['n_controls'] == n_controls]
        
        for l1_ratio in sorted(data['l1_ratio'].unique()):
            l1_data = data[data['l1_ratio'] == l1_ratio].sort_values('alpha')
            ax.errorbar(l1_data['alpha'], l1_data['mean'], yerr=l1_data['std'],
                       marker='o', capsize=3, label=f'l1={l1_ratio:.2f}')
        
        ax.set_xscale('log')
        ax.set_xlabel('Alpha (log scale)')
        ax.set_ylabel(f'{metric}')
        ax.set_title(f'{metric} vs Alpha (n_controls={n_controls})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_repeated_stratified_cv(n_splits=5, n_repeats=10, base_random_state=42):
    """n_repeats StratifiedKFold objects, each with its own random_state (base + i).
    Returns a list of (repeat_idx, random_state, cv) tuples."""
    cv_list = []
    for repeat_idx in range(n_repeats):
        random_state = base_random_state + repeat_idx
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_list.append((repeat_idx, random_state, cv))
    return cv_list


def generate_cv_splits(X, y, n_splits=5, n_repeats=10, base_random_state=42):
    """Flattens the repeated-CV scheme into one list of dicts (repeat_idx, fold_idx,
    random_state, train_idx, val_idx) -- easier to iterate over than nested loops."""
    all_splits = []
    
    for repeat_idx in range(n_repeats):
        random_state = base_random_state + repeat_idx
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y['event'])):
            all_splits.append({
                'repeat_idx': repeat_idx,
                'fold_idx': fold_idx,
                'random_state': random_state,
                'train_idx': train_idx,
                'val_idx': val_idx
            })
    
    return all_splits


# =============================================================================
# COX MODEL FUNCTIONS - REPEATED CV
# =============================================================================

def cox_repeated_cross_validation(X, y, alpha, l1_ratio, metrics, 
                                  scale_data=True, scaler_type='standard',
                                  binary_features=None, continuous_features=None,
                                  n_splits=5, n_repeats=10, base_random_state=42):
    """Repeated stratified CV for a Cox model at one (alpha, l1_ratio) pair. X should be
    passed unscaled when scale_data=True -- scaling is fit per-fold to avoid leakage.
    Returns a long-format DataFrame with repeat_idx, fold_idx and one column per metric.
    """
    from functions.cox import cox_model_evaluation

    all_splits = generate_cv_splits(X, y, n_splits, n_repeats, base_random_state)

    results = []

    for split in all_splits:
        repeat_idx = split['repeat_idx']
        fold_idx = split['fold_idx']
        train_idx = split['train_idx']
        val_idx = split['val_idx']

        # unscaled fold data
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y[val_idx]

        if scale_data:
            if scaler_type == 'standard':
                scaler = StandardScaler()
            elif scaler_type == 'minmax':
                scaler = MinMaxScaler()
            elif scaler_type == 'mixed':
                scaler = ColumnTransformer([
                    ('binary', MinMaxScaler(), binary_features),
                    ('continuous', StandardScaler(), continuous_features)
                ])
            else:
                raise ValueError(f"Unknown scaling method: {scaler_type}")
            
            # Scale INSIDE the fold to prevent data leakage
            X_train_fold_scaled = scaler.fit_transform(X_train_fold)
            X_val_fold_scaled = scaler.transform(X_val_fold)
            
            # Convert back to DataFrame
            X_train_fold_scaled = pd.DataFrame(
                X_train_fold_scaled, 
                columns=X_train_fold.columns, 
                index=X_train_fold.index
            )
            X_val_fold_scaled = pd.DataFrame(
                X_val_fold_scaled, 
                columns=X_val_fold.columns, 
                index=X_val_fold.index
            )
        else:
            X_train_fold_scaled = X_train_fold
            X_val_fold_scaled = X_val_fold
        
        # Evaluate model for this fold
        try:
            fold_results, _ = cox_model_evaluation(
                X_train_fold_scaled, y_train_fold, X_val_fold_scaled, y_val_fold, 
                alpha, l1_ratio, metrics
            )
        except Exception as e:
            fold_results = {metric: np.nan for metric in metrics}
        
        # Store results
        result_row = {
            'repeat_idx': repeat_idx,
            'fold_idx': fold_idx,
        }
        result_row.update(fold_results)
        results.append(result_row)
    
    results_df = pd.DataFrame(results)
    return results_df


def cox_repeated_grid_search(X, y, alpha_range, l1_ratio_range, metrics,
                             scale_data=True, scaler_type='standard',
                             binary_features=None, continuous_features=None,
                             n_splits=5, n_repeats=10, base_random_state=42,
                             verbose=True):
    """
    Grid search over (alpha, l1_ratio) using the same repeated CV splits for every combo
    (generated once up front, so comparisons across the grid are apples-to-apples). Returns
    the full per-fold results_df, an aggregated summary_df, and best_params per metric.
    """
    from functions.cox import cox_model_evaluation

    start_time = time.time()

    all_splits = generate_cv_splits(X, y, n_splits, n_repeats, base_random_state)

    total_combinations = len(alpha_range) * len(l1_ratio_range)
    
    if verbose:
        print(f"{'='*80}")
        print(f"COX MODEL REPEATED GRID SEARCH")
        print(f"{'='*80}")
        print(f"L1 ratios: {l1_ratio_range}")
        print(f"Alphas: {len(alpha_range)} values from {min(alpha_range):.2e} to {max(alpha_range):.2e}")
        print(f"Total hyperparameter combinations: {total_combinations}")
        print(f"Folds per repetition: {n_splits}")
        print(f"Repetitions: {n_repeats}")
        print(f"Total evaluations: {total_combinations * n_splits * n_repeats}")
        print(f"Scaling: {'Yes (' + scaler_type + ')' if scale_data else 'No'}")
        print(f"{'='*80}\n")
    
    all_results = []
    current_iteration = 0
    
    for l1_ratio in l1_ratio_range:
        for alpha in alpha_range:
            current_iteration += 1
            
            if verbose:
                elapsed = time.time() - start_time
                avg_time = elapsed / current_iteration if current_iteration > 0 else 0
                remaining = avg_time * (total_combinations - current_iteration)
                print(f"[{current_iteration}/{total_combinations}] "
                      f"l1={l1_ratio:.3f}, alpha={alpha:.2e} "
                      f"(elapsed: {elapsed/60:.1f}min, est. remaining: {remaining/60:.1f}min)")
            
            for split in all_splits:
                repeat_idx = split['repeat_idx']
                fold_idx = split['fold_idx']
                train_idx = split['train_idx']
                val_idx = split['val_idx']

                X_train_fold = X.iloc[train_idx]
                y_train_fold = y[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_val_fold = y[val_idx]

                if scale_data:
                    if scaler_type == 'standard':
                        scaler = StandardScaler()
                    elif scaler_type == 'minmax':
                        scaler = MinMaxScaler()
                    elif scaler_type == 'mixed':
                        scaler = ColumnTransformer([
                            ('binary', MinMaxScaler(), binary_features),
                            ('continuous', StandardScaler(), continuous_features)
                        ])
                    else:
                        raise ValueError(f"Unknown scaling method: {scaler_type}")
                    
                    # fit scaler on train only, avoid leakage into val
                    X_train_fold_scaled = scaler.fit_transform(X_train_fold)
                    X_val_fold_scaled = scaler.transform(X_val_fold)
                    
                    X_train_fold_scaled = pd.DataFrame(
                        X_train_fold_scaled, 
                        columns=X_train_fold.columns, 
                        index=X_train_fold.index
                    )
                    X_val_fold_scaled = pd.DataFrame(
                        X_val_fold_scaled, 
                        columns=X_val_fold.columns, 
                        index=X_val_fold.index
                    )
                else:
                    X_train_fold_scaled = X_train_fold
                    X_val_fold_scaled = X_val_fold
                
                try:
                    fold_results, _ = cox_model_evaluation(
                        X_train_fold_scaled, y_train_fold, X_val_fold_scaled, y_val_fold,
                        alpha, l1_ratio, metrics
                    )
                except Exception as e:
                    fold_results = {metric: np.nan for metric in metrics}

                result_row = {
                    'alpha': alpha,
                    'l1_ratio': l1_ratio,
                    'repeat_idx': repeat_idx,
                    'fold_idx': fold_idx,
                }
                result_row.update(fold_results)
                all_results.append(result_row)
    
    results_df = pd.DataFrame(all_results)

    # aggregate by hyperparameters, with per-repeat and per-fold breakdowns kept alongside
    summary_rows = []
    for l1_ratio in l1_ratio_range:
        for alpha in alpha_range:
            mask = (results_df['alpha'] == alpha) & (results_df['l1_ratio'] == l1_ratio)
            subset = results_df[mask]
            
            row = {
                'alpha': alpha,
                'l1_ratio': l1_ratio,
            }
            for metric in metrics:
                row[f'{metric}_mean'] = subset[metric].mean()
                row[f'{metric}_std'] = subset[metric].std()
                for repeat_idx in range(n_repeats):
                    repeat_mask = subset['repeat_idx'] == repeat_idx
                    row[f'{metric}_repeat_{repeat_idx+1}_mean'] = subset[repeat_mask][metric].mean()
                for repeat_idx in range(n_repeats):
                    for fold_idx in range(n_splits):
                        fold_repeat_mask = (subset['repeat_idx'] == repeat_idx) & (subset['fold_idx'] == fold_idx)
                        if fold_repeat_mask.any():
                            row[f'{metric}_repeat_{repeat_idx+1}_fold_{fold_idx+1}'] = subset[fold_repeat_mask][metric].values[0]
                        else:
                            row[f'{metric}_repeat_{repeat_idx+1}_fold_{fold_idx+1}'] = np.nan
            
            summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)

    best_params = {}
    for metric in metrics:
        mean_col = f'{metric}_mean'
        valid_results = summary_df[~summary_df[mean_col].isna()].copy()
        
        if len(valid_results) == 0:
            best_params[metric] = {
                'l1_ratio': np.nan,
                'alpha': np.nan,
                'mean': np.nan,
                'std': np.nan
            }
            continue
        
        # lower is better for brier/deviance
        if 'brier' in metric.lower() or 'deviance' in metric.lower():
            best_idx = valid_results[mean_col].idxmin()
        else:
            best_idx = valid_results[mean_col].idxmax()

        best_row = summary_df.loc[best_idx]
        best_params[metric] = {
            'l1_ratio': best_row['l1_ratio'],
            'alpha': best_row['alpha'],
            'mean': best_row[f'{metric}_mean'],
            'std': best_row[f'{metric}_std']
        }
    
    if verbose:
        elapsed_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"REPEATED GRID SEARCH COMPLETE")
        print(f"{'='*80}")
        print(f"Total time: {elapsed_time/60:.1f} minutes")
        print(f"\nBest parameters for each metric:")
        print(f"{'-'*80}")
        for metric, params in best_params.items():
            if not np.isnan(params['mean']):
                print(f"\n{metric}:")
                print(f"  l1_ratio: {params['l1_ratio']:.4f}")
                print(f"  alpha: {params['alpha']:.4e}")
                print(f"  Score: {params['mean']:.4f} ± {params['std']:.4f}")
        print(f"\n{'='*80}\n")
    
    return results_df, summary_df, best_params


# =============================================================================
# SUBSAMPLING COX MODEL FUNCTIONS - REPEATED CV
# =============================================================================

def subsampling_repeated_cross_validation(dataset, X, y, alpha, l1_ratio, metrics, 
                                          n_controls_per_case, feature_columns,
                                          event_col='Regional LN recurrence', 
                                          time_col='Survival days',
                                          scale_data=True, scaler_type='standard',
                                          continuous_features=None,
                                          use_sample_weights=False, weight_ratio=None,
                                          n_splits=5, n_repeats=10, base_random_state=42):
    """Same idea as cox_repeated_cross_validation but with case-control subsampling on
    the training side of each fold. weight_ratio, if use_sample_weights is set, controls
    how much events get upweighted: None uses balanced inverse-frequency weights, a number
    fixes the event weight directly (non-events stay at 1). Returns a DataFrame with
    repeat_idx, fold_idx and the requested metrics.
    """
    all_splits = generate_cv_splits(X, y, n_splits, n_repeats, base_random_state)

    results = []

    for split in all_splits:
        repeat_idx = split['repeat_idx']
        fold_idx = split['fold_idx']
        train_idx = split['train_idx']
        val_idx = split['val_idx']
        random_state = split['random_state']

        train_fold = dataset.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y[val_idx]

        # separate random state per fold so the subsampling draw isn't tied to the CV split itself
        subsampling_random_state = random_state + fold_idx * 100
        X_train_sub, y_train_sub, sub_dataset = sample_case_control(
            train_fold, event_col, time_col, n_controls_per_case, feature_columns,
            random_state=subsampling_random_state
        )

        # scaler is built fresh per fold (fit later on train only) to avoid leakage
        if scale_data:
            if continuous_features is not None:
                all_columns = X.columns.tolist()
                continuous_indices = [all_columns.index(col) for col in continuous_features]
                non_continuous_indices = [i for i in range(len(all_columns)) if i not in continuous_indices]

                if scaler_type == 'standard':
                    cont_scaler = StandardScaler()
                elif scaler_type == 'minmax':
                    cont_scaler = MinMaxScaler()
                else:
                    raise ValueError(f"Unknown scaling method: {scaler_type}")

                transformers = [('continuous', cont_scaler, continuous_indices)]
                if non_continuous_indices:
                    transformers.append(('non_continuous', 'passthrough', non_continuous_indices))

                scaler = ColumnTransformer(transformers)
            else:
                if scaler_type == 'standard':
                    scaler = StandardScaler()
                elif scaler_type == 'minmax':
                    scaler = MinMaxScaler()
                else:
                    raise ValueError(f"Unknown scaling method: {scaler_type}")
        else:
            scaler = None

        if use_sample_weights:
            sample_weight = _compute_class_weights(y_train_sub, event_col_name='event', weight_ratio=weight_ratio)
        else:
            sample_weight = None

        try:
            fold_results = subsampling_model_evaluation(
                X_train_sub, y_train_sub, X_val_fold, y_val_fold,
                alpha, l1_ratio, metrics, scaler=scaler,
                y_train_full=y[train_idx], sample_weight=sample_weight
            )
        except Exception as e:
            fold_results = {metric: np.nan for metric in metrics}

        result_row = {
            'repeat_idx': repeat_idx,
            'fold_idx': fold_idx,
        }
        result_row.update(fold_results)
        results.append(result_row)
    
    results_df = pd.DataFrame(results)
    return results_df


def subsampling_repeated_grid_search(dataset, X, y, alpha_range, l1_ratio_range, 
                                     controls_per_case_range, metrics, feature_columns,
                                     event_col='Regional LN recurrence', 
                                     time_col='Survival days',
                                     scale_data=True, scaler_type='standard',
                                     continuous_features=None,
                                     use_sample_weights=False, weight_ratio=None,
                                     n_splits=5, n_repeats=10, base_random_state=42,
                                     verbose=True, fit_timeout=None):
    """
    Grid search over (alpha, l1_ratio, n_controls_per_case) with repeated CV, subsampling
    applied per fold. controls_per_case_range can mix integers with the string 'Full' (train
    on the whole fold, no subsampling) in the same sweep.

    weight_ratio behaves as in subsampling_repeated_cross_validation. fit_timeout guards
    against pathological fits (extreme sample weights can make Coxnet converge very slowly)
    -- a fit that runs past it is treated as failed for that fold rather than hanging the
    whole search.

    Returns results_df (every fold), summary_df (aggregated per combo), best_params (per metric).
    """

    start_time = time.time()

    all_splits = generate_cv_splits(X, y, n_splits, n_repeats, base_random_state)

    # Total iterations
    total_combinations = len(alpha_range) * len(l1_ratio_range) * len(controls_per_case_range)
    
    if verbose:
        print(f"{'='*80}")
        print(f"SUBSAMPLING REPEATED GRID SEARCH")
        print(f"{'='*80}")
        print(f"Controls per case: {controls_per_case_range}")
        print(f"L1 ratios: {l1_ratio_range}")
        print(f"Alphas: {len(alpha_range)} values from {min(alpha_range):.2e} to {max(alpha_range):.2e}")
        print(f"Total hyperparameter combinations: {total_combinations}")
        print(f"Folds per repetition: {n_splits}")
        print(f"Repetitions: {n_repeats}")
        print(f"Total evaluations: {total_combinations * n_splits * n_repeats}")
        print(f"Scaling: {'Yes (' + scaler_type + ')' if scale_data else 'No'}")
        if use_sample_weights:
            weight_str = 'balanced' if weight_ratio is None else f'ratio={weight_ratio}'
            print(f"Sample weights: Yes ({weight_str})")
        else:
            print(f"Sample weights: No")
        print(f"{'='*80}\n")
    
    all_results = []
    current_iteration = 0
    
    for n_controls in controls_per_case_range:
        for l1_ratio in l1_ratio_range:
            for alpha in alpha_range:
                current_iteration += 1
                
                if verbose:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / current_iteration if current_iteration > 0 else 0
                    remaining = avg_time * (total_combinations - current_iteration)
                    controls_str = 'Full (no subsampling)' if n_controls == 'Full' else str(n_controls)
                    print(f"[{current_iteration}/{total_combinations}] "
                          f"controls={controls_str}, l1={l1_ratio:.3f}, alpha={alpha:.2e} "
                          f"(elapsed: {elapsed/60:.1f}min, est. remaining: {remaining/60:.1f}min)")
                
                # Run repeated CV for this hyperparameter combination
                for split in all_splits:
                    repeat_idx = split['repeat_idx']
                    fold_idx = split['fold_idx']
                    train_idx = split['train_idx']
                    val_idx = split['val_idx']
                    random_state = split['random_state']
                    
                    X_val_fold = X.iloc[val_idx]
                    y_val_fold = y[val_idx]

                    # fresh scaler per fold, fit on train only (no leakage)
                    if scale_data:
                        if continuous_features is not None:
                            all_columns = X.columns.tolist()
                            continuous_indices = [all_columns.index(col) for col in continuous_features]
                            non_continuous_indices = [i for i in range(len(all_columns)) if i not in continuous_indices]

                            if scaler_type == 'standard':
                                cont_scaler = StandardScaler()
                            elif scaler_type == 'minmax':
                                cont_scaler = MinMaxScaler()
                            else:
                                raise ValueError(f"Unknown scaling method: {scaler_type}")
                            
                            transformers = [('continuous', cont_scaler, continuous_indices)]
                            if non_continuous_indices:
                                transformers.append(('non_continuous', 'passthrough', non_continuous_indices))
                            
                            scaler = ColumnTransformer(transformers)
                        else:
                            if scaler_type == 'standard':
                                scaler = StandardScaler()
                            elif scaler_type == 'minmax':
                                scaler = MinMaxScaler()
                            else:
                                raise ValueError(f"Unknown scaling method: {scaler_type}")
                    else:
                        scaler = None

                    if n_controls == 'Full':
                        # no subsampling -- train on the whole fold
                        X_train_fold = X.iloc[train_idx]
                        y_train_fold = y[train_idx]

                        if use_sample_weights:
                            sample_weight = _compute_class_weights(y_train_fold, event_col_name='event', weight_ratio=weight_ratio)
                        else:
                            sample_weight = None

                        try:
                            fold_results = subsampling_model_evaluation(
                                X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                                alpha, l1_ratio, metrics, scaler=scaler,
                                y_train_full=y_train_fold, sample_weight=sample_weight,
                                fit_timeout=fit_timeout
                            )
                        except Exception as e:
                            if verbose:
                                print(f"  Fold {repeat_idx+1}-{fold_idx+1} error: {type(e).__name__}: {str(e)}")
                            fold_results = {metric: np.nan for metric in metrics}
                    else:
                        train_fold = dataset.iloc[train_idx]
                        subsampling_random_state = random_state + fold_idx * 100
                        try:
                            X_train_sub, y_train_sub, sub_dataset = sample_case_control(
                                train_fold, event_col, time_col, n_controls, feature_columns,
                                random_state=subsampling_random_state
                            )

                            if use_sample_weights:
                                sample_weight = _compute_class_weights(y_train_sub, event_col_name='event', weight_ratio=weight_ratio)
                            else:
                                sample_weight = None

                            fold_results = subsampling_model_evaluation(
                                X_train_sub, y_train_sub, X_val_fold, y_val_fold,
                                alpha, l1_ratio, metrics, scaler=scaler,
                                y_train_full=y[train_idx], sample_weight=sample_weight,
                                fit_timeout=fit_timeout
                            )
                        except Exception as e:
                            if verbose:
                                print(f"  Fold {repeat_idx+1}-{fold_idx+1} error: {type(e).__name__}: {str(e)}")
                            fold_results = {metric: np.nan for metric in metrics}

                    result_row = {
                        'n_controls': n_controls,
                        'alpha': alpha,
                        'l1_ratio': l1_ratio,
                        'repeat_idx': repeat_idx,
                        'fold_idx': fold_idx,
                    }
                    result_row.update(fold_results)
                    all_results.append(result_row)

                    # a fold with no c-index already dooms this combo's mean (see the
                    # incomplete_combo check below), so bail out of its remaining
                    # folds/repeats now rather than paying for fits we'll throw away
                    if 'c-index' in metrics:
                        c_index_value = fold_results.get('c-index', np.nan)
                        if pd.isna(c_index_value):
                            if verbose:
                                print(f"  c-index missing at repeat {repeat_idx+1}, fold {fold_idx+1} "
                                      f"— skipping remaining folds/repeats for this combination")
                            break

    results_df = pd.DataFrame(all_results)

    summary_rows = []
    for n_controls in controls_per_case_range:
        for l1_ratio in l1_ratio_range:
            for alpha in alpha_range:
                mask = ((results_df['n_controls'] == n_controls) &
                        (results_df['alpha'] == alpha) &
                        (results_df['l1_ratio'] == l1_ratio))
                subset = results_df[mask]

                # combos skipped early via the break above end up with fewer than
                # n_splits*n_repeats rows -- flag as incomplete so we don't report a
                # partial, artificially-good-looking mean for them
                incomplete_combo = len(subset) < (n_splits * n_repeats)

                row = {
                    'n_controls': n_controls,
                    'alpha': alpha,
                    'l1_ratio': l1_ratio,
                }
                for metric in metrics:
                    if incomplete_combo:
                        row[f'{metric}_mean'] = np.nan
                        row[f'{metric}_std'] = np.nan
                    else:
                        row[f'{metric}_mean'] = subset[metric].mean()
                        row[f'{metric}_std'] = subset[metric].std()
                    for repeat_idx in range(n_repeats):
                        repeat_mask = subset['repeat_idx'] == repeat_idx
                        row[f'{metric}_repeat_{repeat_idx+1}_mean'] = subset[repeat_mask][metric].mean()
                    for repeat_idx in range(n_repeats):
                        for fold_idx in range(n_splits):
                            fold_repeat_mask = (subset['repeat_idx'] == repeat_idx) & (subset['fold_idx'] == fold_idx)
                            if fold_repeat_mask.any():
                                row[f'{metric}_repeat_{repeat_idx+1}_fold_{fold_idx+1}'] = subset[fold_repeat_mask][metric].values[0]
                            else:
                                row[f'{metric}_repeat_{repeat_idx+1}_fold_{fold_idx+1}'] = np.nan
                
                summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)

    best_params = {}
    for metric in metrics:
        mean_col = f'{metric}_mean'
        valid_results = summary_df[~summary_df[mean_col].isna()].copy()
        
        if len(valid_results) == 0:
            best_params[metric] = {
                'n_controls': np.nan,
                'l1_ratio': np.nan,
                'alpha': np.nan,
                'mean': np.nan,
                'std': np.nan
            }
            continue
        
        if 'brier' in metric.lower() or 'deviance' in metric.lower():
            best_idx = valid_results[mean_col].idxmin()
        else:
            best_idx = valid_results[mean_col].idxmax()
        
        best_row = summary_df.loc[best_idx]
        n_controls_value = best_row['n_controls']
        # 'Full' stays a string; numeric n_controls comes back as float from the DataFrame
        if n_controls_value != 'Full':
            n_controls_value = int(n_controls_value)
        
        best_params[metric] = {
            'n_controls': n_controls_value,
            'l1_ratio': best_row['l1_ratio'],
            'alpha': best_row['alpha'],
            'mean': best_row[f'{metric}_mean'],
            'std': best_row[f'{metric}_std']
        }
    
    if verbose:
        elapsed_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"REPEATED GRID SEARCH COMPLETE")
        print(f"{'='*80}")
        print(f"Total time: {elapsed_time/60:.1f} minutes")
        print(f"\nBest parameters for each metric:")
        print(f"{'-'*80}")
        for metric, params in best_params.items():
            if not np.isnan(params['mean']):
                n_controls_str = 'Full (no subsampling)' if params['n_controls'] == 'Full' else str(params['n_controls'])
                print(f"\n{metric}:")
                print(f"  n_controls_per_case: {n_controls_str}")
                print(f"  l1_ratio: {params['l1_ratio']:.4f}")
                print(f"  alpha: {params['alpha']:.4e}")
                print(f"  Score: {params['mean']:.4f} ± {params['std']:.4f}")
        print(f"\n{'='*80}\n")
    
    return results_df, summary_df, best_params


def subsampling_bootstrap_optimism_grid_search(dataset, X, y, alpha_range, l1_ratio_range,
                                               controls_per_case_range, feature_columns,
                                               event_col='Regional LN recurrence',
                                               time_col='Survival days',
                                               scale_data=True, scaler_type='standard',
                                               continuous_features=None,
                                               use_sample_weights=False, weight_ratio=None,
                                               n_bootstraps=200, base_random_state=42,
                                               verbose=True, metrics=None,
                                               return_bootstrap_df=False, fit_timeout=None,
                                               collect_coefficients=False,
                                               require_complete_bootstraps=True):
    """
    Bootstrap-optimism-corrected grid search for the (optionally subsampled) Cox model.
    This is the "real" evaluation -- the CV grid searches elsewhere in this module only
    rank hyperparameters, the numbers actually reported come from here.

    Per hyperparameter combo:
      A) fit on the full training set, evaluate on itself -> C_apparent
      B) for n_bootstraps stratified (with-replacement) draws: fit on the bootstrap
         sample, evaluate on itself (C_boot) and on the original training set (C_test)
      C) optimism O = C_boot - C_test per run, then C_final = C_apparent - mean(O)

    controls_per_case_range can mix ints with 'Full' (no subsampling) as elsewhere.
    metrics beyond 'c-index' get computed too if passed in, but c-index drives the
    optimism correction and is always included regardless.

    Some args worth a note:
    - require_complete_bootstraps (default True): a run with a missing c-index (bad
      fit, non-convergence, timeout) aborts the rest of that combo's bootstrap runs,
      since C_final needs a full, unbiased set to average over. Set False only when
      the goal is collect_coefficients CIs rather than C_final -- there, an occasional
      failed resample is fine to drop rather than letting it starve the CI of draws.
      Either way the combo's summary row still comes back NaN/incomplete.
    - collect_coefficients: reads the Step B (bootstrap-sample) model's coefficients on
      every run -- no extra fits, Step B is already being fit for C_boot -- and adds
      coef_<feature> columns. Column order follows what the scaler actually outputs
      (continuous_features first, in the given order, then the rest of feature_columns)
      rather than feature_columns order, since ColumnTransformer reorders columns when
      continuous_features is set. This is meant for bootstrap CIs around one
      already-chosen combination, not for use across a large grid.
    - fit_timeout: a fit that hangs (extreme sample weights can make Coxnet converge
      very slowly) counts as failed instead of stalling the whole search.

    Returns results_df (one row per bootstrap run), summary_df (one row per combo, with
    C_apparent/optimism/C_final plus n_bootstraps_completed -- incomplete combos come
    back NaN so they can never be picked as best), best_params (highest C_final), and
    bootstrap_df too if return_bootstrap_df=True (same table as results_df).
    """

    def _build_scaler_template(X_ref):
        if not scale_data:
            return None

        if continuous_features is not None:
            all_columns = X_ref.columns.tolist()
            continuous_indices = [all_columns.index(col) for col in continuous_features]
            non_continuous_indices = [i for i in range(len(all_columns)) if i not in continuous_indices]

            if scaler_type == 'standard':
                cont_scaler = StandardScaler()
            elif scaler_type == 'minmax':
                cont_scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaling method: {scaler_type}")

            transformers = [('continuous', cont_scaler, continuous_indices)]
            if non_continuous_indices:
                transformers.append(('non_continuous', 'passthrough', non_continuous_indices))
            return ColumnTransformer(transformers)

        if scaler_type == 'standard':
            return StandardScaler()
        if scaler_type == 'minmax':
            return MinMaxScaler()
        raise ValueError(f"Unknown scaling method: {scaler_type}")

    start_time = time.time()
    total_combinations = len(alpha_range) * len(l1_ratio_range) * len(controls_per_case_range)
    metrics_to_run = ['c-index']
    if metrics is not None:
        for metric_name in metrics:
            if metric_name not in metrics_to_run:
                metrics_to_run.append(metric_name)

    def _metric_suffix(metric_name):
        return metric_name.replace('-', '_')

    if verbose:
        print(f"{'='*80}")
        print("SUBSAMPLING STRATIFIED BOOTSTRAP OPTIMISM GRID SEARCH")
        print(f"{'='*80}")
        print(f"Controls per case: {controls_per_case_range}")
        print(f"L1 ratios: {l1_ratio_range}")
        print(f"Alphas: {len(alpha_range)} values from {min(alpha_range):.2e} to {max(alpha_range):.2e}")
        print(f"Bootstrap runs per combination: {n_bootstraps}")
        print(f"Total hyperparameter combinations: {total_combinations}")
        print(f"Total model loops (B/C runs): {total_combinations * n_bootstraps}")
        print(f"Scaling: {'Yes (' + scaler_type + ')' if scale_data else 'No'}")
        if use_sample_weights:
            weight_str = 'balanced' if weight_ratio is None else f'ratio={weight_ratio}'
            print(f"Sample weights: Yes ({weight_str})")
        else:
            print("Sample weights: No")
        print(f"{'='*80}\n")

    # Precompute class indices for stratified bootstrap
    event_mask = dataset[event_col].values.astype(int) == 1
    event_indices = np.where(event_mask)[0]
    non_event_indices = np.where(~event_mask)[0]

    if len(event_indices) == 0 or len(non_event_indices) == 0:
        raise ValueError("Stratified bootstrap requires both event and non-event samples.")

    # Column order the scaler actually outputs: with continuous_features set,
    # _build_scaler_template's ColumnTransformer emits the scaled continuous
    # columns first (in continuous_features order), then the passthrough columns
    # in their original feature_columns order. Without continuous_features, a
    # plain StandardScaler/MinMaxScaler preserves feature_columns order as-is.
    if collect_coefficients:
        if continuous_features is not None:
            coef_feature_order = list(continuous_features) + [
                col for col in feature_columns if col not in continuous_features
            ]
        else:
            coef_feature_order = list(feature_columns)
    else:
        coef_feature_order = None

    all_results = []
    summary_rows = []
    current_iteration = 0

    for n_controls in controls_per_case_range:
        for l1_ratio in l1_ratio_range:
            for alpha in alpha_range:
                current_iteration += 1
                combo_seed = base_random_state + current_iteration * 10000
                rng = np.random.RandomState(combo_seed)

                if verbose:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / current_iteration if current_iteration > 0 else 0
                    remaining = avg_time * (total_combinations - current_iteration)
                    controls_str = 'Full (no subsampling)' if n_controls == 'Full' else str(n_controls)
                    print(
                        f"[{current_iteration}/{total_combinations}] "
                        f"controls={controls_str}, l1={l1_ratio:.3f}, alpha={alpha:.2e} "
                        f"(elapsed: {elapsed/60:.1f}min, est. remaining: {remaining/60:.1f}min)"
                    )

                scaler_template = _build_scaler_template(X)

                # Step A: apparent performance for this parameter set
                try:
                    if n_controls == 'Full':
                        X_app_train = X.copy()
                        y_app_train = y.copy()
                        y_app_full = y_app_train
                    else:
                        X_app_train, y_app_train, _ = sample_case_control(
                            train_fold=dataset.reset_index(drop=True),
                            event_col=event_col,
                            time_col=time_col,
                            n_controls_per_case=n_controls,
                            feature_columns=feature_columns,
                            random_state=combo_seed
                        )
                        y_app_full = y

                    if use_sample_weights:
                        app_weights = _compute_class_weights(
                            y_app_train, event_col_name='event', weight_ratio=weight_ratio
                        )
                    else:
                        app_weights = None

                    app_res = subsampling_model_evaluation(
                        X_app_train, y_app_train,
                        X_app_train, y_app_train,
                        alpha, l1_ratio,
                        metrics=metrics_to_run,
                        scaler=scaler_template,
                        y_train_full=y_app_full,
                        sample_weight=app_weights,
                        fit_timeout=fit_timeout
                    )
                    c_apparent = app_res.get('c-index', np.nan)
                except Exception as e:
                    if verbose:
                        print(f"  Apparent step failed: {type(e).__name__}: {str(e)}")
                    app_res = {metric: np.nan for metric in metrics_to_run}
                    c_apparent = np.nan

                optimism_values = []
                c_boot_values = []
                c_test_values = []
                combo_incomplete = False

                # If the apparent step failed, c_final = c_apparent - avg_optimism will
                # be NaN no matter what the bootstraps produce, so this combination
                # won't be counted regardless — skip the (expensive) bootstrap loop
                # entirely instead of running n_bootstraps fits for nothing.
                if pd.isna(c_apparent):
                    combo_incomplete = True
                    if verbose:
                        print(f"  Apparent c-index missing — skipping {n_bootstraps} bootstrap runs for this combination")

                for boot_idx in range(0 if combo_incomplete else n_bootstraps):
                    try:
                        # Stratified bootstrap with replacement (per class)
                        boot_event_idx = rng.choice(event_indices, size=len(event_indices), replace=True)
                        boot_non_event_idx = rng.choice(non_event_indices, size=len(non_event_indices), replace=True)
                        boot_indices = np.concatenate([boot_event_idx, boot_non_event_idx])
                        rng.shuffle(boot_indices)

                        boot_df = dataset.iloc[boot_indices].copy().reset_index(drop=True)

                        X_boot_full = boot_df[feature_columns]
                        y_boot_full = Surv.from_arrays(
                            event=boot_df[event_col],
                            time=boot_df[time_col]
                        )

                        # Training data for this bootstrap run
                        if n_controls == 'Full':
                            X_boot_train = X_boot_full
                            y_boot_train = y_boot_full
                        else:
                            X_boot_train, y_boot_train, _ = sample_case_control(
                                train_fold=boot_df,
                                event_col=event_col,
                                time_col=time_col,
                                n_controls_per_case=n_controls,
                                feature_columns=feature_columns,
                                random_state=combo_seed + boot_idx
                            )

                        if use_sample_weights:
                            boot_weights = _compute_class_weights(
                                y_boot_train, event_col_name='event', weight_ratio=weight_ratio
                            )
                        else:
                            boot_weights = None

                        # Step B: evaluate on bootstrap training sample
                        boot_eval = subsampling_model_evaluation(
                            X_boot_train, y_boot_train,
                            X_boot_train, y_boot_train,
                            alpha, l1_ratio,
                            metrics=metrics_to_run,
                            scaler=scaler_template,
                            y_train_full=y_boot_full,
                            sample_weight=boot_weights,
                            fit_timeout=fit_timeout,
                            return_model=collect_coefficients
                        )
                        if collect_coefficients:
                            res_boot, boot_model = boot_eval
                        else:
                            res_boot, boot_model = boot_eval, None
                        c_boot = res_boot.get('c-index', np.nan)

                        if collect_coefficients:
                            if boot_model is not None and hasattr(boot_model, 'coef_'):
                                boot_coef_arr = np.asarray(boot_model.coef_)
                                if boot_coef_arr.ndim > 1:
                                    boot_coef_arr = boot_coef_arr[:, 0]
                                boot_coefs = dict(zip(coef_feature_order, boot_coef_arr))
                            else:
                                boot_coefs = {feat: np.nan for feat in coef_feature_order}

                        # Step C: evaluate same model setup on original training set
                        res_test = subsampling_model_evaluation(
                            X_boot_train, y_boot_train,
                            X, y,
                            alpha, l1_ratio,
                            metrics=metrics_to_run,
                            scaler=scaler_template,
                            y_train_full=y_boot_full,
                            sample_weight=boot_weights,
                            fit_timeout=fit_timeout
                        )
                        c_test = res_test.get('c-index', np.nan)

                        optimism = c_boot - c_test

                    except Exception as e:
                        if verbose:
                            print(
                                f"  Bootstrap {boot_idx+1}/{n_bootstraps} failed: "
                                f"{type(e).__name__}: {str(e)}"
                            )
                        res_boot = {metric: np.nan for metric in metrics_to_run}
                        res_test = {metric: np.nan for metric in metrics_to_run}
                        c_boot = np.nan
                        c_test = np.nan
                        optimism = np.nan
                        boot_coefs = {feat: np.nan for feat in coef_feature_order} if collect_coefficients else None

                    c_boot_values.append(c_boot)
                    c_test_values.append(c_test)
                    optimism_values.append(optimism)

                    bootstrap_row = {
                        'n_controls': n_controls,
                        'alpha': alpha,
                        'l1_ratio': l1_ratio,
                        'bootstrap_idx': boot_idx,
                        'bootstrap_seed': combo_seed + boot_idx,
                        'c_apparent': c_apparent,
                        'c_boot': c_boot,
                        'c_test': c_test,
                        'optimism': optimism,
                        'c_final': (
                            c_apparent - optimism
                            if not np.isnan(c_apparent) and not np.isnan(optimism)
                            else np.nan
                        )
                    }

                    if collect_coefficients:
                        for feat in coef_feature_order:
                            bootstrap_row[f'coef_{feat}'] = boot_coefs.get(feat, np.nan)

                    for metric_name in metrics_to_run:
                        metric_key = _metric_suffix(metric_name)
                        bootstrap_row[f'app_{metric_key}'] = app_res.get(metric_name, np.nan)
                        bootstrap_row[f'boot_{metric_key}'] = res_boot.get(metric_name, np.nan)
                        bootstrap_row[f'test_{metric_key}'] = res_test.get(metric_name, np.nan)
                        boot_metric_value = bootstrap_row[f'boot_{metric_key}']
                        test_metric_value = bootstrap_row[f'test_{metric_key}']
                        bootstrap_row[f'optimism_{metric_key}'] = (
                            boot_metric_value - test_metric_value
                            if not (np.isnan(boot_metric_value) or np.isnan(test_metric_value))
                            else np.nan
                        )

                    all_results.append(bootstrap_row)

                    # Same rule as the fold loop in subsampling_repeated_grid_search:
                    # a missing c-index (e.g. the fit timed out) makes this
                    # combination's optimism average rest on incomplete data, so it
                    # won't be counted regardless — stop now instead of spending the
                    # remaining bootstraps on fits that are failing the same way.
                    if pd.isna(optimism):
                        combo_incomplete = True
                        if require_complete_bootstraps:
                            if verbose:
                                print(f"  c-index missing at bootstrap {boot_idx+1}/{n_bootstraps} "
                                      f"— skipping remaining bootstraps for this combination")
                            break
                        elif verbose:
                            print(f"  c-index missing at bootstrap {boot_idx+1}/{n_bootstraps} "
                                  f"— continuing (require_complete_bootstraps=False)")

                # A combination that was cut short (see the checks above) has fewer
                # bootstraps than requested — averaging what did run would report a
                # partial (and misleadingly optimistic) result, so treat it as invalid.
                if combo_incomplete:
                    avg_optimism = np.nan
                    c_final = np.nan
                    c_boot_mean = np.nan
                    c_boot_std = np.nan
                    c_test_mean = np.nan
                    c_test_std = np.nan
                else:
                    avg_optimism = float(np.nanmean(optimism_values)) if len(optimism_values) > 0 else np.nan
                    c_final = c_apparent - avg_optimism if not np.isnan(c_apparent) and not np.isnan(avg_optimism) else np.nan
                    c_boot_mean = float(np.nanmean(c_boot_values)) if len(c_boot_values) > 0 else np.nan
                    c_boot_std = float(np.nanstd(c_boot_values)) if len(c_boot_values) > 0 else np.nan
                    c_test_mean = float(np.nanmean(c_test_values)) if len(c_test_values) > 0 else np.nan
                    c_test_std = float(np.nanstd(c_test_values)) if len(c_test_values) > 0 else np.nan

                summary_rows.append({
                    'n_controls': n_controls,
                    'alpha': alpha,
                    'l1_ratio': l1_ratio,
                    'n_bootstraps': n_bootstraps,
                    'n_bootstraps_completed': len(optimism_values),
                    'c_apparent': c_apparent,
                    'avg_optimism': avg_optimism,
                    'c_final': c_final,
                    'c_boot_mean': c_boot_mean,
                    'c_boot_std': c_boot_std,
                    'c_test_mean': c_test_mean,
                    'c_test_std': c_test_std,
                })

    results_df = pd.DataFrame(all_results)
    summary_df = pd.DataFrame(summary_rows)

    valid_summary = summary_df[~summary_df['c_final'].isna()].copy()
    if len(valid_summary) == 0:
        best_params = {
            'n_controls': np.nan,
            'l1_ratio': np.nan,
            'alpha': np.nan,
            'c_apparent': np.nan,
            'avg_optimism': np.nan,
            'c_final': np.nan,
        }
    else:
        best_idx = valid_summary['c_final'].idxmax()
        best_row = summary_df.loc[best_idx]
        n_controls_value = best_row['n_controls']
        if n_controls_value != 'Full':
            n_controls_value = int(n_controls_value)
        best_params = {
            'n_controls': n_controls_value,
            'l1_ratio': best_row['l1_ratio'],
            'alpha': best_row['alpha'],
            'c_apparent': best_row['c_apparent'],
            'avg_optimism': best_row['avg_optimism'],
            'c_final': best_row['c_final'],
        }

    if verbose:
        elapsed_time = time.time() - start_time
        print(f"\n{'='*80}")
        print("BOOTSTRAP OPTIMISM GRID SEARCH COMPLETE")
        print(f"{'='*80}")
        print(f"Total time: {elapsed_time/60:.1f} minutes")
        if not np.isnan(best_params['c_final']):
            n_controls_str = 'Full (no subsampling)' if best_params['n_controls'] == 'Full' else str(best_params['n_controls'])
            print("\nBest parameters by optimism-corrected c-index:")
            print(f"  n_controls_per_case: {n_controls_str}")
            print(f"  l1_ratio: {best_params['l1_ratio']:.4f}")
            print(f"  alpha: {best_params['alpha']:.4e}")
            print(f"  C_apparent: {best_params['c_apparent']:.4f}")
            print(f"  Avg optimism: {best_params['avg_optimism']:.4f}")
            print(f"  C_final: {best_params['c_final']:.4f}")
        print(f"\n{'='*80}\n")

    if return_bootstrap_df:
        return results_df, summary_df, best_params, results_df.copy()

    return results_df, summary_df, best_params


# =============================================================================
# ENSEMBLE MODEL FUNCTIONS - REPEATED CV
# =============================================================================

from sksurv.nonparametric import kaplan_meier_estimator
import numpy as np

def get_ipcw_weights(y_train, y_test, eval_time, max_weight=20.0):
    """IPCW weights for ROC-AUC at eval_time, capped at max_weight so the tail doesn't
    blow up. (Duplicated from the Cox section above so the ensemble code doesn't need
    to import across modules -- same logic. Use get_ipcw_weights_for_prauc for PR-AUC.)"""
    cens_time, cens_surv = kaplan_meier_estimator(
        ~y_train['event'], y_train['time']
    )

    weights = np.zeros(len(y_test))
    valid_mask = np.zeros(len(y_test), dtype=bool)
    y_binary = np.zeros(len(y_test), dtype=int)

    for i, (event, time) in enumerate(zip(y_test['event'], y_test['time'])):
        if time <= eval_time and event:
            valid_mask[i] = True
            y_binary[i] = 1
            idx = np.searchsorted(cens_time, time) - 1
            g_t = cens_surv[idx] if idx >= 0 else 1.0
            weights[i] = min(1.0 / g_t, max_weight) if g_t > 0 else 0.0

        elif time > eval_time:
            valid_mask[i] = True
            y_binary[i] = 0
            idx = np.searchsorted(cens_time, eval_time) - 1
            g_t = cens_surv[idx] if idx >= 0 else 1.0
            weights[i] = min(1.0 / g_t, max_weight) if g_t > 0 else 0.0

    return valid_mask, y_binary, weights


def get_ipcw_weights_for_prauc(y_train, y_test, eval_time, max_weight=20.0):
    """PR-AUC version of the IPCW weighting above. Positives (event by eval_time) get
    1/G(T_i); negatives (alive past eval_time) get 1/G(eval_time); anyone censored
    before eval_time is dropped rather than downweighted, since we don't know what
    they'd have done. G is the KM estimate of P(censoring > t)."""
    cens_time, cens_surv = kaplan_meier_estimator(
        ~y_train['event'],  # flip so the KM here estimates the censoring distribution
        y_train['time']
    )

    weights = np.zeros(len(y_test))
    valid_mask = np.zeros(len(y_test), dtype=bool)
    y_binary = np.zeros(len(y_test), dtype=int)
    
    for i, (event, time) in enumerate(zip(y_test['event'], y_test['time'])):
        if event and time <= eval_time:
            # Case 1: Event occurred before or at eval_time → POSITIVE
            valid_mask[i] = True
            y_binary[i] = 1
            
            # Weight by 1/G(T_i) where T_i is the event time
            idx = np.searchsorted(cens_time, time) - 1
            g_ti = cens_surv[idx] if idx >= 0 else 1.0
            weights[i] = min(1.0 / g_ti, max_weight) if g_ti > 0 else max_weight
            
        elif time > eval_time:
            # Case 2: Survived past eval_time (censored or not) → NEGATIVE
            valid_mask[i] = True
            y_binary[i] = 0
            
            # Weight by 1/G(eval_time)
            idx = np.searchsorted(cens_time, eval_time) - 1
            g_t = cens_surv[idx] if idx >= 0 else 1.0
            weights[i] = min(1.0 / g_t, max_weight) if g_t > 0 else max_weight
            
        # Case 3: Censored before eval_time → EXCLUDED (valid_mask stays False)
        # These observations are uninformative for PR-AUC at this time point
    
    return valid_mask, y_binary, weights

def ensemble_repeated_cross_validation(model_class, X, y, model_params, metrics,
                                       scale_data=False, scaler_type='standard',
                                       continuous_features=None,
                                       random_state=None, n_jobs=None,
                                       n_controls_per_case=None, dataset=None, 
                                       feature_columns=None,
                                       event_col='Regional LN recurrence', 
                                       time_col='Survival days',
                                       n_iter_no_change=None, tol=1e-4,
                                       n_splits=5, n_repeats=10, base_random_state=42,
                                       verbose=False):
    """Repeated stratified CV for an ensemble model (RSF/GBM/SVM), with the same optional
    case-control subsampling used for Cox above."""

    TIME_POINTS = {
        '6m': 0.5 * 365,
        '1y': 1 * 365,
        '2y': 2 * 365,
        '3y': 3 * 365,
        '4y': 4 * 365,
        '5y': 5 * 365,
        '10y': 10 * 365,
        '14y': 14 * 365
    }

    all_splits = generate_cv_splits(X, y, n_splits, n_repeats, base_random_state)

    results = {metric: [] for metric in metrics}
    error_flags = {metric: [] for metric in metrics}
    result_rows = []

    for split in all_splits:
        repeat_idx = split['repeat_idx']
        fold_idx = split['fold_idx']
        train_idx = split['train_idx']
        val_idx = split['val_idx']
        split_random_state = split['random_state']

        X_train, X_test = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_test = y[train_idx], y[val_idx]

        # keep the un-subsampled y around for KM null model / IPCW weights later,
        # so subsampling doesn't distort the censoring distribution they're based on
        y_train_full = y_train.copy()

        if n_controls_per_case is not None and n_controls_per_case != 'Full':
            if dataset is None or feature_columns is None:
                raise ValueError("dataset and feature_columns must be provided when using subsampling")
            
            train_fold = dataset.iloc[train_idx]
            subsampling_random_state = split_random_state + fold_idx * 100
            
            X_train, y_train, _ = sample_case_control(
                train_fold=train_fold,
                event_col=event_col,
                time_col=time_col,
                n_controls_per_case=n_controls_per_case,
                feature_columns=feature_columns,
                random_state=subsampling_random_state
            )
        
        if scale_data:
            if continuous_features is not None:
                scaler_cls = StandardScaler if scaler_type == 'standard' else MinMaxScaler
                scaler = ColumnTransformer([
                    ('continuous', scaler_cls(), continuous_features)
                ], remainder='passthrough')
                
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                feature_names = continuous_features + [col for col in X_train.columns if col not in continuous_features]
                
                X_train = pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train.index)
                X_test = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index)
            else:
                scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
                
                X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
                X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        
        model_params_copy = model_params.copy()
        if random_state is not None:
            model_params_copy['random_state'] = random_state
        if n_jobs is not None:
            model_params_copy['n_jobs'] = n_jobs

        try:
            model = model_class(**model_params_copy)
        except TypeError as e:
            # some estimators don't take random_state -- fall back without it
            if 'unexpected keyword argument' in str(e):
                model_params_copy = model_params.copy()
                if random_state is not None:
                    try:
                        model = model_class(**model_params_copy, random_state=random_state)
                    except TypeError:
                        model = model_class(**model_params_copy)
                else:
                    model = model_class(**model_params_copy)
            else:
                raise
        
        try:
            fit_kwargs = {}
            monitor = None
            
            if n_iter_no_change is not None and hasattr(model, 'n_estimators'):
                if verbose:
                    print(f"    Early stopping with CV validation fold: n_iter_no_change={n_iter_no_change}, tol={tol}")
                monitor = _EarlyStoppingMonitor(
                    X_val=X_test, y_val=y_test,
                    n_iter_no_change=n_iter_no_change, tol=tol, verbose=False
                )
                fit_kwargs['monitor'] = monitor
            
            model.fit(X_train, y_train, **fit_kwargs)
            
            if verbose and monitor is not None and hasattr(model, 'n_estimators_'):
                if monitor.stopped_early:
                    saved = model_params_copy.get('n_estimators', model.n_estimators) - monitor.best_iteration - 1
                    print(f"    Early stop at iter {monitor.best_iteration+1}/{model_params_copy.get('n_estimators', '?')} (saved {saved} iters)")
                else:
                    print(f"    Completed all {model.n_estimators_} iterations (no early stop)")
        except Exception as e:
            print(f"  Error in repeat {repeat_idx+1}, fold {fold_idx+1}: Model fitting failed - {type(e).__name__}: {str(e)}")
            result_row = {'repeat_idx': repeat_idx, 'fold_idx': fold_idx}
            for metric in metrics:
                result_row[metric] = np.nan
                error_flags[metric].append(True)
            result_rows.append(result_row)
            continue
        
        result_row = {'repeat_idx': repeat_idx, 'fold_idx': fold_idx}

        # only GBM-style models expose n_estimators_ (actual trees fit, after early stopping)
        if hasattr(model, 'n_estimators') and hasattr(model, 'n_estimators_'):
            result_row['n_estimators_trained'] = model.n_estimators_
            result_row['n_estimators_max'] = model_params_copy.get('n_estimators', model.n_estimators)
            result_row['stopped_early'] = monitor.stopped_early if monitor is not None else False
        else:
            result_row['n_estimators_trained'] = None
            result_row['n_estimators_max'] = None
            result_row['stopped_early'] = None
            
        # compute once, reuse across all the per-metric branches below
        pred = None
        surv_funcs = None
        try:
            pred = model.predict(X_test)
        except Exception:
            pass
        
        if hasattr(model, 'predict_survival_function'):
            try:
                surv_funcs = model.predict_survival_function(X_test)
            except Exception:
                pass

        for metric in metrics:
            error_occurred = False
            
            if metric == 'c-index':
                try:
                    result_row[metric] = concordance_index_censored(y_test['event'], y_test['time'], pred)[0]
                except Exception as e:
                    if verbose: print(f"    Warning: C-index failed: {str(e)}")
                    result_row[metric] = np.nan
                    error_occurred = True
                    
            elif metric == 'c-index-ipcw':
                try:
                    tau = min(y_train['time'].max(), y_test['time'].max()) * 0.95
                    result_row[metric] = concordance_index_ipcw(y_train, y_test, pred, tau=tau)[0]
                except Exception as e:
                    if verbose: print(f"    Warning: C-index IPCW failed: {str(e)}")
                    result_row[metric] = np.nan
                    error_occurred = True
                    
            elif metric == 'int-brier-score':
                try:
                    if surv_funcs is not None:
                        max_time = min(y_train['time'].max(), y_test['time'].max()) * 0.95
                        times = np.percentile(y_train['time'], np.linspace(5, 90, 10))
                        times = np.unique(times[(times > 0) & (times <= max_time)])
                        
                        if len(times) > 1:
                            predicted_survival_probabilities = np.asarray([[fn(t) for t in times] for fn in surv_funcs])
                            result_row[metric] = integrated_brier_score(y_train, y_test, predicted_survival_probabilities, times)
                        else:
                            result_row[metric] = np.nan
                            error_occurred = True
                    else:
                        result_row[metric] = np.nan
                        error_occurred = True
                except Exception as e:
                    if verbose: print(f"    Warning: Brier score failed: {str(e)}")
                    result_row[metric] = np.nan
                    error_occurred = True
                    
            elif metric.startswith('brier-'):
                try:
                    time_key = metric.replace('brier-', '')
                    if surv_funcs is not None and time_key in TIME_POINTS:
                        time_point = TIME_POINTS[time_key]
                        max_time = min(y_train['time'].max(), y_test['time'].max())
                        
                        if time_point <= max_time:
                            predicted_survival_probabilities = np.asarray([fn(time_point) for fn in surv_funcs])
                            result_row[metric] = float(brier_score(y_train, y_test, predicted_survival_probabilities, time_point)[1][0])
                        else:
                            result_row[metric] = np.nan
                            error_occurred = True
                    else:
                        result_row[metric] = np.nan
                        error_occurred = True
                except Exception as e:
                    if verbose: print(f"    Warning: {metric} failed: {str(e)}")
                    result_row[metric] = np.nan
                    error_occurred = True
                    
            elif metric == 'r2-ibs':
                try:
                    if surv_funcs is not None:
                        max_time = min(y_train_full['time'].max(), y_test['time'].max()) * 0.95
                        times = np.percentile(y_train_full['time'], np.linspace(5, 90, 10))
                        times = np.unique(times[(times > 0) & (times <= max_time)])
                        
                        if len(times) > 1:
                            model_surv_probs = np.asarray([[fn(t) for t in times] for fn in surv_funcs])
                            ibs_model = integrated_brier_score(y_train_full, y_test, model_surv_probs, times)
                            
                            km_time, km_surv = kaplan_meier_estimator(y_train_full['event'], y_train_full['time'])
                            km_surv_at_times = np.interp(times, km_time, km_surv, left=1.0, right=km_surv[-1])
                            km_surv_probs = np.tile(km_surv_at_times, (len(y_test), 1))
                            ibs_null = integrated_brier_score(y_train_full, y_test, km_surv_probs, times)
                            
                            result_row[metric] = 1.0 - (ibs_model / ibs_null) if ibs_null > 0 else np.nan
                        else:
                            result_row[metric] = np.nan
                            error_occurred = True
                    else:
                        result_row[metric] = np.nan
                        error_occurred = True
                except Exception as e:
                    if verbose: print(f"    Warning: R²_IBS failed: {str(e)}")
                    result_row[metric] = np.nan
                    error_occurred = True
                    
            elif metric.startswith('r2-brier-'):
                try:
                    time_key = metric.replace('r2-brier-', '')
                    if surv_funcs is not None and time_key in TIME_POINTS:
                        time_point = TIME_POINTS[time_key]
                        max_time = min(y_train_full['time'].max(), y_test['time'].max())
                        
                        if time_point <= max_time:
                            model_surv_probs = np.asarray([fn(time_point) for fn in surv_funcs])
                            bs_model = float(brier_score(y_train_full, y_test, model_surv_probs, time_point)[1][0])

                            # KM null-model Brier score for the R^2 denominator
                            km_time, km_surv = kaplan_meier_estimator(y_train_full['event'], y_train_full['time'])
                            km_surv_at_time = np.interp(time_point, km_time, km_surv, left=1.0, right=km_surv[-1])
                            km_surv_probs = np.full(len(y_test), km_surv_at_time)
                            bs_null = float(brier_score(y_train_full, y_test, km_surv_probs, time_point)[1][0])

                            result_row[metric] = 1.0 - (bs_model / bs_null) if bs_null > 0 else np.nan
                        else:
                            result_row[metric] = np.nan
                            error_occurred = True
                    else:
                        result_row[metric] = np.nan
                        error_occurred = True
                except Exception as e:
                    if verbose: print(f"    Warning: {metric} failed: {str(e)}")
                    result_row[metric] = np.nan
                    error_occurred = True
                    
            elif metric == 'cumulative-dynamic-auc':
                try:
                    max_time = min(y_train['time'].max(), y_test['time'].max()) * 0.95
                    times = np.percentile(y_train['time'], np.linspace(10, 90, 10))
                    times = np.unique(times[(times > 0) & (times <= max_time)])
                    
                    if len(times) >= 2 and pred is not None:
                        _, mean_auc = cumulative_dynamic_auc(y_train, y_test, pred, times)
                        result_row[metric] = mean_auc
                    else:
                        result_row[metric] = np.nan
                        error_occurred = True
                except Exception as e:
                    if verbose: print(f"    Warning: Cumulative Dynamic AUC failed: {str(e)}")
                    result_row[metric] = np.nan
                    error_occurred = True
                    
            elif metric.startswith('auc-'):
                try:
                    time_key = metric.replace('auc-', '')
                    if pred is not None and time_key in TIME_POINTS:
                        time_point = TIME_POINTS[time_key]
                        max_time = min(y_train['time'].max(), y_test['time'].max())
                        
                        if time_point <= max_time:
                            auc_values, _ = cumulative_dynamic_auc(y_train, y_test, pred, [time_point])
                            result_row[metric] = auc_values[0]
                        else:
                            result_row[metric] = np.nan
                            error_occurred = True
                    else:
                        result_row[metric] = np.nan
                        error_occurred = True
                except Exception as e:
                    if verbose: print(f"    Warning: {metric} failed: {str(e)}")
                    result_row[metric] = np.nan
                    error_occurred = True
                    
            elif metric == 'cumulative-dynamic-prauc':
                try:
                    max_time = min(y_train['time'].max(), y_test['time'].max()) * 0.95
                    times = np.percentile(y_train['time'], np.linspace(10, 90, 10))
                    times = np.unique(times[(times > 0) & (times <= max_time)])
                    
                    if len(times) >= 2 and pred is not None:
                        prauc_scores = []
                        for t in times:
                            valid_mask, y_binary, weights = get_ipcw_weights_for_prauc(y_train, y_test, t)
                            if valid_mask.sum() > 0 and y_binary[valid_mask].sum() > 0:
                                precision_vals, recall_vals, _ = precision_recall_curve(
                                    y_binary[valid_mask], pred[valid_mask], 
                                    sample_weight=weights[valid_mask]
                                )
                                prauc_scores.append(sklearn_auc(recall_vals, precision_vals))
                        
                        if len(prauc_scores) > 0:
                            result_row[metric] = np.mean(prauc_scores)
                        else:
                            result_row[metric] = np.nan
                            error_occurred = True
                    else:
                        result_row[metric] = np.nan
                        error_occurred = True
                except Exception as e:
                    if verbose: print(f"    Warning: Cumulative Dynamic PR-AUC failed: {str(e)}")
                    result_row[metric] = np.nan
                    error_occurred = True
                    
            elif metric.startswith('prauc-'):
                try:
                    time_key = metric.replace('prauc-', '')
                    if pred is not None and time_key in TIME_POINTS:
                        time_point = TIME_POINTS[time_key]
                        max_time = min(y_train['time'].max(), y_test['time'].max())
                        
                        if time_point <= max_time:
                            valid_mask, y_binary, weights = get_ipcw_weights_for_prauc(y_train, y_test, time_point)
                            if valid_mask.sum() > 0 and y_binary[valid_mask].sum() > 0:
                                precision_vals, recall_vals, _ = precision_recall_curve(
                                    y_binary[valid_mask], pred[valid_mask], 
                                    sample_weight=weights[valid_mask]
                                )
                                result_row[metric] = sklearn_auc(recall_vals, precision_vals)
                            else:
                                result_row[metric] = np.nan
                                error_occurred = True
                        else:
                            result_row[metric] = np.nan
                            error_occurred = True
                    else:
                        result_row[metric] = np.nan
                        error_occurred = True
                except Exception as e:
                    if verbose: print(f"    Warning: {metric} calculation failed: {str(e)}")
                    result_row[metric] = np.nan
                    error_occurred = True
            
            elif metric == 'partial-likelihood':
                # Cox partial log-likelihood (higher is better)
                try:
                    if pred is not None:
                        sort_idx = np.argsort(y_test['time'])
                        sorted_time = y_test['time'][sort_idx]
                        sorted_event = y_test['event'][sort_idx]
                        sorted_risk = pred[sort_idx]
                        
                        log_likelihood = 0.0
                        for i in range(len(sorted_time)):
                            if sorted_event[i]:
                                risk_set_mask = sorted_time >= sorted_time[i]
                                risk_set_scores = sorted_risk[risk_set_mask]
                                log_likelihood += sorted_risk[i] - np.log(np.sum(np.exp(risk_set_scores)))
                        
                        result_row[metric] = log_likelihood
                    else:
                        result_row[metric] = np.nan
                        error_occurred = True
                except Exception as e:
                    if verbose: print(f"    Warning: Partial likelihood failed: {str(e)}")
                    result_row[metric] = np.nan
                    error_occurred = True
            
            elif metric == 'log-loss':
                # Integrated log loss across multiple time points
                try:
                    if surv_funcs is not None:
                        max_time = min(y_train['time'].max(), y_test['time'].max()) * 0.95
                        times = np.percentile(y_train['time'], np.linspace(10, 90, 10))
                        times = np.unique(times[(times > 0) & (times <= max_time)])
                        
                        if len(times) >= 2:
                            log_losses = []
                            for t in times:
                                y_binary_ll = ((y_test['time'] <= t) & y_test['event']).astype(int)
                                pred_proba = 1.0 - np.array([fn(t) for fn in surv_funcs])
                                pred_proba = np.clip(pred_proba, 1e-15, 1 - 1e-15)
                                
                                if len(np.unique(y_binary_ll)) >= 2:
                                    log_losses.append(log_loss(y_binary_ll, pred_proba))
                            
                            if len(log_losses) > 0:
                                result_row[metric] = np.mean(log_losses)
                            else:
                                result_row[metric] = np.nan
                                error_occurred = True
                        else:
                            result_row[metric] = np.nan
                            error_occurred = True
                    else:
                        result_row[metric] = np.nan
                        error_occurred = True
                except Exception as e:
                    if verbose: print(f"    Warning: Log loss failed: {str(e)}")
                    result_row[metric] = np.nan
                    error_occurred = True
            
            elif metric.startswith('log-loss-'):
                # Log loss at specific time points
                try:
                    time_key = metric.replace('log-loss-', '')
                    if surv_funcs is not None and time_key in TIME_POINTS:
                        time_point = TIME_POINTS[time_key]
                        max_time = min(y_train['time'].max(), y_test['time'].max())
                        
                        if time_point <= max_time:
                            y_binary_ll = ((y_test['time'] <= time_point) & y_test['event']).astype(int)
                            pred_proba = 1.0 - np.array([fn(time_point) for fn in surv_funcs])
                            pred_proba = np.clip(pred_proba, 1e-15, 1 - 1e-15)
                            
                            if len(np.unique(y_binary_ll)) >= 2:
                                result_row[metric] = log_loss(y_binary_ll, pred_proba)
                            else:
                                result_row[metric] = np.nan
                                error_occurred = True
                        else:
                            result_row[metric] = np.nan
                            error_occurred = True
                    else:
                        result_row[metric] = np.nan
                        error_occurred = True
                except Exception as e:
                    if verbose: print(f"    Warning: {metric} failed: {str(e)}")
                    result_row[metric] = np.nan
                    error_occurred = True
            
            error_flags[metric].append(error_occurred)
        
        result_rows.append(result_row)
    
    results_df = pd.DataFrame(result_rows)
    return results_df, error_flags                                       


def _build_scaler_template_for_ensemble(scale_data, scaler_type, continuous_features, X_ref):
    """Build a reusable scaler template for ensemble train/validation evaluation."""
    if not scale_data:
        return None

    if continuous_features is not None:
        valid_continuous = [col for col in continuous_features if col in X_ref.columns]
        if len(valid_continuous) > 0:
            scaler_cls = StandardScaler if scaler_type == 'standard' else MinMaxScaler
            return ColumnTransformer(
                [('continuous', scaler_cls(), valid_continuous)],
                remainder='passthrough'
            )

    return StandardScaler() if scaler_type == 'standard' else MinMaxScaler()


def _evaluate_ensemble_train_validation(model_class, X_train, y_train, X_val, y_val,
                                        model_params, metrics,
                                        scaler_template=None,
                                        random_state=None, n_jobs=None,
                                        y_train_full=None):
    """Fit an ensemble survival model once and evaluate requested metrics."""

    # Preserve censoring distribution for KM/IPCW-based null models if provided.
    y_for_km = y_train_full if y_train_full is not None else y_train

    TIME_POINTS = {
        '6m': 0.5 * 365,
        '1y': 1 * 365,
        '2y': 2 * 365,
        '3y': 3 * 365,
        '4y': 4 * 365,
        '5y': 5 * 365,
        '10y': 10 * 365,
        '14y': 14 * 365
    }

    results = {metric: np.nan for metric in metrics}

    X_train_eval = X_train.copy() if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train)
    X_val_eval = X_val.copy() if isinstance(X_val, pd.DataFrame) else pd.DataFrame(X_val)

    if scaler_template is not None:
        try:
            scaler = clone(scaler_template)
            X_train_eval = scaler.fit_transform(X_train_eval)
            X_val_eval = scaler.transform(X_val_eval)
        except Exception:
            return results

    model_params_copy = model_params.copy()
    if random_state is not None:
        model_params_copy['random_state'] = random_state
    if n_jobs is not None:
        model_params_copy['n_jobs'] = n_jobs

    try:
        model = model_class(**model_params_copy)
    except TypeError as e:
        if 'unexpected keyword argument' not in str(e):
            return results

        model_params_copy = model_params.copy()
        if random_state is not None:
            try:
                model = model_class(**model_params_copy, random_state=random_state)
            except TypeError:
                model = model_class(**model_params_copy)
        else:
            model = model_class(**model_params_copy)
    except Exception:
        return results

    try:
        model.fit(X_train_eval, y_train)
    except Exception:
        return results

    pred = None
    surv_funcs = None
    try:
        pred = model.predict(X_val_eval)
    except Exception:
        pass

    if hasattr(model, 'predict_survival_function'):
        try:
            surv_funcs = model.predict_survival_function(X_val_eval)
        except Exception:
            pass

    for metric in metrics:
        if metric == 'c-index':
            try:
                results[metric] = concordance_index_censored(y_val['event'], y_val['time'], pred)[0]
            except Exception:
                results[metric] = np.nan

        elif metric == 'c-index-ipcw':
            try:
                tau = min(y_train['time'].max(), y_val['time'].max()) * 0.95
                results[metric] = concordance_index_ipcw(y_train, y_val, pred, tau=tau)[0]
            except Exception:
                results[metric] = np.nan

        elif metric == 'int-brier-score':
            try:
                if surv_funcs is not None:
                    max_time = min(y_train['time'].max(), y_val['time'].max()) * 0.95
                    times = np.percentile(y_train['time'], np.linspace(5, 90, 10))
                    times = np.unique(times[(times > 0) & (times <= max_time)])
                    if len(times) > 1:
                        probs = np.asarray([[fn(t) for t in times] for fn in surv_funcs])
                        results[metric] = integrated_brier_score(y_train, y_val, probs, times)
            except Exception:
                results[metric] = np.nan

        elif metric.startswith('brier-'):
            try:
                time_key = metric.replace('brier-', '')
                if surv_funcs is not None and time_key in TIME_POINTS:
                    time_point = TIME_POINTS[time_key]
                    max_time = min(y_train['time'].max(), y_val['time'].max())
                    if time_point <= max_time:
                        probs = np.asarray([fn(time_point) for fn in surv_funcs])
                        results[metric] = float(brier_score(y_train, y_val, probs, time_point)[1][0])
            except Exception:
                results[metric] = np.nan

        elif metric == 'r2-ibs':
            try:
                if surv_funcs is not None:
                    max_time = min(y_for_km['time'].max(), y_val['time'].max()) * 0.95
                    times = np.percentile(y_for_km['time'], np.linspace(5, 90, 10))
                    times = np.unique(times[(times > 0) & (times <= max_time)])
                    if len(times) > 1:
                        model_surv_probs = np.asarray([[fn(t) for t in times] for fn in surv_funcs])
                        ibs_model = integrated_brier_score(y_for_km, y_val, model_surv_probs, times)

                        km_time, km_surv = kaplan_meier_estimator(y_for_km['event'], y_for_km['time'])
                        km_surv_at_times = np.interp(times, km_time, km_surv, left=1.0, right=km_surv[-1])
                        km_surv_probs = np.tile(km_surv_at_times, (len(y_val), 1))
                        ibs_null = integrated_brier_score(y_for_km, y_val, km_surv_probs, times)

                        if ibs_null > 0:
                            results[metric] = 1.0 - (ibs_model / ibs_null)
            except Exception:
                results[metric] = np.nan

        elif metric.startswith('r2-brier-'):
            try:
                time_key = metric.replace('r2-brier-', '')
                if surv_funcs is not None and time_key in TIME_POINTS:
                    time_point = TIME_POINTS[time_key]
                    max_time = min(y_for_km['time'].max(), y_val['time'].max())
                    if time_point <= max_time:
                        model_surv_probs = np.asarray([fn(time_point) for fn in surv_funcs])
                        bs_model = float(brier_score(y_for_km, y_val, model_surv_probs, time_point)[1][0])

                        km_time, km_surv = kaplan_meier_estimator(y_for_km['event'], y_for_km['time'])
                        km_surv_at_time = np.interp(time_point, km_time, km_surv, left=1.0, right=km_surv[-1])
                        km_surv_probs = np.full(len(y_val), km_surv_at_time)
                        bs_null = float(brier_score(y_for_km, y_val, km_surv_probs, time_point)[1][0])
                        if bs_null > 0:
                            results[metric] = 1.0 - (bs_model / bs_null)
            except Exception:
                results[metric] = np.nan

        elif metric == 'cumulative-dynamic-auc':
            try:
                if pred is not None:
                    max_time = min(y_train['time'].max(), y_val['time'].max()) * 0.95
                    times = np.percentile(y_train['time'], np.linspace(10, 90, 10))
                    times = np.unique(times[(times > 0) & (times <= max_time)])
                    if len(times) >= 2:
                        _, mean_auc = cumulative_dynamic_auc(y_train, y_val, pred, times)
                        results[metric] = mean_auc
            except Exception:
                results[metric] = np.nan

        elif metric.startswith('auc-'):
            try:
                time_key = metric.replace('auc-', '')
                if pred is not None and time_key in TIME_POINTS:
                    time_point = TIME_POINTS[time_key]
                    max_time = min(y_train['time'].max(), y_val['time'].max())
                    if time_point <= max_time:
                        auc_values, _ = cumulative_dynamic_auc(y_train, y_val, pred, [time_point])
                        results[metric] = auc_values[0]
            except Exception:
                results[metric] = np.nan

        elif metric == 'cumulative-dynamic-prauc':
            try:
                if pred is not None:
                    max_time = min(y_train['time'].max(), y_val['time'].max()) * 0.95
                    times = np.percentile(y_train['time'], np.linspace(10, 90, 10))
                    times = np.unique(times[(times > 0) & (times <= max_time)])
                    if len(times) >= 2:
                        prauc_scores = []
                        for t in times:
                            valid_mask, y_binary, weights = get_ipcw_weights_for_prauc(y_train, y_val, t)
                            if valid_mask.sum() > 0 and y_binary[valid_mask].sum() > 0:
                                precision_vals, recall_vals, _ = precision_recall_curve(
                                    y_binary[valid_mask], pred[valid_mask], sample_weight=weights[valid_mask]
                                )
                                prauc_scores.append(sklearn_auc(recall_vals, precision_vals))
                        if len(prauc_scores) > 0:
                            results[metric] = np.mean(prauc_scores)
            except Exception:
                results[metric] = np.nan

        elif metric.startswith('prauc-'):
            try:
                time_key = metric.replace('prauc-', '')
                if pred is not None and time_key in TIME_POINTS:
                    time_point = TIME_POINTS[time_key]
                    max_time = min(y_train['time'].max(), y_val['time'].max())
                    if time_point <= max_time:
                        valid_mask, y_binary, weights = get_ipcw_weights_for_prauc(y_train, y_val, time_point)
                        if valid_mask.sum() > 0 and y_binary[valid_mask].sum() > 0:
                            precision_vals, recall_vals, _ = precision_recall_curve(
                                y_binary[valid_mask], pred[valid_mask], sample_weight=weights[valid_mask]
                            )
                            results[metric] = sklearn_auc(recall_vals, precision_vals)
            except Exception:
                results[metric] = np.nan

        elif metric == 'partial-likelihood':
            try:
                if pred is not None:
                    sort_idx = np.argsort(y_val['time'])
                    sorted_time = y_val['time'][sort_idx]
                    sorted_event = y_val['event'][sort_idx]
                    sorted_risk = pred[sort_idx]

                    log_likelihood = 0.0
                    for i in range(len(sorted_time)):
                        if sorted_event[i]:
                            risk_set_mask = sorted_time >= sorted_time[i]
                            risk_set_scores = sorted_risk[risk_set_mask]
                            log_likelihood += sorted_risk[i] - np.log(np.sum(np.exp(risk_set_scores)))

                    results[metric] = log_likelihood
            except Exception:
                results[metric] = np.nan

        elif metric == 'log-loss':
            try:
                if surv_funcs is not None:
                    max_time = min(y_train['time'].max(), y_val['time'].max()) * 0.95
                    times = np.percentile(y_train['time'], np.linspace(10, 90, 10))
                    times = np.unique(times[(times > 0) & (times <= max_time)])
                    if len(times) >= 2:
                        log_losses = []
                        for t in times:
                            y_binary_ll = ((y_val['time'] <= t) & y_val['event']).astype(int)
                            pred_proba = 1.0 - np.array([fn(t) for fn in surv_funcs])
                            pred_proba = np.clip(pred_proba, 1e-15, 1 - 1e-15)
                            if len(np.unique(y_binary_ll)) >= 2:
                                log_losses.append(log_loss(y_binary_ll, pred_proba))
                        if len(log_losses) > 0:
                            results[metric] = np.mean(log_losses)
            except Exception:
                results[metric] = np.nan

        elif metric.startswith('log-loss-'):
            try:
                time_key = metric.replace('log-loss-', '')
                if surv_funcs is not None and time_key in TIME_POINTS:
                    time_point = TIME_POINTS[time_key]
                    max_time = min(y_train['time'].max(), y_val['time'].max())
                    if time_point <= max_time:
                        y_binary_ll = ((y_val['time'] <= time_point) & y_val['event']).astype(int)
                        pred_proba = 1.0 - np.array([fn(time_point) for fn in surv_funcs])
                        pred_proba = np.clip(pred_proba, 1e-15, 1 - 1e-15)
                        if len(np.unique(y_binary_ll)) >= 2:
                            results[metric] = log_loss(y_binary_ll, pred_proba)
            except Exception:
                results[metric] = np.nan

    return results


def ensemble_bootstrap_optimism_grid_search(model_class, dataset, X, y, param_grid,
                                            controls_per_case_range=('Full',),
                                            feature_columns=None,
                                            event_col='Regional LN recurrence',
                                            time_col='Survival days',
                                            scale_data=False, scaler_type='standard',
                                            continuous_features=None,
                                            random_state=None, n_jobs=None,
                                            n_bootstraps=200, base_random_state=42,
                                            verbose=True, metrics=None,
                                            return_bootstrap_df=False):
    """
    Bootstrap optimism-corrected grid search for ensemble survival models.

    Supports RSF/GBM/SSVM (or any model class with fit/predict used by
    ensemble_repeated_grid_search), with optional case-control subsampling.
    """

    start_time = time.time()

    if feature_columns is None:
        feature_columns = X.columns.tolist()

    metrics_to_run = ['c-index']
    if metrics is not None:
        for metric_name in metrics:
            if metric_name not in metrics_to_run:
                metrics_to_run.append(metric_name)

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    total_combinations = len(param_combinations) * len(controls_per_case_range)

    def _metric_suffix(metric_name):
        return metric_name.replace('-', '_')

    event_mask = dataset[event_col].values.astype(int) == 1
    event_indices = np.where(event_mask)[0]
    non_event_indices = np.where(~event_mask)[0]

    if len(event_indices) == 0 or len(non_event_indices) == 0:
        raise ValueError("Stratified bootstrap requires both event and non-event samples.")

    if verbose:
        print(f"{'='*80}")
        print("ENSEMBLE STRATIFIED BOOTSTRAP OPTIMISM GRID SEARCH")
        print(f"{'='*80}")
        print(f"Model: {model_class.__name__}")
        print(f"Controls per case: {list(controls_per_case_range)}")
        print(f"Total hyperparameter combinations: {total_combinations}")
        print(f"Bootstrap runs per combination: {n_bootstraps}")
        print(f"Total model loops (B/C runs): {total_combinations * n_bootstraps}")
        print(f"Scaling: {'Yes (' + scaler_type + ')' if scale_data else 'No'}")
        print(f"Metrics: {metrics_to_run}")
        print(f"{'='*80}\n")

    all_results = []
    summary_rows = []
    current_iteration = 0

    for n_controls in controls_per_case_range:
        for param_values_tuple in param_combinations:
            current_iteration += 1
            params = dict(zip(param_names, param_values_tuple))
            combo_seed = base_random_state + current_iteration * 10000
            rng = np.random.RandomState(combo_seed)

            if verbose:
                elapsed = time.time() - start_time
                avg_time = elapsed / current_iteration if current_iteration > 0 else 0
                remaining = avg_time * (total_combinations - current_iteration)
                controls_str = 'Full (no subsampling)' if n_controls == 'Full' else str(n_controls)
                print(
                    f"[{current_iteration}/{total_combinations}] "
                    f"controls={controls_str}, params={params} "
                    f"(elapsed: {elapsed/60:.1f}min, est. remaining: {remaining/60:.1f}min)"
                )

            scaler_template = _build_scaler_template_for_ensemble(
                scale_data=scale_data,
                scaler_type=scaler_type,
                continuous_features=continuous_features,
                X_ref=X
            )

            try:
                if n_controls == 'Full':
                    X_app_train = X.copy()
                    y_app_train = y.copy()
                    y_app_full = y_app_train
                else:
                    X_app_train, y_app_train, _ = sample_case_control(
                        train_fold=dataset.reset_index(drop=True),
                        event_col=event_col,
                        time_col=time_col,
                        n_controls_per_case=n_controls,
                        feature_columns=feature_columns,
                        random_state=combo_seed
                    )
                    y_app_full = y

                app_res = _evaluate_ensemble_train_validation(
                    model_class=model_class,
                    X_train=X_app_train,
                    y_train=y_app_train,
                    X_val=X_app_train,
                    y_val=y_app_train,
                    model_params=params,
                    metrics=metrics_to_run,
                    scaler_template=scaler_template,
                    random_state=random_state,
                    n_jobs=n_jobs,
                    y_train_full=y_app_full
                )
                c_apparent = app_res.get('c-index', np.nan)
            except Exception as e:
                if verbose:
                    print(f"  Warning: Apparent step failed: {type(e).__name__}: {str(e)}")
                app_res = {metric: np.nan for metric in metrics_to_run}
                c_apparent = np.nan

            optimism_values = []
            c_boot_values = []
            c_test_values = []

            for boot_idx in range(n_bootstraps):
                try:
                    boot_event_idx = rng.choice(event_indices, size=len(event_indices), replace=True)
                    boot_non_event_idx = rng.choice(non_event_indices, size=len(non_event_indices), replace=True)
                    boot_indices = np.concatenate([boot_event_idx, boot_non_event_idx])
                    rng.shuffle(boot_indices)

                    boot_df = dataset.iloc[boot_indices].copy().reset_index(drop=True)
                    X_boot_full = boot_df[feature_columns]
                    y_boot_full = Surv.from_arrays(event=boot_df[event_col], time=boot_df[time_col])

                    if n_controls == 'Full':
                        X_boot_train = X_boot_full
                        y_boot_train = y_boot_full
                    else:
                        X_boot_train, y_boot_train, _ = sample_case_control(
                            train_fold=boot_df,
                            event_col=event_col,
                            time_col=time_col,
                            n_controls_per_case=n_controls,
                            feature_columns=feature_columns,
                            random_state=combo_seed + boot_idx
                        )

                    boot_res = _evaluate_ensemble_train_validation(
                        model_class=model_class,
                        X_train=X_boot_train,
                        y_train=y_boot_train,
                        X_val=X_boot_train,
                        y_val=y_boot_train,
                        model_params=params,
                        metrics=metrics_to_run,
                        scaler_template=scaler_template,
                        random_state=random_state,
                        n_jobs=n_jobs,
                        y_train_full=y_boot_full
                    )
                    c_boot = boot_res.get('c-index', np.nan)

                    test_res = _evaluate_ensemble_train_validation(
                        model_class=model_class,
                        X_train=X_boot_train,
                        y_train=y_boot_train,
                        X_val=X,
                        y_val=y,
                        model_params=params,
                        metrics=metrics_to_run,
                        scaler_template=scaler_template,
                        random_state=random_state,
                        n_jobs=n_jobs,
                        y_train_full=y_boot_full
                    )
                    c_test = test_res.get('c-index', np.nan)
                    optimism = c_boot - c_test

                except Exception as e:
                    if verbose:
                        print(
                            f"  Warning: Bootstrap {boot_idx+1}/{n_bootstraps} failed: "
                            f"{type(e).__name__}: {str(e)}"
                        )
                    boot_res = {metric: np.nan for metric in metrics_to_run}
                    test_res = {metric: np.nan for metric in metrics_to_run}
                    c_boot = np.nan
                    c_test = np.nan
                    optimism = np.nan

                c_boot_values.append(c_boot)
                c_test_values.append(c_test)
                optimism_values.append(optimism)

                bootstrap_row = {
                    'n_controls': n_controls,
                    'bootstrap_idx': boot_idx,
                    'bootstrap_seed': combo_seed + boot_idx,
                    'c_apparent': c_apparent,
                    'c_boot': c_boot,
                    'c_test': c_test,
                    'optimism': optimism
                }
                for param_name, param_value in params.items():
                    bootstrap_row[param_name] = param_value

                for metric_name in metrics_to_run:
                    metric_key = _metric_suffix(metric_name)
                    bootstrap_row[f'app_{metric_key}'] = app_res.get(metric_name, np.nan)
                    bootstrap_row[f'boot_{metric_key}'] = boot_res.get(metric_name, np.nan)
                    bootstrap_row[f'test_{metric_key}'] = test_res.get(metric_name, np.nan)
                    boot_metric_value = bootstrap_row[f'boot_{metric_key}']
                    test_metric_value = bootstrap_row[f'test_{metric_key}']
                    bootstrap_row[f'optimism_{metric_key}'] = (
                        boot_metric_value - test_metric_value
                        if not (np.isnan(boot_metric_value) or np.isnan(test_metric_value))
                        else np.nan
                    )

                all_results.append(bootstrap_row)

            avg_optimism = float(np.nanmean(optimism_values)) if len(optimism_values) > 0 else np.nan
            c_final = c_apparent - avg_optimism if not np.isnan(c_apparent) and not np.isnan(avg_optimism) else np.nan

            summary_row = {
                'n_controls': n_controls,
                'n_bootstraps': n_bootstraps,
                'c_apparent': c_apparent,
                'avg_optimism': avg_optimism,
                'c_final': c_final,
                'c_boot_mean': float(np.nanmean(c_boot_values)) if len(c_boot_values) > 0 else np.nan,
                'c_boot_std': float(np.nanstd(c_boot_values)) if len(c_boot_values) > 0 else np.nan,
                'c_test_mean': float(np.nanmean(c_test_values)) if len(c_test_values) > 0 else np.nan,
                'c_test_std': float(np.nanstd(c_test_values)) if len(c_test_values) > 0 else np.nan,
            }
            for param_name, param_value in params.items():
                summary_row[param_name] = param_value
            summary_rows.append(summary_row)

    results_df = pd.DataFrame(all_results)
    summary_df = pd.DataFrame(summary_rows)

    valid_summary = summary_df[~summary_df['c_final'].isna()].copy()
    if len(valid_summary) == 0:
        best_params = {'n_controls': np.nan, 'c_apparent': np.nan, 'avg_optimism': np.nan, 'c_final': np.nan}
        for param_name in param_names:
            best_params[param_name] = np.nan
    else:
        best_idx = valid_summary['c_final'].idxmax()
        best_row = summary_df.loc[best_idx]
        best_params = {
            'n_controls': best_row['n_controls'],
            'c_apparent': best_row['c_apparent'],
            'avg_optimism': best_row['avg_optimism'],
            'c_final': best_row['c_final'],
        }
        for param_name in param_names:
            best_params[param_name] = best_row[param_name]

    if verbose:
        elapsed_time = time.time() - start_time
        print(f"\n{'='*80}")
        print("ENSEMBLE BOOTSTRAP OPTIMISM GRID SEARCH COMPLETE")
        print(f"{'='*80}")
        print(f"Total time: {elapsed_time/60:.1f} minutes")
        if not np.isnan(best_params['c_final']):
            n_controls_str = 'Full (no subsampling)' if best_params['n_controls'] == 'Full' else str(best_params['n_controls'])
            print("\nBest parameters by optimism-corrected c-index:")
            print(f"  n_controls_per_case: {n_controls_str}")
            for param_name in param_names:
                print(f"  {param_name}: {best_params[param_name]}")
            print(f"  C_apparent: {best_params['c_apparent']:.4f}")
            print(f"  Avg optimism: {best_params['avg_optimism']:.4f}")
            print(f"  C_final: {best_params['c_final']:.4f}")
        print(f"\n{'='*80}\n")

    if return_bootstrap_df:
        return results_df, summary_df, best_params, results_df.copy()

    return results_df, summary_df, best_params

def ensemble_repeated_grid_search(model_class, X, y, param_grid, metrics,
                                  model_name='Model',
                                  scale_data=False, scaler_type='standard',
                                  continuous_features=None,
                                  random_state=None, n_jobs=None,
                                  n_controls_per_case=None, dataset=None,
                                  feature_columns=None,
                                  event_col='Regional LN recurrence',
                                  time_col='Survival days',
                                  n_iter_no_change=None, tol=1e-4,
                                  n_splits=5, n_repeats=10, base_random_state=42,
                                  verbose=True):
    """
    Grid search over param_grid for an ensemble model class (RSF, GBM, SSVM, ...), using
    the same repeated CV splits for every combination.

    n_controls_per_case works like elsewhere: an int subsamples, 'Full' or None trains on
    the whole fold (dataset/feature_columns only needed for the subsampled case).
    n_iter_no_change enables GBM early stopping against the CV validation fold itself
    (no extra held-out split needed) -- ignored for RSF, which has no boosting rounds.

    Returns results_df (per fold), summary_df (aggregated), best_params (per metric).
    """
    start_time = time.time()

    all_splits = generate_cv_splits(X, y, n_splits, n_repeats, base_random_state)

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    total_combinations = len(param_combinations)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"{model_name.upper()} REPEATED GRID SEARCH")
        print(f"{'='*80}")
        print(f"Model: {model_class.__name__}")
        print(f"Scaling: {'Yes (' + scaler_type + ')' if scale_data else 'No'}")
        print(f"Subsampling: {'Yes (' + str(n_controls_per_case) + ' controls per case)' if n_controls_per_case else 'No'}")
        print(f"Metrics: {metrics}")
        print(f"Parameter Grid:")
        for param, values in param_grid.items():
            print(f"  - {param}: {values}")
        print(f"Total hyperparameter combinations: {total_combinations}")
        print(f"Folds per repetition: {n_splits}")
        print(f"Repetitions: {n_repeats}")
        print(f"Total evaluations: {total_combinations * n_splits * n_repeats}")
        if n_iter_no_change is not None:
            print(f"Early stopping: n_iter_no_change={n_iter_no_change}, tol={tol}")
            print(f"  - Uses CV validation fold (no training data lost)")
        print(f"{'='*80}\n")
    
    all_results = []
    interrupted_at = None

    try:
        for comb_idx, param_values_tuple in enumerate(param_combinations):
            params = dict(zip(param_names, param_values_tuple))

            # n_controls_per_case can be swept in the grid itself, or fixed via the outer arg
            current_n_controls = params.pop('n_controls_per_case', n_controls_per_case)

            if verbose:
                elapsed = time.time() - start_time
                if current_n_controls is not None and current_n_controls != 'Full':
                    print(f"[{comb_idx + 1}/{total_combinations}] Testing params: {params}, n_controls_per_case={current_n_controls}")
                elif current_n_controls == 'Full':
                    print(f"[{comb_idx + 1}/{total_combinations}] Testing params: {params}, n_controls_per_case=Full (no subsampling)")
                else:
                    print(f"[{comb_idx + 1}/{total_combinations}] Testing params: {params}")
                if comb_idx > 0:
                    avg_time = elapsed / comb_idx
                    remaining = avg_time * (total_combinations - comb_idx)
                    print(f"  Elapsed: {elapsed/60:.1f}min, Est. remaining: {remaining/60:.1f}min")

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
                    n_controls_per_case=current_n_controls,
                    dataset=dataset,
                    feature_columns=feature_columns,
                    event_col=event_col,
                    time_col=time_col,
                    n_iter_no_change=n_iter_no_change,
                    tol=tol,
                    n_splits=n_splits,
                    n_repeats=n_repeats,
                    base_random_state=base_random_state,
                    verbose=False
                )

                for idx, row in cv_results.iterrows():
                    result_row = params.copy()
                    if current_n_controls is not None:
                        result_row['n_controls_per_case'] = current_n_controls
                    result_row['repeat_idx'] = row['repeat_idx']
                    result_row['fold_idx'] = row['fold_idx']
                    for metric in metrics:
                        result_row[metric] = row[metric]
                    if 'n_estimators_trained' in row:
                        result_row['n_estimators_trained'] = row['n_estimators_trained']
                        result_row['n_estimators_max'] = row['n_estimators_max']
                        result_row['stopped_early'] = row['stopped_early']
                    all_results.append(result_row)

                if verbose:
                    for metric in metrics:
                        score = cv_results[metric].mean()
                        std = cv_results[metric].std()
                        print(f"  {metric}: {score:.4f} ± {std:.4f}")

                    # Show convergence info
                    if 'n_estimators_trained' in cv_results.columns:
                        trained = cv_results['n_estimators_trained']
                        if trained.notna().any():
                            max_iters = cv_results['n_estimators_max'].iloc[0]
                            stopped_early = cv_results['stopped_early']
                            n_converged = stopped_early.sum() if stopped_early.notna().any() else 0
                            avg_iters = trained.mean()
                            print(f"  Convergence: {n_converged}/{len(cv_results)} folds stopped early")
                            print(f"  Avg iterations: {avg_iters:.0f}/{max_iters}")

            except KeyboardInterrupt:
                raise
            except Exception as e:
                if verbose:
                    print(f"  Error: {str(e)}")

                # Store NaN for failed combinations
                for split in all_splits:
                    result_row = params.copy()
                    if current_n_controls is not None:
                        result_row['n_controls_per_case'] = current_n_controls
                    result_row['repeat_idx'] = split['repeat_idx']
                    result_row['fold_idx'] = split['fold_idx']
                    for metric in metrics:
                        result_row[metric] = np.nan
                    all_results.append(result_row)
    except KeyboardInterrupt:
        interrupted_at = comb_idx + 1
        if verbose:
            print(f"\nInterrupted during combination {interrupted_at}/{total_combinations} "
                  f"-- returning results for the {interrupted_at - 1} combination(s) that "
                  f"finished, plus any folds already run on the one in progress.")

    # Create detailed results DataFrame
    results_df = pd.DataFrame(all_results)

    if results_df.empty:
        if verbose:
            print("\nInterrupted before any fold finished -- nothing to summarize.")
        return results_df, pd.DataFrame(), {metric: None for metric in metrics}
    
    # dropna=False so a None n_controls_per_case (etc.) doesn't silently drop rows
    param_cols = [col for col in results_df.columns
                  if col not in ['repeat_idx', 'fold_idx'] + metrics]

    summary_rows = []
    for combo, group in results_df.groupby(param_cols, dropna=False):
        # groupby gives a bare value instead of a 1-tuple when there's only one param col
        row = dict(zip(param_cols, combo if isinstance(combo, tuple) else [combo]))

        for metric in metrics:
            row[f'{metric}_mean'] = group[metric].mean()
            row[f'{metric}_std'] = group[metric].std()

            for repeat_idx in range(n_repeats):
                repeat_mask = group['repeat_idx'] == repeat_idx
                row[f'{metric}_repeat_{repeat_idx+1}_mean'] = group[repeat_mask][metric].mean()

            for repeat_idx in range(n_repeats):
                for fold_idx in range(n_splits):
                    fold_repeat_mask = (group['repeat_idx'] == repeat_idx) & (group['fold_idx'] == fold_idx)
                    if fold_repeat_mask.any():
                        row[f'{metric}_repeat_{repeat_idx+1}_fold_{fold_idx+1}'] = group[fold_repeat_mask][metric].values[0]
                    else:
                        row[f'{metric}_repeat_{repeat_idx+1}_fold_{fold_idx+1}'] = np.nan
        
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)

    param_cols_for_best = [col for col in summary_df.columns
                           if not col.endswith('_mean') and not col.endswith('_std')
                           and not col.startswith(tuple(metrics))
                           and '_repeat_' not in col and '_fold_' not in col]

    best_params = {}
    for metric in metrics:
        mean_col = f'{metric}_mean'
        valid_results = summary_df[~summary_df[mean_col].isna()].copy()

        if len(valid_results) == 0:
            if verbose:
                print(f"\nWarning: No valid results for {metric}")
            best_params[metric] = None
            continue

        # lower is better for brier score
        if 'brier' in metric.lower():
            best_idx = valid_results[mean_col].idxmin()
        else:
            best_idx = valid_results[mean_col].idxmax()

        best_row = summary_df.loc[best_idx]

        best_param_dict = {col: best_row[col] for col in param_cols_for_best}
        
        best_params[metric] = {
            'params': best_param_dict,
            'mean': best_row[f'{metric}_mean'],
            'std': best_row[f'{metric}_std']
        }
    
    elapsed_time = time.time() - start_time
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"REPEATED GRID SEARCH {'INTERRUPTED' if interrupted_at is not None else 'COMPLETE'}")
        print(f"{'='*80}")
        print(f"Total time: {elapsed_time/60:.1f} minutes")
        if interrupted_at is not None:
            print(f"Stopped during combination {interrupted_at}/{total_combinations} -- "
                  f"best parameters below are based on partial results only.")
        print(f"\nBest parameters for each metric:")
        print(f"{'-'*80}")
        for metric, info in best_params.items():
            if info is not None:
                print(f"\n{metric}:")
                print(f"  Score: {info['mean']:.4f} ± {info['std']:.4f}")
                print(f"  Parameters:")
                for param, value in info['params'].items():
                    if param == 'n_controls_per_case' and value == 'Full':
                        print(f"    {param}: Full (no subsampling)")
                    else:
                        print(f"    {param}: {value}")
        print(f"\n{'='*80}\n")
    
    return results_df, summary_df, best_params


def ensemble_grid_search(model_class, X, y, param_grid, metrics,
                         model_name='Model',
                         scale_data=False, scaler_type='standard',
                         continuous_features=None,
                         random_state=None, n_jobs=None,
                         n_controls_per_case=None, dataset=None,
                         feature_columns=None,
                         event_col='Regional LN recurrence',
                         time_col='Survival days',
                         n_splits=5, cv_random_state=42,
                         verbose=True):
    """
    Same as ensemble_repeated_grid_search but a single CV pass (n_repeats=1) rather than
    repeated -- quicker, cruder, mostly useful for a first pass over a wide param_grid
    before committing to the expensive repeated version.
    """
    start_time = time.time()

    cv_splits = generate_cv_splits(X, y, n_splits=n_splits, n_repeats=1, base_random_state=cv_random_state)

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    total_combinations = len(param_combinations)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"{model_name.upper()} GRID SEARCH (Single CV)")
        print(f"{'='*80}")
        print(f"Model: {model_class.__name__}")
        print(f"Scaling: {'Yes (' + scaler_type + ')' if scale_data else 'No'}")
        print(f"Subsampling: {'Yes (' + str(n_controls_per_case) + ' controls per case)' if n_controls_per_case else 'No'}")
        print(f"Metrics: {metrics}")
        print(f"Parameter Grid:")
        for param, values in param_grid.items():
            print(f"  - {param}: {values}")
        print(f"Total hyperparameter combinations: {total_combinations}")
        print(f"Folds: {n_splits}")
        print(f"Total evaluations: {total_combinations * n_splits}")
        print(f"{'='*80}\n")
    
    all_results = []
    
    for comb_idx, param_values_tuple in enumerate(param_combinations):
        params = dict(zip(param_names, param_values_tuple))

        current_n_controls = params.pop('n_controls_per_case', n_controls_per_case)

        if verbose:
            elapsed = time.time() - start_time
            if current_n_controls is not None and current_n_controls != 'Full':
                print(f"[{comb_idx + 1}/{total_combinations}] Testing params: {params}, n_controls_per_case={current_n_controls}")
            elif current_n_controls == 'Full':
                print(f"[{comb_idx + 1}/{total_combinations}] Testing params: {params}, n_controls_per_case=Full (no subsampling)")
            else:
                print(f"[{comb_idx + 1}/{total_combinations}] Testing params: {params}")
            if comb_idx > 0:
                avg_time = elapsed / comb_idx
                remaining = avg_time * (total_combinations - comb_idx)
                print(f"  Elapsed: {elapsed/60:.1f}min, Est. remaining: {remaining/60:.1f}min")
        
        # Run single CV for this parameter combination
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
                n_controls_per_case=current_n_controls,
                dataset=dataset,
                feature_columns=feature_columns,
                event_col=event_col,
                time_col=time_col,
                n_splits=n_splits,
                n_repeats=1,  # Single CV, no repetition
                base_random_state=cv_random_state,
                verbose=False
            )
            
            # Add parameter columns to results
            for idx, row in cv_results.iterrows():
                result_row = params.copy()
                if current_n_controls is not None:
                    result_row['n_controls_per_case'] = current_n_controls
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
                print(f"  Error: {str(e)}")

            for split in cv_splits:
                result_row = params.copy()
                if current_n_controls is not None:
                    result_row['n_controls_per_case'] = current_n_controls
                result_row['fold_idx'] = split['fold_idx']
                for metric in metrics:
                    result_row[metric] = np.nan
                all_results.append(result_row)

    results_df = pd.DataFrame(all_results)

    param_cols = [col for col in results_df.columns
                  if col not in ['fold_idx'] + metrics]

    summary_rows = []
    for combo, group in results_df.groupby(param_cols, dropna=False):
        row = dict(zip(param_cols, combo if isinstance(combo, tuple) else [combo]))

        for metric in metrics:
            row[f'{metric}_mean'] = group[metric].mean()
            row[f'{metric}_std'] = group[metric].std()

            for fold_idx in range(n_splits):
                fold_mask = group['fold_idx'] == fold_idx
                if fold_mask.any():
                    row[f'{metric}_fold_{fold_idx+1}'] = group[fold_mask][metric].values[0]
                else:
                    row[f'{metric}_fold_{fold_idx+1}'] = np.nan
        
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)

    param_cols_for_best = [col for col in summary_df.columns
                           if not col.endswith('_mean') and not col.endswith('_std')
                           and not col.startswith(tuple(metrics))
                           and '_fold_' not in col]

    best_params = {}
    for metric in metrics:
        mean_col = f'{metric}_mean'
        valid_results = summary_df[~summary_df[mean_col].isna()].copy()

        if len(valid_results) == 0:
            if verbose:
                print(f"\nWarning: No valid results for {metric}")
            best_params[metric] = None
            continue

        if 'brier' in metric.lower():
            best_idx = valid_results[mean_col].idxmin()
        else:
            best_idx = valid_results[mean_col].idxmax()

        best_row = summary_df.loc[best_idx]

        best_param_dict = {col: best_row[col] for col in param_cols_for_best}
        
        best_params[metric] = {
            'params': best_param_dict,
            'mean': best_row[f'{metric}_mean'],
            'std': best_row[f'{metric}_std']
        }
    
    elapsed_time = time.time() - start_time
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"GRID SEARCH COMPLETE")
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
                    if param == 'n_controls_per_case' and value == 'Full':
                        print(f"    {param}: Full (no subsampling)")
                    else:
                        print(f"    {param}: {value}")
        print(f"\n{'='*80}\n")
    
    return results_df, summary_df, best_params


# =============================================================================
# BALANCED ENSEMBLE - CLASS-WEIGHTED TRAINING
# =============================================================================

def _compute_class_weights(y_train, event_col_name='event', weight_ratio=None):
    """Balances events vs non-events via sample weights. Default (weight_ratio=None) is
    inverse-frequency weighting so both classes carry equal total weight: w_event =
    n/(2*n_events), w_non_event = n/(2*n_non_events). Pass a number instead to fix the
    event weight directly (non-events stay at 1)."""
    events_mask = y_train[event_col_name].astype(bool)
    n = len(events_mask)
    n_events = events_mask.sum()
    n_non_events = n - n_events

    sample_weight = np.ones(n, dtype=float)

    if weight_ratio is not None:
        sample_weight[events_mask] = weight_ratio
    else:
        if n_events > 0 and n_non_events > 0:
            sample_weight[events_mask] = n / (2.0 * n_events)
            sample_weight[~events_mask] = n / (2.0 * n_non_events)
    
    return sample_weight


class _EarlyStoppingMonitor:
    """Early-stopping callback for GradientBoostingSurvivalAnalysis. Tracks c-index on
    the CV validation fold itself rather than sklearn's validation_fraction, which would
    carve a chunk out of the training fold -- this way no training data is sacrificed
    just to get an early-stopping signal."""
    
    def __init__(self, X_val, y_val, n_iter_no_change=10, tol=1e-4, verbose=False):
        self.X_val = X_val
        self.y_val = y_val
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.verbose = verbose
        self.best_score = -np.inf
        self.no_improve_count = 0
        self.best_iteration = 0
        self.stopped_early = False
    
    def __call__(self, i, est, locals_):
        """Called by sklearn after each boosting iteration; True stops training."""
        # give the model a few rounds before trusting its predictions
        if i < 15:
            return False

        try:
            pred = est.predict(self.X_val)
            score = concordance_index_censored(
                self.y_val['event'], self.y_val['time'], pred
            )[0]
        except Exception as e:
            # not ready yet (rare past iter 15, but don't want to crash training over it)
            return False

        improved = False
        if score > self.best_score + self.tol:
            self.best_score = score
            self.no_improve_count = 0
            self.best_iteration = i
            improved = True
        else:
            self.no_improve_count += 1

        if self.no_improve_count >= self.n_iter_no_change:
            self.stopped_early = True
            return True
        
        return False


def balanced_ensemble_repeated_cross_validation(model_class, X, y, model_params, metrics,
                                                weight_ratio=None,
                                                event_col_name='event',
                                                n_iter_no_change=None, tol=1e-4,
                                                scale_data=False, scaler_type='standard',
                                                continuous_features=None,
                                                random_state=None, n_jobs=None,
                                                n_splits=5, n_repeats=10, base_random_state=42,
                                                verbose=False):
    """
    Perform repeated stratified cross-validation for ensemble survival models
    using class-weighted training to handle class imbalance.
    
    For each CV fold, sample_weight is computed so that events are upweighted
    relative to non-events, and passed to model.fit(X, y, sample_weight=...).
    The model sees ALL training data every iteration but gives more importance 
    to events.
    
    Weight scheme:
    - weight_ratio=None (default): balanced inverse-frequency weighting
        w_event = n_total / (2 * n_events)
        w_non_event = n_total / (2 * n_non_events)
        → total weight of events = total weight of non-events
    - weight_ratio=K: events get weight K, non-events get weight 1
    
    Early stopping (GBM only):
    - Set n_iter_no_change to enable early stopping using the CV validation fold
    - No training data is lost (unlike validation_fraction)
    - Training stops if c-index on validation fold doesn't improve by tol 
      for n_iter_no_change consecutive boosting iterations
    
    Parameters:
    -----------
    model_class : class
        Model class (RandomSurvivalForest, GradientBoostingSurvivalAnalysis)
    X : DataFrame
        Feature matrix
    y : structured array
        Survival data with 'event' and 'time' fields
    model_params : dict
        Dictionary of model parameters
    metrics : list
        List of metrics to evaluate: 'c-index', 'c-index-ipcw', 'int-brier-score',
        'brier-Xy', 'cumulative-dynamic-auc'
    weight_ratio : float or None
        If None, uses balanced inverse-frequency weights.
        If a number, events get this weight and non-events get 1.
    event_col_name : str
        Name of the event field in the structured array y (default: 'event')
    n_iter_no_change : int or None
        If set, enables early stopping using the CV validation fold.
        Training stops if no improvement in c-index for this many consecutive
        boosting iterations. Only works with GBM, ignored for RSF.
    tol : float
        Minimum improvement in c-index to count as an improvement (default: 1e-4).
    scale_data : bool
        Whether to scale the data
    scaler_type : str
        'standard' or 'minmax'
    continuous_features : list or None
        List of continuous feature names to scale. If None and scale_data=True, all features are scaled.
    random_state : int
        Random state for model
    n_jobs : int
        Number of parallel jobs for model
    n_splits : int
        Number of folds per repetition
    n_repeats : int
        Number of repetitions
    base_random_state : int
        Base random state
    verbose : bool
        Whether to print details
    
    Returns:
    --------
    results_df : DataFrame with repeat_idx, fold_idx, and metrics
    error_flags : dict tracking errors per metric per fold
    """
    # Generate all splits
    all_splits = generate_cv_splits(X, y, n_splits, n_repeats, base_random_state)
    
    error_flags = {metric: [] for metric in metrics}
    result_rows = []
    
    for split in all_splits:
        repeat_idx = split['repeat_idx']
        fold_idx = split['fold_idx']
        train_idx = split['train_idx']
        val_idx = split['val_idx']
        
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_test = y[train_idx], y[val_idx]
        
        # Compute class weights for this fold's training data
        sample_weight = _compute_class_weights(y_train, event_col_name, weight_ratio)
        
        if verbose:
            events_mask = y_train[event_col_name].astype(bool)
            n_events = events_mask.sum()
            n_non_events = len(events_mask) - n_events
            w_event = sample_weight[events_mask][0] if n_events > 0 else 0
            w_non_event = sample_weight[~events_mask][0] if n_non_events > 0 else 0
            print(f"  Fold {fold_idx}: {n_events} events (w={w_event:.2f}), "
                  f"{n_non_events} non-events (w={w_non_event:.2f})")
        
        # Scale data
        if scale_data:
            if continuous_features is not None:
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
                
                feature_names = continuous_features + [col for col in X_train.columns if col not in continuous_features]
                
                X_train = pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train.index)
                X_test = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index)
            else:
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
        
        # Initialize model
        model_params_copy = model_params.copy()
        if random_state is not None:
            model_params_copy['random_state'] = random_state
        if n_jobs is not None:
            model_params_copy['n_jobs'] = n_jobs
        
        # DEBUG: Log parameters being passed to model
        if verbose:
            print(f"    DEBUG: model_params_copy = {model_params_copy}")
        
        try:
            # Try to initialize model with all parameters
            model = model_class(**model_params_copy)
            
            # DEBUG: Log resulting model attributes
            if verbose and hasattr(model, 'n_estimators'):
                print(f"    DEBUG: model.n_estimators = {model.n_estimators}")
                
        except TypeError as e:
            # DEBUG: Log TypeError during initialization
            if verbose:
                print(f"    DEBUG: TypeError during initialization: {e}")
                
            # If initialization fails due to unexpected parameters, retry without them
            if 'unexpected keyword argument' in str(e):
                model_params_copy = model_params.copy()
                if random_state is not None:
                    try:
                        model = model_class(**model_params_copy, random_state=random_state)
                    except TypeError:
                        model = model_class(**model_params_copy)
                else:
                    model = model_class(**model_params_copy)
                
                # DEBUG: Log model after retry
                if verbose and hasattr(model, 'n_estimators'):
                    print(f"    DEBUG: model.n_estimators (after retry) = {model.n_estimators}")
            else:
                raise
        
        try:
            
            # Build fit kwargs
            fit_kwargs = {'sample_weight': sample_weight}
            monitor = None
            
            # Early stopping via monitor - uses existing CV validation fold (no data loss)
            if n_iter_no_change is not None and hasattr(model, 'n_estimators'):
                if verbose:
                    print(f"    Early stopping with CV validation fold: n_iter_no_change={n_iter_no_change}, tol={tol}")
                monitor = _EarlyStoppingMonitor(
                    X_val=X_test, y_val=y_test,
                    n_iter_no_change=n_iter_no_change,
                    tol=tol,
                    verbose=False  # Silent to reduce clutter
                )
                fit_kwargs['monitor'] = monitor
            
            model.fit(X_train, y_train, **fit_kwargs)
            
            if verbose and monitor is not None and hasattr(model, 'n_estimators_'):
                if monitor.stopped_early:
                    saved = model_params_copy.get('n_estimators', model.n_estimators) - monitor.best_iteration - 1
                    print(f"    Early stop at iter {monitor.best_iteration+1}/{model_params_copy.get('n_estimators', '?')} (saved {saved} iters)")
                else:
                    print(f"    Completed all {model.n_estimators_} iterations (no early stop)")
        except Exception as e:
            print(f"  Error in repeat {repeat_idx+1}, fold {fold_idx+1}: Model fitting failed - {type(e).__name__}: {str(e)}")
            result_row = {'repeat_idx': repeat_idx, 'fold_idx': fold_idx}
            for metric in metrics:
                result_row[metric] = np.nan
                error_flags[metric].append(True)
            result_rows.append(result_row)
            continue
        
        result_row = {'repeat_idx': repeat_idx, 'fold_idx': fold_idx}
        
        # Get predictions
        pred = None
        has_surv_func = hasattr(model, 'predict_survival_function')
        
        try:
            pred = model.predict(X_test)
        except Exception:
            pass
        
        # Evaluate metrics
        for metric in metrics:
            error_occurred = False
            
            if metric == 'c-index':
                try:
                    c_index = concordance_index_censored(
                        y_test['event'], y_test['time'], pred
                    )
                    result_row[metric] = c_index[0]
                except Exception as e:
                    if verbose:
                        print(f"    Warning: C-index calculation failed: {str(e)}")
                    result_row[metric] = np.nan
                    error_occurred = True
                    
            elif metric == 'c-index-ipcw':
                try:
                    tau = min(y_train['time'].max(), y_test['time'].max()) * 0.95
                    c_index_ipcw = concordance_index_ipcw(
                        y_train, y_test, pred, tau=tau
                    )
                    result_row[metric] = c_index_ipcw[0]
                except Exception as e:
                    if verbose:
                        print(f"    Warning: C-index IPCW calculation failed: {str(e)}")
                    result_row[metric] = np.nan
                    error_occurred = True
                    
            elif metric == 'int-brier-score':
                try:
                    if has_surv_func:
                        max_time = min(y_train['time'].max(), y_test['time'].max()) * 0.95
                        times = np.percentile(y_train['time'], np.linspace(5, 90, 10))
                        times = np.unique(times)
                        times = times[(times > 0) & (times <= max_time)]
                        
                        if len(times) > 1:
                            surv_funcs = model.predict_survival_function(X_test)
                            probs = np.asarray([[fn(t) for t in times] for fn in surv_funcs])
                            score = integrated_brier_score(y_train, y_test, probs, times)
                            result_row[metric] = score
                        else:
                            result_row[metric] = np.nan
                            error_occurred = True
                    else:
                        result_row[metric] = np.nan
                        error_occurred = True
                except Exception as e:
                    if verbose:
                        print(f"    Warning: Integrated Brier Score calculation failed: {str(e)}")
                    result_row[metric] = np.nan
                    error_occurred = True
                    
            elif metric.startswith('brier-'):
                try:
                    if has_surv_func:
                        max_time = min(y_train['time'].max(), y_test['time'].max())
                        
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
                                surv_funcs = model.predict_survival_function(X_test)
                                probs = np.asarray([fn(time_point) for fn in surv_funcs])
                                
                                score = float(brier_score(y_train, y_test, probs, time_point)[1][0])
                                result_row[metric] = score
                            else:
                                result_row[metric] = np.nan
                                error_occurred = True
                        else:
                            result_row[metric] = np.nan
                            error_occurred = True
                    else:
                        result_row[metric] = np.nan
                        error_occurred = True
                except Exception as e:
                    if verbose:
                        print(f"    Warning: {metric} calculation failed: {str(e)}")
                    result_row[metric] = np.nan
                    error_occurred = True
            
            elif metric == 'r2-ibs':
                # R²_IBS - Index of Prediction Proficiency
                # Measures improvement over Kaplan-Meier null model
                try:
                    if has_surv_func:
                        max_time = min(y_train['time'].max(), y_test['time'].max()) * 0.95
                        times = np.percentile(y_train['time'], np.linspace(5, 90, 10))
                        times = np.unique(times)
                        times = times[(times > 0) & (times <= max_time)]
                        
                        if len(times) > 1:
                            # Model predictions
                            surv_funcs = model.predict_survival_function(X_test)
                            model_surv_probs = np.asarray([[fn(t) for t in times] for fn in surv_funcs])
                            ibs_model = integrated_brier_score(y_train, y_test, model_surv_probs, times)
                            
                            # Null model (Kaplan-Meier) predictions
                            km_time, km_surv = kaplan_meier_estimator(
                                y_train['event'], y_train['time']
                            )
                            # Interpolate KM survival function at evaluation times
                            km_surv_at_times = np.interp(times, km_time, km_surv, left=1.0, right=km_surv[-1])
                            # Repeat KM predictions for all test samples
                            km_surv_probs = np.tile(km_surv_at_times, (len(y_test), 1))
                            ibs_null = integrated_brier_score(y_train, y_test, km_surv_probs, times)
                            
                            # R²_IBS = 1 - (IBS_model / IBS_null)
                            if ibs_null > 0:
                                r2_ibs = 1.0 - (ibs_model / ibs_null)
                                result_row[metric] = r2_ibs
                            else:
                                result_row[metric] = np.nan
                                error_occurred = True
                        else:
                            result_row[metric] = np.nan
                            error_occurred = True
                    else:
                        result_row[metric] = np.nan
                        error_occurred = True
                except Exception as e:
                    if verbose:
                        print(f"    Warning: R²_IBS calculation failed: {str(e)}")
                    result_row[metric] = np.nan
                    error_occurred = True
            
            elif metric.startswith('r2-brier-'):
                # R² Brier score at specific time points
                # Measures improvement over Kaplan-Meier null model at a given time
                try:
                    if has_surv_func:
                        max_time = min(y_train['time'].max(), y_test['time'].max())
                        
                        time_points = {
                            'r2-brier-6m': 0.5 * 365,
                            'r2-brier-1y': 1 * 365,
                            'r2-brier-2y': 2 * 365,
                            'r2-brier-3y': 3 * 365,
                            'r2-brier-4y': 4 * 365,
                            'r2-brier-5y': 5 * 365,
                            'r2-brier-10y': 10 * 365,
                            'r2-brier-14y': 14 * 365
                        }
                        
                        if metric in time_points:
                            time_point = time_points[metric]
                            if time_point <= max_time * 0.95:
                                # Model Brier score at this time point
                                surv_funcs = model.predict_survival_function(X_test)
                                model_surv_probs = np.asarray([fn(time_point) for fn in surv_funcs])
                                bs_model = float(brier_score(y_train, y_test, model_surv_probs, time_point)[1][0])
                                
                                # Null model (Kaplan-Meier) Brier score at this time point
                                km_time, km_surv = kaplan_meier_estimator(y_train['event'], y_train['time'])
                                km_surv_at_time = np.interp(time_point, km_time, km_surv, left=1.0, right=km_surv[-1])
                                km_surv_probs = np.full(len(y_test), km_surv_at_time)
                                bs_null = float(brier_score(y_train, y_test, km_surv_probs, time_point)[1][0])
                                
                                # R² = 1 - (BS_model / BS_null)
                                result_row[metric] = 1.0 - (bs_model / bs_null) if bs_null > 0 else np.nan
                            else:
                                result_row[metric] = np.nan
                                error_occurred = True
                        else:
                            result_row[metric] = np.nan
                            error_occurred = True
                    else:
                        result_row[metric] = np.nan
                        error_occurred = True
                except Exception as e:
                    if verbose:
                        print(f"    Warning: {metric} calculation failed: {str(e)}")
                    result_row[metric] = np.nan
                    error_occurred = True
            
            elif metric == 'cumulative-dynamic-auc':
                try:
                    max_time = min(y_train['time'].max(), y_test['time'].max()) * 0.95
                    times = np.percentile(y_train['time'], np.linspace(10, 90, 10))
                    times = np.unique(times[(times > 0) & (times <= max_time)])
                    
                    if len(times) >= 2 and pred is not None:
                        _, mean_auc = cumulative_dynamic_auc(y_train, y_test, pred, times)
                        result_row[metric] = mean_auc
                    else:
                        result_row[metric] = np.nan
                        error_occurred = True
                except Exception as e:
                    if verbose: print(f"    Warning: Cumulative Dynamic AUC failed: {str(e)}")
                    result_row[metric] = np.nan
                    error_occurred = True
                    
            elif metric == 'cumulative-dynamic-prauc':
                # Cumulative/Dynamic PR-AUC across multiple time points
                try:
                    max_time = min(y_train['time'].max(), y_test['time'].max()) * 0.95
                    times = np.percentile(y_train['time'], np.linspace(10, 90, 10))
                    times = np.unique(times[(times > 0) & (times <= max_time)])
                    
                    if len(times) >= 2 and pred is not None:
                        prauc_scores = []
                        for t in times:
                            valid_mask, y_binary, weights = get_ipcw_weights_for_prauc(y_train, y_test, t)
                            if valid_mask.sum() > 0 and y_binary[valid_mask].sum() > 0:
                                precision_vals, recall_vals, _ = precision_recall_curve(
                                    y_binary[valid_mask], pred[valid_mask],
                                    sample_weight=weights[valid_mask]
                                )
                                prauc_scores.append(sklearn_auc(recall_vals, precision_vals))
                        
                        if len(prauc_scores) > 0:
                            result_row[metric] = np.mean(prauc_scores)
                        else:
                            result_row[metric] = np.nan
                            error_occurred = True
                    else:
                        result_row[metric] = np.nan
                        error_occurred = True
                except Exception as e:
                    if verbose:
                        print(f"    Warning: Cumulative Dynamic PR-AUC calculation failed: {str(e)}")
                    result_row[metric] = np.nan
                    error_occurred = True
            
            elif metric.startswith('prauc-'):
                # PR-AUC at specific time points
                try:
                    max_time = min(y_train['time'].max(), y_test['time'].max())
                    
                    time_points = {
                        'prauc-6m': 0.5 * 365,
                        'prauc-1y': 1 * 365,
                        'prauc-2y': 2 * 365,
                        'prauc-3y': 3 * 365,
                        'prauc-4y': 4 * 365,
                        'prauc-5y': 5 * 365,
                        'prauc-10y': 10 * 365,
                        'prauc-14y': 14 * 365
                    }
                    
                    if metric in time_points and pred is not None:
                        time_point = time_points[metric]
                        if time_point <= max_time:
                            valid_mask, y_binary, weights = get_ipcw_weights_for_prauc(y_train, y_test, time_point)
                            if valid_mask.sum() > 0 and y_binary[valid_mask].sum() > 0:
                                precision_vals, recall_vals, _ = precision_recall_curve(
                                    y_binary[valid_mask], pred[valid_mask],
                                    sample_weight=weights[valid_mask]
                                )
                                result_row[metric] = sklearn_auc(recall_vals, precision_vals)
                            else:
                                result_row[metric] = np.nan
                                error_occurred = True
                        else:
                            result_row[metric] = np.nan
                            error_occurred = True
                    else:
                        result_row[metric] = np.nan
                        error_occurred = True
                except Exception as e:
                    if verbose:
                        print(f"    Warning: {metric} calculation failed: {str(e)}")
                    result_row[metric] = np.nan
                    error_occurred = True
            
            error_flags[metric].append(error_occurred)
        
        result_rows.append(result_row)
    
    results_df = pd.DataFrame(result_rows)
    return results_df, error_flags


def balanced_ensemble_repeated_grid_search(model_class, X, y, param_grid, metrics,
                                           weight_ratio_range,
                                           model_name='Model',
                                           event_col_name='event',
                                           n_iter_no_change=None, tol=1e-4,
                                           scale_data=False, scaler_type='standard',
                                           continuous_features=None,
                                           random_state=None, n_jobs=None,
                                           n_splits=5, n_repeats=10, base_random_state=42,
                                           verbose=True):
    """
    Perform repeated stratified cross-validation grid search for ensemble survival 
    models using class-weighted training to handle class imbalance.
    
    For each fold, sample_weight is passed to model.fit() so that events are
    upweighted. The model sees ALL data every iteration but gives more importance
    to events via weights.
    
    Early stopping (GBM only):
    - Set n_iter_no_change to enable early stopping using the CV validation fold
    - No training data is lost (unlike validation_fraction)
    - Training stops if c-index on validation fold doesn't improve by tol 
      for n_iter_no_change consecutive boosting iterations
    
    Parameters:
    -----------
    model_class : class
        Model class (RandomSurvivalForest, GradientBoostingSurvivalAnalysis)
    X : DataFrame  
        Feature matrix
    y : structured array
        Survival data with 'event' and 'time' fields
    param_grid : dict
        Dictionary with parameter names as keys and lists of parameter settings.
        Should NOT include 'weight_ratio' — use weight_ratio_range instead.
    metrics : list
        List of metrics to evaluate: 'c-index', 'c-index-ipcw', 'int-brier-score',
        'brier-Xy', 'cumulative-dynamic-auc'
    weight_ratio_range : list
        List of weight_ratio values to search over.
        - None or 'balanced': balanced inverse-frequency weights
        - A number K: events get weight K, non-events get 1
        - 1 or 'None' or 'Full': no weighting (standard training)
    model_name : str
        Name of the model for display
    event_col_name : str
        Name of the event field in the structured array y (default: 'event')
    n_iter_no_change : int or None
        If set, enables early stopping using the CV validation fold.
        Training stops if no improvement in c-index for this many consecutive
        boosting iterations. Only works with GBM, ignored for RSF.
    tol : float
        Minimum improvement in c-index to count as an improvement (default: 1e-4).
    scale_data : bool
        Whether to scale the data
    scaler_type : str
        'standard' or 'minmax'
    continuous_features : list or None
        List of continuous feature names to scale.
    random_state : int
        Random state for model
    n_jobs : int
        Number of parallel jobs for model
    n_splits : int
        Number of folds per repetition
    n_repeats : int
        Number of repetitions
    base_random_state : int
        Base random state
    verbose : bool
        Whether to print progress
    
    Returns:
    --------
    results_df : DataFrame with all detailed results
    summary_df : DataFrame with aggregated results
    best_params : Dictionary with best parameters for each metric
    """
    start_time = time.time()
    
    # Generate all splits once
    all_splits = generate_cv_splits(X, y, n_splits, n_repeats, base_random_state)
    
    # Generate parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    model_param_combinations = list(product(*param_values))
    
    total_combinations = len(model_param_combinations) * len(weight_ratio_range)
    
    # Dataset info
    n_events_total = y[event_col_name].sum()
    n_non_events_total = len(y) - n_events_total
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"{model_name.upper()} CLASS-WEIGHTED REPEATED GRID SEARCH")
        print(f"{'='*80}")
        print(f"Model: {model_class.__name__}")
        print(f"Strategy: Class-weighted training via sample_weight in .fit()")
        print(f"  - Model sees ALL training data every iteration")
        print(f"  - Events are upweighted to compensate for class imbalance")
        print(f"Dataset: {n_events_total} events, {n_non_events_total} non-events "
              f"(ratio 1:{n_non_events_total/max(n_events_total,1):.1f})")
        print(f"Scaling: {'Yes (' + scaler_type + ')' if scale_data else 'No'}")
        print(f"Metrics: {metrics}")
        print(f"Weight ratio range: {weight_ratio_range}")
        for wr in weight_ratio_range:
            if wr is None or wr == 'balanced':
                w_e = n_events_total + n_non_events_total
                print(f"    weight_ratio=balanced → w_event≈{w_e/(2*max(n_events_total,1)):.2f}, "
                      f"w_non_event≈{w_e/(2*max(n_non_events_total,1)):.2f}")
            elif wr == 1 or wr == 'Full' or wr == 'None':
                print(f"    weight_ratio={wr} → no weighting (standard)")
            else:
                print(f"    weight_ratio={wr} → w_event={wr}, w_non_event=1")
        print(f"Parameter Grid:")
        for param, values in param_grid.items():
            print(f"  - {param}: {values}")
        print(f"Total hyperparameter combinations: {total_combinations}")
        print(f"Folds per repetition: {n_splits}")
        print(f"Repetitions: {n_repeats}")
        if n_iter_no_change is not None:
            print(f"Early stopping: n_iter_no_change={n_iter_no_change}, tol={tol}")
            print(f"  - Uses CV validation fold (no training data lost)")
        print(f"{'='*80}\n")
    
    all_results = []
    current_iteration = 0
    
    for weight_ratio in weight_ratio_range:
        for param_values_tuple in model_param_combinations:
            params = dict(zip(param_names, param_values_tuple))
            current_iteration += 1
            
            if verbose:
                print(f"\n[{current_iteration}/{total_combinations}]")
                print(f"  DEBUG: weight_ratio = {weight_ratio}")
                print(f"  DEBUG: param_names = {param_names}")
                print(f"  DEBUG: param_values_tuple = {param_values_tuple}")
                print(f"  DEBUG: params dict = {params}")
                
                elapsed = time.time() - start_time
                if weight_ratio is None or weight_ratio == 'balanced':
                    wr_str = 'balanced'
                elif weight_ratio == 1 or weight_ratio == 'Full' or weight_ratio == 'None':
                    wr_str = 'no weighting'
                else:
                    wr_str = str(weight_ratio)
                print(f"  Testing: weight_ratio={wr_str}, params: {params}")
                if current_iteration > 1:
                    avg_time = elapsed / (current_iteration - 1)
                    remaining = avg_time * (total_combinations - current_iteration)
                    print(f"  Elapsed: {elapsed/60:.1f}min, Est. remaining: {remaining/60:.1f}min")
            
            try:
                # Determine actual weight_ratio for the CV function
                # 'Full', 'None', or 1 means no weighting → use weight_ratio=1
                # 'balanced' or None → use weight_ratio=None (balanced)
                # A number → use that number
                if weight_ratio == 'Full' or weight_ratio == 'None':
                    actual_wr = 1.0  # No weighting
                elif weight_ratio == 'balanced':
                    actual_wr = None  # Balanced weighting
                else:
                    actual_wr = weight_ratio  # None (balanced) or a number
                
                use_weighting = actual_wr is None or (actual_wr is not None and actual_wr != 1)
                
                if use_weighting:
                    cv_results, error_flags = balanced_ensemble_repeated_cross_validation(
                        model_class=model_class,
                        X=X, y=y,
                        model_params=params,
                        metrics=metrics,
                        weight_ratio=actual_wr,
                        event_col_name=event_col_name,
                        n_iter_no_change=n_iter_no_change,
                        tol=tol,
                        scale_data=scale_data,
                        scaler_type=scaler_type,
                        continuous_features=continuous_features,
                        random_state=random_state,
                        n_jobs=n_jobs,
                        n_splits=n_splits,
                        n_repeats=n_repeats,
                        base_random_state=base_random_state,
                        verbose=False
                    )
                else:
                    # No weighting — use regular ensemble CV
                    cv_results, error_flags = ensemble_repeated_cross_validation(
                        model_class=model_class,
                        X=X, y=y,
                        model_params=params,
                        metrics=metrics,
                        scale_data=scale_data,
                        scaler_type=scaler_type,
                        continuous_features=continuous_features,
                        random_state=random_state,
                        n_jobs=n_jobs,
                        n_controls_per_case=None,
                        dataset=None,
                        feature_columns=None,
                        n_splits=n_splits,
                        n_repeats=n_repeats,
                        base_random_state=base_random_state,
                        verbose=False
                    )
                
                # Store weight_ratio as a readable label
                if weight_ratio is None or weight_ratio == 'balanced':
                    wr_label = 'balanced'
                elif weight_ratio == 1 or weight_ratio == 'Full' or weight_ratio == 'None':
                    wr_label = 'Full'
                else:
                    wr_label = weight_ratio
                
                # Add parameter columns to results
                for idx, row in cv_results.iterrows():
                    result_row = params.copy()
                    result_row['weight_ratio'] = wr_label
                    result_row['repeat_idx'] = row['repeat_idx']
                    result_row['fold_idx'] = row['fold_idx']
                    for metric in metrics:
                        result_row[metric] = row[metric]
                    # Add convergence info
                    if 'n_estimators_trained' in row:
                        result_row['n_estimators_trained'] = row['n_estimators_trained']
                        result_row['n_estimators_max'] = row['n_estimators_max']
                        result_row['stopped_early'] = row['stopped_early']
                    all_results.append(result_row)
                
                if verbose:
                    for metric in metrics:
                        score = cv_results[metric].mean()
                        std = cv_results[metric].std()
                        print(f"  {metric}: {score:.4f} ± {std:.4f}")
                    
                    # Show convergence info
                    if 'n_estimators_trained' in cv_results.columns:
                        trained = cv_results['n_estimators_trained']
                        if trained.notna().any():
                            max_iters = cv_results['n_estimators_max'].iloc[0]
                            stopped_early = cv_results['stopped_early']
                            n_converged = stopped_early.sum() if stopped_early.notna().any() else 0
                            avg_iters = trained.mean()
                            print(f"  Convergence: {n_converged}/{len(cv_results)} folds stopped early")
                            print(f"  Avg iterations: {avg_iters:.0f}/{max_iters}")
                        
            except Exception as e:
                if verbose:
                    print(f"  Error: {str(e)}")
                
                if weight_ratio is None or weight_ratio == 'balanced':
                    wr_label = 'balanced'
                elif weight_ratio == 1 or weight_ratio == 'Full' or weight_ratio == 'None':
                    wr_label = 'Full'
                else:
                    wr_label = weight_ratio
                
                for split in all_splits:
                    result_row = params.copy()
                    result_row['weight_ratio'] = wr_label
                    result_row['repeat_idx'] = split['repeat_idx']
                    result_row['fold_idx'] = split['fold_idx']
                    for metric in metrics:
                        result_row[metric] = np.nan
                    all_results.append(result_row)
    
    # Create detailed results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Create summary DataFrame using groupby with dropna=False
    param_cols = [col for col in results_df.columns 
                  if col not in ['repeat_idx', 'fold_idx'] + metrics]
    
    summary_rows = []
    for combo, group in results_df.groupby(param_cols, dropna=False):
        row = dict(zip(param_cols, combo if isinstance(combo, tuple) else [combo]))
        
        for metric in metrics:
            row[f'{metric}_mean'] = group[metric].mean()
            row[f'{metric}_std'] = group[metric].std()
            
            for repeat_idx in range(n_repeats):
                repeat_mask = group['repeat_idx'] == repeat_idx
                row[f'{metric}_repeat_{repeat_idx+1}_mean'] = group[repeat_mask][metric].mean()
            
            for repeat_idx in range(n_repeats):
                for fold_idx in range(n_splits):
                    fold_repeat_mask = (group['repeat_idx'] == repeat_idx) & (group['fold_idx'] == fold_idx)
                    if fold_repeat_mask.any():
                        row[f'{metric}_repeat_{repeat_idx+1}_fold_{fold_idx+1}'] = group[fold_repeat_mask][metric].values[0]
                    else:
                        row[f'{metric}_repeat_{repeat_idx+1}_fold_{fold_idx+1}'] = np.nan
        
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Best parameters
    param_cols_for_best = [col for col in summary_df.columns 
                           if not col.endswith('_mean') and not col.endswith('_std') 
                           and not col.startswith(tuple(metrics)) 
                           and '_repeat_' not in col and '_fold_' not in col]
    
    best_params = {}
    for metric in metrics:
        mean_col = f'{metric}_mean'
        valid_results = summary_df[~summary_df[mean_col].isna()].copy()
        
        if len(valid_results) == 0:
            if verbose:
                print(f"\nWarning: No valid results for {metric}")
            best_params[metric] = None
            continue
        
        if 'brier' in metric.lower():
            best_idx = valid_results[mean_col].idxmin()
        else:
            best_idx = valid_results[mean_col].idxmax()
        
        best_row = summary_df.loc[best_idx]
        best_param_dict = {col: best_row[col] for col in param_cols_for_best}
        
        best_params[metric] = {
            'params': best_param_dict,
            'mean': best_row[f'{metric}_mean'],
            'std': best_row[f'{metric}_std']
        }
    
    elapsed_time = time.time() - start_time
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"CLASS-WEIGHTED GRID SEARCH COMPLETE")
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
                    if param == 'weight_ratio':
                        print(f"    {param}: {value}")
                    else:
                        print(f"    {param}: {value}")
        print(f"\n{'='*80}\n")
    
    return results_df, summary_df, best_params


# =============================================================================
# CASE-CONTROL RANDOM SURVIVAL FOREST - PER-TREE SUBSAMPLING
# =============================================================================

class CaseControlRandomSurvivalForest:
    """
    Random Survival Forest with per-tree case-control subsampling.
    
    Each tree is trained on a balanced subsample containing:
    - ALL cases (events) from the training data
    - n_controls_per_case * n_cases random controls (sampled without replacement)
    
    Different trees see different random control subsamples, providing diversity
    while ensuring every tree sees all events. This follows the Balanced Random
    Forest approach adapted for survival analysis.
    
    Parameters:
    -----------
    n_estimators : int
        Number of trees in the forest. Each tree gets its own subsample.
    n_controls_per_case : int
        Number of controls to include per case in each tree's subsample.
        Total controls per tree = n_controls_per_case * n_cases.
    max_depth : int or None
        Maximum depth of each tree.
    min_samples_split : int
        Minimum samples required to split a node.
    min_samples_leaf : int
        Minimum samples required at a leaf node.
    max_features : str, int, float, or None
        Number of features to consider at each split ('sqrt', 'log2', int, float).
    bootstrap : bool
        Whether to bootstrap within each subsample when building each tree.
    random_state : int or None
        Random seed for reproducibility.
    n_jobs : int or None
        Number of parallel jobs for training trees. -1 uses all cores.
    event_col_name : str
        Name of the event field in the structured array y.
    """
    
    def __init__(self, n_estimators=100, n_controls_per_case=5,
                 max_depth=None, min_samples_split=6, min_samples_leaf=3,
                 max_features='sqrt', bootstrap=True,
                 random_state=None, n_jobs=None,
                 event_col_name='event'):
        self.n_estimators = n_estimators
        self.n_controls_per_case = n_controls_per_case
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.event_col_name = event_col_name
    
    def fit(self, X, y, verbose=False):
        """
        Train the forest by fitting each tree on a different case-control subsample.
        
        Parameters:
        -----------
        X : DataFrame or array-like
            Feature matrix.
        y : structured array
            Survival data with event and time fields.
        verbose : bool
            If True, print detailed progress during tree training.
        
        Returns:
        --------
        self
        """
        from sksurv.ensemble import RandomSurvivalForest
        from joblib import Parallel, delayed
        import time
        
        fit_start_time = time.time()
        rng = np.random.RandomState(self.random_state)
        
        events_mask = y[self.event_col_name].astype(bool)
        case_idx = np.where(events_mask)[0]
        control_idx = np.where(~events_mask)[0]
        n_cases = len(case_idx)
        n_controls_available = len(control_idx)
        n_controls_needed = self.n_controls_per_case * n_cases
        
        # Cap at available controls
        n_controls_to_sample = min(n_controls_needed, n_controls_available)
        use_replace = n_controls_needed > n_controls_available
        
        self.n_estimators_ = self.n_estimators
        self.n_cases_ = n_cases
        self.n_controls_per_tree_ = n_controls_to_sample
        
        if verbose:
            print(f"    CC-RSF: Training {self.n_estimators} trees with {n_cases} cases + "
                  f"{n_controls_to_sample} controls per tree...")
            print(f"    Using n_jobs={self.n_jobs} for parallel training")
        
        # Generate all random states upfront
        tree_random_states = [rng.randint(0, 2**31) for _ in range(self.n_estimators)]
        
        def _fit_single_tree(tree_idx, tree_rs):
            """Fit a single tree on a case-control subsample."""
            tree_rng = np.random.RandomState(tree_rs)
            
            # Sample controls for this tree
            sampled_controls = tree_rng.choice(
                control_idx, size=n_controls_to_sample, replace=use_replace
            )
            tree_indices = np.concatenate([case_idx, sampled_controls])
            tree_rng.shuffle(tree_indices)
            
            # Build subsample
            X_sub = X.iloc[tree_indices] if hasattr(X, 'iloc') else X[tree_indices]
            y_sub = y[tree_indices]
            
            # Train individual survival tree (RSF with 1 estimator)
            tree = RandomSurvivalForest(
                n_estimators=1,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                random_state=tree_rs,
                n_jobs=1
            )
            tree.fit(X_sub, y_sub)
            return tree
        
        # Parallel training: use n_jobs=-1 to use all cores, or specified n_jobs
        n_jobs_to_use = self.n_jobs if self.n_jobs is not None else 1
        
        if n_jobs_to_use == 1:
            # Sequential fallback
            if verbose:
                print(f"    Training trees sequentially (n_jobs=1)...")
            self.trees_ = []
            for i, tree_rs in enumerate(tree_random_states):
                if verbose and (i + 1) % max(1, self.n_estimators // 10) == 0:
                    elapsed = time.time() - fit_start_time
                    print(f"      Tree {i+1}/{self.n_estimators} ({100*(i+1)/self.n_estimators:.0f}%) - "
                          f"Elapsed: {elapsed:.1f}s")
                tree = _fit_single_tree(i, tree_rs)
                self.trees_.append(tree)
        else:
            # Parallel training
            if verbose:
                print(f"    Training trees in parallel (n_jobs={n_jobs_to_use})...")
                parallel_start = time.time()
            
            # Use 'loky' backend for true parallelism (multiprocessing)
            # 'threading' backend has issues with Python's GIL
            self.trees_ = Parallel(n_jobs=n_jobs_to_use, backend='loky', verbose=0)(
                delayed(_fit_single_tree)(i, tree_rs) 
                for i, tree_rs in enumerate(tree_random_states)
            )
            
            if verbose:
                parallel_time = time.time() - parallel_start
                print(f"    Parallel training completed in {parallel_time:.1f}s "
                      f"({parallel_time/self.n_estimators:.2f}s per tree)")
                trees_per_sec = self.n_estimators / parallel_time
                print(f"    Throughput: {trees_per_sec:.1f} trees/second")
        
        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.array(X.columns.tolist())
        
        total_fit_time = time.time() - fit_start_time
        if verbose:
            print(f"    CC-RSF fit completed in {total_fit_time:.1f}s")
        
        return self
    
    def predict(self, X):
        """Average risk scores across all trees."""
        predictions = np.array([tree.predict(X) for tree in self.trees_])
        return predictions.mean(axis=0)
    
    def predict_survival_function(self, X):
        """
        Average survival functions across all trees.
        
        Returns a list of callable StepFunction objects (one per sample)
        representing the averaged survival probability over time.
        """
        from sksurv.functions import StepFunction
        
        # Collect survival functions from all trees
        all_surv_funcs = [tree.predict_survival_function(X) for tree in self.trees_]
        
        # Gather all unique time points across trees
        all_times = set()
        for tree_surv_funcs in all_surv_funcs:
            for fn in tree_surv_funcs:
                all_times.update(fn.x)
        all_times = np.sort(np.array(list(all_times)))
        
        # Evaluate each tree's survival functions at common time points and average
        n_samples = len(all_surv_funcs[0])
        avg_surv_probs = np.zeros((n_samples, len(all_times)))
        
        for tree_surv_funcs in all_surv_funcs:
            for j, fn in enumerate(tree_surv_funcs):
                avg_surv_probs[j] += np.array([fn(t) for t in all_times])
        
        avg_surv_probs /= len(self.trees_)
        
        # Create averaged StepFunction objects
        result = []
        for j in range(n_samples):
            result.append(StepFunction(all_times, avg_surv_probs[j]))
        
        return result
    
    def predict_cumulative_hazard_function(self, X):
        """
        Average cumulative hazard functions across all trees.
        
        Returns a list of callable StepFunction objects (one per sample).
        """
        from sksurv.functions import StepFunction
        
        all_chf_funcs = [tree.predict_cumulative_hazard_function(X) for tree in self.trees_]
        
        all_times = set()
        for tree_chf_funcs in all_chf_funcs:
            for fn in tree_chf_funcs:
                all_times.update(fn.x)
        all_times = np.sort(np.array(list(all_times)))
        
        n_samples = len(all_chf_funcs[0])
        avg_chf = np.zeros((n_samples, len(all_times)))
        
        for tree_chf_funcs in all_chf_funcs:
            for j, fn in enumerate(tree_chf_funcs):
                avg_chf[j] += np.array([fn(t) for t in all_times])
        
        avg_chf /= len(self.trees_)
        
        result = []
        for j in range(n_samples):
            result.append(StepFunction(all_times, avg_chf[j]))
        
        return result
    
    @property
    def feature_importances_(self):
        """Average feature importances across all trees."""
        importances = np.array([tree.feature_importances_ for tree in self.trees_])
        return importances.mean(axis=0)
    
    def score(self, X, y):
        """Compute c-index on given data."""
        pred = self.predict(X)
        return concordance_index_censored(
            y[self.event_col_name], y['time'], pred
        )[0]


def cc_rsf_repeated_cross_validation(X, y, model_params, metrics,
                                     n_controls_per_case=5,
                                     event_col_name='event',
                                     scale_data=False, scaler_type='standard',
                                     continuous_features=None,
                                     random_state=None, n_jobs=None,
                                     n_splits=5, n_repeats=10, base_random_state=42,
                                     verbose=False):
    """
    Perform repeated stratified cross-validation for a Case-Control Random 
    Survival Forest where each tree uses a per-tree case-control subsample.
    
    For each CV fold, a CaseControlRandomSurvivalForest is trained where each
    tree sees ALL cases and a random subsample of controls. Different trees
    get different control subsamples.
    
    Parameters:
    -----------
    X : DataFrame
        Feature matrix
    y : structured array
        Survival data with 'event' and 'time' fields
    model_params : dict
        Dictionary of RSF parameters (n_estimators, max_depth, min_samples_split,
        min_samples_leaf, max_features, bootstrap).
        Do NOT include n_controls_per_case here — use the separate parameter.
    metrics : list
        List of metrics to evaluate: 'c-index', 'c-index-ipcw', 'int-brier-score',
        'brier-Xy', 'cumulative-dynamic-auc'
    n_controls_per_case : int
        Number of controls per case in each tree's subsample.
        Total controls per tree = n_controls_per_case * n_cases_in_train_fold.
    event_col_name : str
        Name of the event field in the structured array y (default: 'event')
    scale_data : bool
        Whether to scale the data
    scaler_type : str
        'standard' or 'minmax'
    continuous_features : list or None
        List of continuous feature names to scale. If None and scale_data=True,
        all features are scaled.
    random_state : int or None
        Random state for the model
    n_jobs : int or None
        Number of parallel jobs (currently unused, reserved for future)
    n_splits : int
        Number of folds per repetition
    n_repeats : int
        Number of repetitions
    base_random_state : int
        Base random state for CV splits
    verbose : bool
        Whether to print details
    
    Returns:
    --------
    results_df : DataFrame with repeat_idx, fold_idx, and metrics
    error_flags : dict tracking errors per metric per fold
    """
    # Generate all splits
    all_splits = generate_cv_splits(X, y, n_splits, n_repeats, base_random_state)
    
    error_flags = {metric: [] for metric in metrics}
    result_rows = []
    
    total_folds = len(all_splits)
    overall_start_time = time.time()
    
    for split_idx, split in enumerate(all_splits):
        repeat_idx = split['repeat_idx']
        fold_idx = split['fold_idx']
        train_idx = split['train_idx']
        val_idx = split['val_idx']
        
        fold_start_time = time.time()
        
        if verbose:
            elapsed = time.time() - overall_start_time
            print(f"\n{'='*80}")
            print(f"  FOLD {split_idx+1}/{total_folds} (Repeat {repeat_idx+1}, Fold {fold_idx+1})")
            print(f"  Elapsed: {elapsed/60:.1f}min")
            if split_idx > 0:
                avg_time_per_fold = elapsed / split_idx
                remaining_time = avg_time_per_fold * (total_folds - split_idx)
                print(f"  Est. remaining: {remaining_time/60:.1f}min")
            print(f"{'='*80}")
        
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_test = y[train_idx], y[val_idx]
        
        if verbose:
            events_mask = y_train[event_col_name].astype(bool)
            n_events = events_mask.sum()
            n_non_events = len(events_mask) - n_events
            n_controls_total = min(n_controls_per_case * n_events, n_non_events)
            print(f"  Train: {len(y_train)} samples ({n_events} cases, {n_non_events} controls)")
            print(f"  Val: {len(y_test)} samples")
            print(f"  Each tree will use: {n_events} cases + {n_controls_total} controls "
                  f"({n_controls_per_case} controls/case)")
        
        # Scale data
        if scale_data:
            if continuous_features is not None:
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
                
                feature_names = continuous_features + [col for col in X_train.columns if col not in continuous_features]
                
                X_train = pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train.index)
                X_test = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index)
            else:
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
        
        # Build model parameters
        model_params_copy = model_params.copy()
        model_params_copy['n_controls_per_case'] = n_controls_per_case
        model_params_copy['event_col_name'] = event_col_name
        if random_state is not None:
            model_params_copy['random_state'] = random_state
        if n_jobs is not None:
            model_params_copy['n_jobs'] = n_jobs
        
        try:
            model = CaseControlRandomSurvivalForest(**model_params_copy)
            
            if verbose:
                print(f"\n  Training model...")
            train_start = time.time()
            model.fit(X_train, y_train, verbose=verbose)
            train_time = time.time() - train_start
            
            if verbose:
                print(f"  Model trained in {train_time:.1f}s")
                print(f"  ({model.n_estimators_} trees, "
                      f"{model.n_cases_} cases + {model.n_controls_per_tree_} controls per tree)")
        except Exception as e:
            if verbose:
                print(f"  ERROR: Model fitting failed: {str(e)}")
            result_row = {'repeat_idx': repeat_idx, 'fold_idx': fold_idx}
            for metric in metrics:
                result_row[metric] = np.nan
                error_flags[metric].append(True)
            result_rows.append(result_row)
            continue
        
        result_row = {'repeat_idx': repeat_idx, 'fold_idx': fold_idx}
        
        # Predictions
        if verbose:
            print(f"\n  Making predictions on validation set ({len(X_test)} samples)...")
        pred_start = time.time()
        pred = model.predict(X_test)
        pred_time = time.time() - pred_start
        if verbose:
            print(f"  Risk score predictions completed in {pred_time:.1f}s")
        
        has_surv_func = hasattr(model, 'predict_survival_function')
        
        if verbose:
            print(f"\n  Calculating metrics...")
        
        # Evaluate each metric
        for metric in metrics:
            if verbose:
                metric_start = time.time()
                print(f"    Computing {metric}...", end='', flush=True)
            error_occurred = False
            
            if metric == 'c-index':
                try:
                    c_index = concordance_index_censored(
                        y_test['event'], y_test['time'], pred
                    )[0]
                    result_row[metric] = c_index
                    if verbose:
                        metric_time = time.time() - metric_start
                        print(f" {c_index:.4f} ({metric_time:.1f}s)")
                except Exception as e:
                    if verbose:
                        print(f" FAILED: {str(e)}")
                    result_row[metric] = np.nan
                    error_occurred = True
                    
            elif metric == 'c-index-ipcw':
                try:
                    tau = min(y_train['time'].max(), y_test['time'].max()) * 0.95
                    c_index_ipcw = concordance_index_ipcw(
                        y_train, y_test, pred, tau=tau
                    )[0]
                    result_row[metric] = c_index_ipcw
                    if verbose:
                        metric_time = time.time() - metric_start
                        print(f" {c_index_ipcw:.4f} ({metric_time:.1f}s)")
                except Exception as e:
                    if verbose:
                        print(f" FAILED: {str(e)}")
                    result_row[metric] = np.nan
                    error_occurred = True
                    
            elif metric == 'int-brier-score':
                try:
                    if has_surv_func:
                        if verbose:
                            print(f" [getting survival functions]", end='', flush=True)
                        max_time = min(y_train['time'].max(), y_test['time'].max()) * 0.95
                        times = np.percentile(y_train['time'], np.linspace(5, 90, 10))
                        times = np.unique(times)
                        times = times[(times > 0) & (times <= max_time)]
                        
                        if len(times) > 1:
                            surv_funcs = model.predict_survival_function(X_test)
                            if verbose:
                                print(f" [evaluating at {len(times)} time points]", end='', flush=True)
                            surv_probs = np.asarray([[fn(t) for t in times] for fn in surv_funcs])
                            
                            ibs = integrated_brier_score(y_train, y_test, surv_probs, times)
                            result_row[metric] = ibs
                            if verbose:
                                metric_time = time.time() - metric_start
                                print(f" {ibs:.4f} ({metric_time:.1f}s)")
                        else:
                            result_row[metric] = np.nan
                            error_occurred = True
                            if verbose:
                                print(f" FAILED: insufficient time points")
                    else:
                        result_row[metric] = np.nan
                        error_occurred = True
                        if verbose:
                            print(f" FAILED: no survival function")
                except Exception as e:
                    if verbose:
                        print(f" FAILED: {str(e)}")
                    result_row[metric] = np.nan
                    error_occurred = True
                    
            elif metric.startswith('brier-'):
                try:
                    if has_surv_func:
                        max_time = min(y_train['time'].max(), y_test['time'].max())
                        
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
                                surv_funcs = model.predict_survival_function(X_test)
                                probs = np.asarray([fn(time_point) for fn in surv_funcs])
                                
                                score = float(brier_score(y_train, y_test, probs, time_point)[1][0])
                                result_row[metric] = score
                            else:
                                result_row[metric] = np.nan
                                error_occurred = True
                        else:
                            result_row[metric] = np.nan
                            error_occurred = True
                    else:
                        result_row[metric] = np.nan
                        error_occurred = True
                except Exception as e:
                    if verbose:
                        print(f"    Warning: {metric} calculation failed: {str(e)}")
                    result_row[metric] = np.nan
                    error_occurred = True
            
            elif metric == 'r2-ibs':
                # R²_IBS - Index of Prediction Proficiency
                # Measures improvement over Kaplan-Meier null model
                try:
                    if has_surv_func:
                        max_time = min(y_train['time'].max(), y_test['time'].max()) * 0.95
                        times = np.percentile(y_train['time'], np.linspace(5, 90, 10))
                        times = np.unique(times)
                        times = times[(times > 0) & (times <= max_time)]
                        
                        if len(times) > 1:
                            # Model predictions
                            surv_funcs = model.predict_survival_function(X_test)
                            model_surv_probs = np.asarray([[fn(t) for t in times] for fn in surv_funcs])
                            ibs_model = integrated_brier_score(y_train, y_test, model_surv_probs, times)
                            
                            # Null model (Kaplan-Meier) predictions
                            km_time, km_surv = kaplan_meier_estimator(
                                y_train['event'], y_train['time']
                            )
                            # Interpolate KM survival function at evaluation times
                            km_surv_at_times = np.interp(times, km_time, km_surv, left=1.0, right=km_surv[-1])
                            # Repeat KM predictions for all test samples
                            km_surv_probs = np.tile(km_surv_at_times, (len(y_test), 1))
                            ibs_null = integrated_brier_score(y_train, y_test, km_surv_probs, times)
                            
                            # R²_IBS = 1 - (IBS_model / IBS_null)
                            if ibs_null > 0:
                                r2_ibs = 1.0 - (ibs_model / ibs_null)
                                result_row[metric] = r2_ibs
                            else:
                                result_row[metric] = np.nan
                                error_occurred = True
                        else:
                            result_row[metric] = np.nan
                            error_occurred = True
                    else:
                        result_row[metric] = np.nan
                        error_occurred = True
                except Exception as e:
                    if verbose:
                        print(f"    Warning: R²_IBS calculation failed: {str(e)}")
                    result_row[metric] = np.nan
                    error_occurred = True
            
            elif metric.startswith('r2-brier-'):
                # R² Brier score at specific time points
                # Measures improvement over Kaplan-Meier null model at a given time
                try:
                    if has_surv_func:
                        max_time = min(y_train['time'].max(), y_test['time'].max())
                        
                        time_points = {
                            'r2-brier-6m': 0.5 * 365,
                            'r2-brier-1y': 1 * 365,
                            'r2-brier-2y': 2 * 365,
                            'r2-brier-3y': 3 * 365,
                            'r2-brier-4y': 4 * 365,
                            'r2-brier-5y': 5 * 365,
                            'r2-brier-10y': 10 * 365,
                            'r2-brier-14y': 14 * 365
                        }
                        
                        if metric in time_points:
                            time_point = time_points[metric]
                            if time_point <= max_time * 0.95:
                                # Model Brier score at this time point
                                surv_funcs = model.predict_survival_function(X_test)
                                model_surv_probs = np.asarray([fn(time_point) for fn in surv_funcs])
                                bs_model = float(brier_score(y_train, y_test, model_surv_probs, time_point)[1][0])
                                
                                # Null model (Kaplan-Meier) Brier score at this time point
                                km_time, km_surv = kaplan_meier_estimator(y_train['event'], y_train['time'])
                                km_surv_at_time = np.interp(time_point, km_time, km_surv, left=1.0, right=km_surv[-1])
                                km_surv_probs = np.full(len(y_test), km_surv_at_time)
                                bs_null = float(brier_score(y_train, y_test, km_surv_probs, time_point)[1][0])
                                
                                # R² = 1 - (BS_model / BS_null)
                                result_row[metric] = 1.0 - (bs_model / bs_null) if bs_null > 0 else np.nan
                            else:
                                result_row[metric] = np.nan
                                error_occurred = True
                        else:
                            result_row[metric] = np.nan
                            error_occurred = True
                    else:
                        result_row[metric] = np.nan
                        error_occurred = True
                except Exception as e:
                    if verbose:
                        print(f"    Warning: {metric} calculation failed: {str(e)}")
                    result_row[metric] = np.nan
                    error_occurred = True
            
            elif metric == 'cumulative-dynamic-auc':
                try:
                    if verbose:
                        print(f" [computing AUC at multiple times]", end='', flush=True)
                    max_time = min(y_train['time'].max(), y_test['time'].max()) * 0.95
                    times = np.percentile(y_train['time'], np.linspace(10, 90, 10))
                    times = np.unique(times)
                    times = times[(times > 0) & (times <= max_time)]
                    
                    if len(times) >= 2:
                        auc_values, mean_auc = cumulative_dynamic_auc(
                            y_train, y_test, pred, times
                        )
                        result_row[metric] = mean_auc
                        if verbose:
                            metric_time = time.time() - metric_start
                            print(f" {mean_auc:.4f} ({metric_time:.1f}s)")
                    else:
                        result_row[metric] = np.nan
                        error_occurred = True
                        if verbose:
                            print(f" FAILED: insufficient time points")
                except Exception as e:
                    if verbose:
                        print(f" FAILED: {str(e)}")
                    result_row[metric] = np.nan
                    error_occurred = True
            
            elif metric == 'cumulative-dynamic-prauc':
                # Cumulative/Dynamic PR-AUC across multiple time points
                try:
                    if verbose:
                        print(f" [computing PR-AUC at multiple times]", end='', flush=True)
                    max_time = min(y_train['time'].max(), y_test['time'].max()) * 0.95
                    times = np.percentile(y_train['time'], np.linspace(10, 90, 10))
                    times = np.unique(times[(times > 0) & (times <= max_time)])
                    
                    if len(times) >= 2 and pred is not None:
                        prauc_scores = []
                        if verbose:
                            print(f" [evaluating {len(times)} time points]", end='', flush=True)
                        for t_idx, t in enumerate(times):
                            if verbose and t_idx % 3 == 0:
                                print(f".", end='', flush=True)
                            valid_mask, y_binary, weights = get_ipcw_weights_for_prauc(y_train, y_test, t)
                            if valid_mask.sum() > 0 and y_binary[valid_mask].sum() > 0:
                                precision_vals, recall_vals, _ = precision_recall_curve(
                                    y_binary[valid_mask], pred[valid_mask],
                                    sample_weight=weights[valid_mask]
                                )
                                prauc_scores.append(sklearn_auc(recall_vals, precision_vals))
                        
                        if len(prauc_scores) > 0:
                            result_row[metric] = np.mean(prauc_scores)
                        else:
                            result_row[metric] = np.nan
                            error_occurred = True
                    else:
                        result_row[metric] = np.nan
                        error_occurred = True
                except Exception as e:
                    if verbose:
                        print(f"    Warning: Cumulative Dynamic PR-AUC calculation failed: {str(e)}")
                    result_row[metric] = np.nan
                    error_occurred = True
            
            elif metric.startswith('prauc-'):
                # PR-AUC at specific time points
                try:
                    max_time = min(y_train['time'].max(), y_test['time'].max())
                    
                    time_points = {
                        'prauc-6m': 0.5 * 365,
                        'prauc-1y': 1 * 365,
                        'prauc-2y': 2 * 365,
                        'prauc-3y': 3 * 365,
                        'prauc-4y': 4 * 365,
                        'prauc-5y': 5 * 365,
                        'prauc-10y': 10 * 365,
                        'prauc-14y': 14 * 365
                    }
                    
                    if metric in time_points and pred is not None:
                        time_point = time_points[metric]
                        if time_point <= max_time:
                            valid_mask, y_binary, weights = get_ipcw_weights_for_prauc(y_train, y_test, time_point)
                            if valid_mask.sum() > 0 and y_binary[valid_mask].sum() > 0:
                                precision_vals, recall_vals, _ = precision_recall_curve(
                                    y_binary[valid_mask], pred[valid_mask],
                                    sample_weight=weights[valid_mask]
                                )
                                result_row[metric] = sklearn_auc(recall_vals, precision_vals)
                            else:
                                result_row[metric] = np.nan
                                error_occurred = True
                        else:
                            result_row[metric] = np.nan
                            error_occurred = True
                    else:
                        result_row[metric] = np.nan
                        error_occurred = True
                except Exception as e:
                    if verbose:
                        print(f"    Warning: {metric} calculation failed: {str(e)}")
                    result_row[metric] = np.nan
                    error_occurred = True
            
            error_flags[metric].append(error_occurred)
        
        result_rows.append(result_row)
        
        fold_time = time.time() - fold_start_time
        if verbose:
            print(f"\n  Fold {split_idx+1}/{total_folds} Results:")
            for metric in metrics:
                if not pd.isna(result_row[metric]):
                    print(f"    {metric}: {result_row[metric]:.4f}")
                else:
                    print(f"    {metric}: N/A (error)")
            print(f"  Fold total time: {fold_time:.1f}s ({fold_time/60:.1f}min)")
    
    if verbose:
        total_time = time.time() - overall_start_time
        print(f"\n{'='*80}")
        print(f"All {total_folds} folds completed in {total_time/60:.1f}min")
        print(f"Average time per fold: {total_time/total_folds:.1f}s")
        print(f"{'='*80}")
    
    results_df = pd.DataFrame(result_rows)
    return results_df, error_flags


def cc_rsf_repeated_grid_search(X, y, param_grid, metrics,
                                n_controls_per_case_range,
                                model_name='CC-RSF',
                                event_col_name='event',
                                scale_data=False, scaler_type='standard',
                                continuous_features=None,
                                random_state=None, n_jobs=None,
                                n_splits=5, n_repeats=10, base_random_state=42,
                                verbose=True):
    """
    Perform repeated stratified cross-validation grid search for a Case-Control
    Random Survival Forest with per-tree case-control subsampling.
    
    Each tree in the forest is trained on a balanced subsample containing all
    cases and n_controls_per_case * n_cases random controls. Different trees
    see different control subsamples.
    
    The grid search iterates over:
    - RSF hyperparameters in param_grid (n_estimators, max_depth, etc.)
    - n_controls_per_case values in n_controls_per_case_range
    
    Parameters:
    -----------
    X : DataFrame
        Feature matrix
    y : structured array
        Survival data with 'event' and 'time' fields
    param_grid : dict
        Dictionary with RSF parameter names as keys and lists of settings.
        Supported: n_estimators, max_depth, min_samples_split, min_samples_leaf,
        max_features, bootstrap.
        Do NOT include n_controls_per_case — use n_controls_per_case_range.
    metrics : list
        List of metrics: 'c-index', 'c-index-ipcw', 'int-brier-score',
        'brier-Xy', 'cumulative-dynamic-auc'
    n_controls_per_case_range : list
        List of n_controls_per_case values to search over.
        - int: number of controls per case per tree
        - 'Full': train standard RSF on full training data (no subsampling)
    model_name : str
        Name of the model for display
    event_col_name : str
        Name of the event field in y (default: 'event')
    scale_data : bool
        Whether to scale the data
    scaler_type : str
        'standard' or 'minmax'
    continuous_features : list or None
        List of continuous feature names to scale.
    random_state : int or None
        Random state for the model
    n_jobs : int or None
        Number of parallel jobs
    n_splits : int
        Number of folds per repetition
    n_repeats : int
        Number of repetitions
    base_random_state : int
        Base random state
    verbose : bool
        Whether to print progress
    
    Returns:
    --------
    results_df : DataFrame with all detailed results
    summary_df : DataFrame with aggregated results
    best_params : Dictionary with best parameters for each metric
    """
    start_time = time.time()
    
    # Generate all splits once
    all_splits = generate_cv_splits(X, y, n_splits, n_repeats, base_random_state)
    
    # Generate parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    model_param_combinations = list(product(*param_values))
    
    total_combinations = len(model_param_combinations) * len(n_controls_per_case_range)
    
    # Dataset info
    n_events_total = y[event_col_name].sum()
    n_non_events_total = len(y) - n_events_total
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"{model_name.upper()} PER-TREE CASE-CONTROL REPEATED GRID SEARCH")
        print(f"{'='*80}")
        print(f"Model: CaseControlRandomSurvivalForest")
        print(f"Strategy: Per-tree case-control subsampling")
        print(f"  - Each tree sees ALL cases + random controls")
        print(f"  - Different trees get different control subsamples")
        print(f"Dataset: {n_events_total} events, {n_non_events_total} non-events "
              f"(ratio 1:{n_non_events_total/max(n_events_total,1):.1f})")
        print(f"Scaling: {'Yes (' + scaler_type + ')' if scale_data else 'No'}")
        print(f"Metrics: {metrics}")
        print(f"Controls per case range: {n_controls_per_case_range}")
        for ncc in n_controls_per_case_range:
            if ncc == 'Full':
                print(f"    n_controls_per_case=Full → standard RSF (no subsampling)")
            else:
                n_controls = ncc * n_events_total
                print(f"    n_controls_per_case={ncc} → ~{min(n_controls, n_non_events_total):.0f} "
                      f"controls + {n_events_total:.0f} cases per tree")
        print(f"Parameter Grid:")
        for param, values in param_grid.items():
            print(f"  - {param}: {values}")
        print(f"Total hyperparameter combinations: {total_combinations}")
        print(f"Folds per repetition: {n_splits}")
        print(f"Repetitions: {n_repeats}")
        print(f"Total evaluations: {total_combinations * n_splits * n_repeats}")
        print(f"{'='*80}\n")
    
    all_results = []
    current_iteration = 0
    
    for n_controls_per_case in n_controls_per_case_range:
        for param_values_tuple in model_param_combinations:
            params = dict(zip(param_names, param_values_tuple))
            current_iteration += 1
            
            if verbose:
                elapsed = time.time() - start_time
                ncc_str = str(n_controls_per_case) if n_controls_per_case != 'Full' else 'Full (no subsampling)'
                print(f"[{current_iteration}/{total_combinations}] "
                      f"n_controls_per_case={ncc_str}, params: {params}")
                if current_iteration > 1:
                    avg_time = elapsed / (current_iteration - 1)
                    remaining = avg_time * (total_combinations - current_iteration)
                    print(f"  Elapsed: {elapsed/60:.1f}min, Est. remaining: {remaining/60:.1f}min")
            
            try:
                if n_controls_per_case == 'Full':
                    # Standard RSF — no per-tree subsampling
                    from sksurv.ensemble import RandomSurvivalForest
                    cv_results, error_flags = ensemble_repeated_cross_validation(
                        model_class=RandomSurvivalForest,
                        X=X, y=y,
                        model_params=params,
                        metrics=metrics,
                        scale_data=scale_data,
                        scaler_type=scaler_type,
                        continuous_features=continuous_features,
                        random_state=random_state,
                        n_jobs=n_jobs,
                        n_controls_per_case=None,
                        dataset=None,
                        feature_columns=None,
                        n_splits=n_splits,
                        n_repeats=n_repeats,
                        base_random_state=base_random_state,
                        verbose=False
                    )
                else:
                    # Per-tree case-control subsampling
                    cv_results, error_flags = cc_rsf_repeated_cross_validation(
                        X=X, y=y,
                        model_params=params,
                        metrics=metrics,
                        n_controls_per_case=n_controls_per_case,
                        event_col_name=event_col_name,
                        scale_data=scale_data,
                        scaler_type=scaler_type,
                        continuous_features=continuous_features,
                        random_state=random_state,
                        n_jobs=n_jobs,
                        n_splits=n_splits,
                        n_repeats=n_repeats,
                        base_random_state=base_random_state,
                        verbose=verbose  # Pass through verbose flag
                    )
                
                # Store n_controls_per_case label
                ncc_label = n_controls_per_case if n_controls_per_case != 'Full' else 'Full'
                
                # Add parameter columns to results
                for idx, row in cv_results.iterrows():
                    result_row = params.copy()
                    result_row['n_controls_per_case'] = ncc_label
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
                    print(f"  Error: {str(e)}")
                
                ncc_label = n_controls_per_case if n_controls_per_case != 'Full' else 'Full'
                
                for split in all_splits:
                    result_row = params.copy()
                    result_row['n_controls_per_case'] = ncc_label
                    result_row['repeat_idx'] = split['repeat_idx']
                    result_row['fold_idx'] = split['fold_idx']
                    for metric in metrics:
                        result_row[metric] = np.nan
                    all_results.append(result_row)
    
    # Create detailed results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Create summary DataFrame using groupby with dropna=False
    param_cols = [col for col in results_df.columns 
                  if col not in ['repeat_idx', 'fold_idx'] + metrics]
    
    summary_rows = []
    for combo, group in results_df.groupby(param_cols, dropna=False):
        row = dict(zip(param_cols, combo if isinstance(combo, tuple) else [combo]))
        
        for metric in metrics:
            row[f'{metric}_mean'] = group[metric].mean()
            row[f'{metric}_std'] = group[metric].std()
            
            for repeat_idx in range(n_repeats):
                repeat_mask = group['repeat_idx'] == repeat_idx
                row[f'{metric}_repeat_{repeat_idx+1}_mean'] = group[repeat_mask][metric].mean()
            
            for repeat_idx in range(n_repeats):
                for fold_idx in range(n_splits):
                    fold_repeat_mask = (group['repeat_idx'] == repeat_idx) & (group['fold_idx'] == fold_idx)
                    if fold_repeat_mask.any():
                        row[f'{metric}_repeat_{repeat_idx+1}_fold_{fold_idx+1}'] = group[fold_repeat_mask][metric].values[0]
                    else:
                        row[f'{metric}_repeat_{repeat_idx+1}_fold_{fold_idx+1}'] = np.nan
        
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Best parameters
    param_cols_for_best = [col for col in summary_df.columns 
                           if not col.endswith('_mean') and not col.endswith('_std') 
                           and not col.startswith(tuple(metrics)) 
                           and '_repeat_' not in col and '_fold_' not in col]
    
    best_params = {}
    for metric in metrics:
        mean_col = f'{metric}_mean'
        valid_results = summary_df[~summary_df[mean_col].isna()].copy()
        
        if len(valid_results) == 0:
            if verbose:
                print(f"\nWarning: No valid results for {metric}")
            best_params[metric] = None
            continue
        
        if 'brier' in metric.lower():
            best_idx = valid_results[mean_col].idxmin()
        else:
            best_idx = valid_results[mean_col].idxmax()
        
        best_row = summary_df.loc[best_idx]
        best_param_dict = {col: best_row[col] for col in param_cols_for_best}
        
        best_params[metric] = {
            'params': best_param_dict,
            'mean': best_row[f'{metric}_mean'],
            'std': best_row[f'{metric}_std']
        }
    
    elapsed_time = time.time() - start_time
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"CC-RSF GRID SEARCH COMPLETE")
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


# =============================================================================
# ANALYSIS AND VISUALIZATION FUNCTIONS
# =============================================================================

def track_gbm_estimator_performance(model_class, X_train, y_train, X_val, y_val,
                                    model_params, metrics, 
                                    scale_data=False, scaler_type='standard',
                                    continuous_features=None,
                                    n_estimators_range=None,
                                    sample_every=10):
    """
    Track GBM model performance as the number of estimators increases to identify
    when performance plateaus.
    
    Parameters:
    -----------
    model_class : class
        GradientBoostingSurvivalAnalysis or similar model class
    X_train, y_train : Training data
    X_val, y_val : Validation data
    model_params : dict
        Model parameters (should include n_estimators)
    metrics : list
        List of metrics to track
    scale_data : bool
        Whether to scale the data
    scaler_type : str
        'standard' or 'minmax'
    continuous_features : list or None
        Features to scale (if None, scales all)
    n_estimators_range : list or None
        Specific estimator counts to evaluate. If None, samples from 1 to n_estimators
    sample_every : int
        If n_estimators_range is None, evaluate every N estimators
    
    Returns:
    --------
    results_df : DataFrame with estimator counts and metric values
    fig : matplotlib figure
    """
    import matplotlib.pyplot as plt
    from sksurv.metrics import concordance_index_censored, integrated_brier_score, cumulative_dynamic_auc
    
    # Scale data if requested
    if scale_data:
        if continuous_features is not None:
            if scaler_type == 'standard':
                scaler = ColumnTransformer([
                    ('continuous', StandardScaler(), continuous_features)
                ], remainder='passthrough')
            else:
                scaler = ColumnTransformer([
                    ('continuous', MinMaxScaler(), continuous_features)
                ], remainder='passthrough')
            
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            feature_names = continuous_features + [col for col in X_train.columns if col not in continuous_features]
            X_train = pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train.index)
            X_val = pd.DataFrame(X_val_scaled, columns=feature_names, index=X_val.index)
        else:
            if scaler_type == 'standard':
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()
            
            X_train = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_val = pd.DataFrame(
                scaler.transform(X_val),
                columns=X_val.columns,
                index=X_val.index
            )
    
    # Train model with full n_estimators
    n_estimators = model_params.get('n_estimators', 100)
    model = model_class(**model_params)
    model.fit(X_train, y_train)
    
    # Determine which estimator counts to evaluate
    if n_estimators_range is None:
        n_estimators_range = list(range(sample_every, n_estimators + 1, sample_every))
        if n_estimators not in n_estimators_range:
            n_estimators_range.append(n_estimators)
        n_estimators_range = sorted(n_estimators_range)
    
    # Track performance at each stage
    results = []
    
    # Use staged_predict if available
    if hasattr(model, 'staged_predict'):
        for i, y_pred in enumerate(model.staged_predict(X_val)):
            n_est = i + 1
            if n_est not in n_estimators_range:
                continue
            
            result_row = {'n_estimators': n_est}
            
            for metric in metrics:
                try:
                    if metric == 'c-index':
                        score = concordance_index_censored(
                            y_val['event'], y_val['time'], y_pred
                        )[0]
                        result_row[metric] = score
                    
                    elif metric == 'int-brier-score':
                        # For Brier score, we need survival functions
                        # Use a simple approximation or skip for staged prediction
                        result_row[metric] = np.nan
                        
                    elif metric == 'cumulative-dynamic-auc':
                        result_row[metric] = np.nan
                        
                except Exception as e:
                    result_row[metric] = np.nan
            
            results.append(result_row)
    else:
        # Fallback: retrain with different n_estimators
        for n_est in n_estimators_range:
            params_copy = model_params.copy()
            params_copy['n_estimators'] = n_est
            
            temp_model = model_class(**params_copy)
            temp_model.fit(X_train, y_train)
            
            result_row = {'n_estimators': n_est}
            
            for metric in metrics:
                try:
                    if metric == 'c-index':
                        y_pred = temp_model.predict(X_val)
                        score = concordance_index_censored(
                            y_val['event'], y_val['time'], y_pred
                        )[0]
                        result_row[metric] = score
                    
                    elif metric == 'int-brier-score':
                        # Calculate integrated Brier score
                        surv_funcs = temp_model.predict_survival_function(X_val)
                        times = np.percentile(y_train['time'][y_train['event']], np.linspace(5, 95, 15))
                        preds = np.asarray([[fn(t) for t in times] for fn in surv_funcs])
                        score = integrated_brier_score(y_train, y_val, preds, times)
                        result_row[metric] = score
                        
                    elif metric == 'cumulative-dynamic-auc':
                        # Calculate cumulative dynamic AUC
                        surv_funcs = temp_model.predict_survival_function(X_val)
                        times = np.percentile(y_train['time'][y_train['event']], np.linspace(10, 90, 10))
                        va_times, va_auc = cumulative_dynamic_auc(y_train, y_val, y_pred, times)
                        result_row[metric] = np.mean(va_auc)
                        
                except Exception as e:
                    result_row[metric] = np.nan
            
            results.append(result_row)
    
    results_df = pd.DataFrame(results)
    
    # Create plot
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        valid_data = results_df[~results_df[metric].isna()]
        
        if len(valid_data) > 0:
            ax.plot(valid_data['n_estimators'], valid_data[metric], 'o-', linewidth=2, markersize=4)
            ax.set_xlabel('Number of Estimators', fontsize=12)
            ax.set_ylabel(metric, fontsize=12)
            ax.set_title(f'{metric} vs Number of Estimators', fontsize=13)
            ax.grid(True, alpha=0.3)
            
            # Add horizontal line at best value
            if 'brier' in metric.lower():
                best_val = valid_data[metric].min()
                best_idx = valid_data[metric].idxmin()
            else:
                best_val = valid_data[metric].max()
                best_idx = valid_data[metric].idxmax()
            
            best_n_est = valid_data.loc[best_idx, 'n_estimators']
            ax.axhline(y=best_val, color='r', linestyle='--', alpha=0.5, 
                      label=f'Best: {best_val:.4f} @ {best_n_est} est.')
            ax.legend(fontsize=10)
        else:
            ax.text(0.5, 0.5, f'No valid data for {metric}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    return results_df, fig


RANKING_METRICS = (
    'c-index', 'c-index-ipcw', 'cumulative-dynamic-auc', 'cumulative-dynamic-prauc',
    'partial-likelihood'
)


def _ranking_metrics_from_pred(pred, y_train, y_val, metrics):
    """Compute risk-score-only metrics from a single prediction vector.

    Only metrics that need nothing but a risk score are supported (see
    RANKING_METRICS plus the 'auc-Xy' / 'prauc-Xy' time-point variants).
    Anything requiring predict_survival_function (Brier, IBS, log-loss) is
    returned as NaN, since staged boosting predictions give risk scores only.

    Definitions match _evaluate_ensemble_train_validation exactly so results
    stay comparable with the other grid searches in this module.
    """
    TIME_POINTS = {
        '6m': 0.5 * 365, '1y': 1 * 365, '2y': 2 * 365, '3y': 3 * 365,
        '4y': 4 * 365, '5y': 5 * 365, '10y': 10 * 365, '14y': 14 * 365
    }

    results = {metric: np.nan for metric in metrics}
    if pred is None:
        return results

    for metric in metrics:
        if metric == 'c-index':
            try:
                results[metric] = concordance_index_censored(y_val['event'], y_val['time'], pred)[0]
            except Exception:
                pass

        elif metric == 'c-index-ipcw':
            try:
                tau = min(y_train['time'].max(), y_val['time'].max()) * 0.95
                results[metric] = concordance_index_ipcw(y_train, y_val, pred, tau=tau)[0]
            except Exception:
                pass

        elif metric == 'cumulative-dynamic-auc':
            try:
                max_time = min(y_train['time'].max(), y_val['time'].max()) * 0.95
                times = np.percentile(y_train['time'], np.linspace(10, 90, 10))
                times = np.unique(times[(times > 0) & (times <= max_time)])
                if len(times) >= 2:
                    _, mean_auc = cumulative_dynamic_auc(y_train, y_val, pred, times)
                    results[metric] = mean_auc
            except Exception:
                pass

        elif metric.startswith('auc-'):
            try:
                time_key = metric.replace('auc-', '')
                if time_key in TIME_POINTS:
                    time_point = TIME_POINTS[time_key]
                    if time_point <= min(y_train['time'].max(), y_val['time'].max()):
                        auc_values, _ = cumulative_dynamic_auc(y_train, y_val, pred, [time_point])
                        results[metric] = auc_values[0]
            except Exception:
                pass

        elif metric == 'cumulative-dynamic-prauc':
            try:
                max_time = min(y_train['time'].max(), y_val['time'].max()) * 0.95
                times = np.percentile(y_train['time'], np.linspace(10, 90, 10))
                times = np.unique(times[(times > 0) & (times <= max_time)])
                if len(times) >= 2:
                    prauc_scores = []
                    for t in times:
                        valid_mask, y_binary, weights = get_ipcw_weights_for_prauc(y_train, y_val, t)
                        if valid_mask.sum() > 0 and y_binary[valid_mask].sum() > 0:
                            precision_vals, recall_vals, _ = precision_recall_curve(
                                y_binary[valid_mask], pred[valid_mask], sample_weight=weights[valid_mask]
                            )
                            prauc_scores.append(sklearn_auc(recall_vals, precision_vals))
                    if len(prauc_scores) > 0:
                        results[metric] = np.mean(prauc_scores)
            except Exception:
                pass

        elif metric.startswith('prauc-'):
            try:
                time_key = metric.replace('prauc-', '')
                if time_key in TIME_POINTS:
                    time_point = TIME_POINTS[time_key]
                    if time_point <= min(y_train['time'].max(), y_val['time'].max()):
                        valid_mask, y_binary, weights = get_ipcw_weights_for_prauc(y_train, y_val, time_point)
                        if valid_mask.sum() > 0 and y_binary[valid_mask].sum() > 0:
                            precision_vals, recall_vals, _ = precision_recall_curve(
                                y_binary[valid_mask], pred[valid_mask], sample_weight=weights[valid_mask]
                            )
                            results[metric] = sklearn_auc(recall_vals, precision_vals)
            except Exception:
                pass

        elif metric == 'partial-likelihood':
            try:
                sort_idx = np.argsort(y_val['time'])
                sorted_time = y_val['time'][sort_idx]
                sorted_event = y_val['event'][sort_idx]
                sorted_risk = pred[sort_idx]

                log_likelihood = 0.0
                for i in range(len(sorted_time)):
                    if sorted_event[i]:
                        risk_set_scores = sorted_risk[sorted_time >= sorted_time[i]]
                        log_likelihood += sorted_risk[i] - np.log(np.sum(np.exp(risk_set_scores)))
                results[metric] = log_likelihood
            except Exception:
                pass

    return results


def gbm_staged_repeated_grid_search(X, y, param_grid, metrics,
                                    n_estimators_grid,
                                    model_class=None,
                                    model_name='GBM (staged)',
                                    scale_data=False, scaler_type='standard',
                                    continuous_features=None,
                                    random_state=None,
                                    n_splits=5, n_repeats=10, base_random_state=42,
                                    verbose=True):
    """
    Repeated stratified CV grid search for boosting models where n_estimators is
    tuned for free via staged_predict.

    A boosting model with N estimators contains, as prefixes, every model with
    fewer estimators. So instead of refitting once per n_estimators value, this
    fits a single model at max(n_estimators_grid) per fold and reads the
    validation metrics off staged_predict at each requested checkpoint. Cost
    drops from len(n_estimators_grid) fits per fold to one.

    Uses the same generate_cv_splits() folds as cox_repeated_grid_search,
    subsampling_repeated_grid_search and ensemble_repeated_grid_search, so with
    matching n_splits/n_repeats/base_random_state the folds are identical and
    the scores are directly comparable across model families.

    Parameters:
    -----------
    X, y : Training data (DataFrame, structured survival array)
    param_grid : dict
        Grid over every hyperparameter EXCEPT n_estimators (which comes from
        n_estimators_grid). Passing 'n_estimators' here raises ValueError.
    metrics : list
        Risk-score-only metrics. See RANKING_METRICS and the 'auc-Xy' /
        'prauc-Xy' variants. Metrics needing predict_survival_function
        (Brier, IBS, r2-ibs, log-loss) are not available from staged
        predictions and raise ValueError.
    n_estimators_grid : list of int
        Boosting-round checkpoints to evaluate, e.g. [50, 100, 200, 500, 1000].
        The model is fitted once at max() of these per fold.
    model_class : class or None
        Boosting model exposing staged_predict. Defaults to
        GradientBoostingSurvivalAnalysis. ComponentwiseGradientBoosting-
        SurvivalAnalysis also works.
    scale_data, scaler_type, continuous_features :
        Same fold-internal scaling as ensemble_repeated_cross_validation.
        Not needed for tree-based boosting.
    random_state : int or None
        Passed to the model for reproducibility.
    n_splits, n_repeats, base_random_state :
        CV layout. Match the Cox run to keep folds identical.

    Returns:
    --------
    results_df, summary_df, best_params
        Same schema as ensemble_repeated_grid_search, with 'n_estimators' as an
        ordinary parameter column, so downstream helpers (clean_cox_data,
        plot_repeated_cv_results, ...) work unchanged. A KeyboardInterrupt
        (e.g. stopping the cell) is caught: whatever combinations finished
        before the interrupt are still returned instead of being lost.

    Note:
    -----
    dropout_rate > 0 rescales earlier trees as fitting proceeds, so a staged
    prefix is not identical to a model fitted with that many rounds. A warning
    is issued if dropout is used.
    """
    if model_class is None:
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis
        model_class = GradientBoostingSurvivalAnalysis

    if 'n_estimators' in param_grid:
        raise ValueError(
            "n_estimators must be passed via n_estimators_grid, not param_grid: "
            "it is evaluated from a single fit through staged_predict."
        )

    unsupported = [m for m in metrics
                   if m not in RANKING_METRICS
                   and not m.startswith('auc-') and not m.startswith('prauc-')]
    if unsupported:
        raise ValueError(
            f"Metrics {unsupported} need predict_survival_function, which staged "
            f"predictions do not provide. Supported: {list(RANKING_METRICS)} plus "
            f"'auc-<t>' / 'prauc-<t>'. Use ensemble_repeated_grid_search for "
            f"Brier/IBS-based selection."
        )

    n_estimators_grid = sorted(set(int(n) for n in n_estimators_grid))
    if min(n_estimators_grid) < 1:
        raise ValueError("n_estimators_grid values must be >= 1.")
    max_estimators = max(n_estimators_grid)
    checkpoints = set(n_estimators_grid)

    if any(param_grid.get('dropout_rate', [0])):
        warnings.warn(
            "dropout_rate > 0 rescales earlier trees during fitting, so staged "
            "prefixes are only an approximation of independently fitted models.",
            UserWarning
        )

    start_time = time.time()

    all_splits = generate_cv_splits(X, y, n_splits, n_repeats, base_random_state)

    param_names = list(param_grid.keys())
    param_combinations = list(product(*param_grid.values()))
    total_combinations = len(param_combinations)

    if verbose:
        print(f"\n{'='*80}")
        print(f"{model_name.upper()} STAGED REPEATED GRID SEARCH")
        print(f"{'='*80}")
        print(f"Model: {model_class.__name__}")
        print(f"Scaling: {'Yes (' + scaler_type + ')' if scale_data else 'No'}")
        print(f"Metrics: {metrics}")
        print(f"Parameter Grid (excluding n_estimators):")
        for param, values in param_grid.items():
            print(f"  - {param}: {values}")
        print(f"  - n_estimators (staged): {n_estimators_grid}")
        print(f"Base combinations (model fits per fold): {total_combinations}")
        print(f"Effective grid points: {total_combinations * len(n_estimators_grid)}")
        print(f"Folds per repetition: {n_splits}")
        print(f"Repetitions: {n_repeats}")
        print(f"Model fits: {total_combinations * n_splits * n_repeats} "
              f"(at n_estimators={max_estimators})")
        print(f"Evaluations: {total_combinations * len(n_estimators_grid) * n_splits * n_repeats}")
        print(f"{'='*80}\n")

    all_results = []
    interrupted_at = None

    try:
        for comb_idx, param_values_tuple in enumerate(param_combinations):
            params = dict(zip(param_names, param_values_tuple))

            if verbose:
                elapsed = time.time() - start_time
                print(f"[{comb_idx + 1}/{total_combinations}] Testing params: {params}")
                if comb_idx > 0:
                    avg_time = elapsed / comb_idx
                    print(f"  Elapsed: {elapsed/60:.1f}min, "
                          f"Est. remaining: {avg_time * (total_combinations - comb_idx)/60:.1f}min")

            for split in all_splits:
                repeat_idx, fold_idx = split['repeat_idx'], split['fold_idx']
                train_idx, val_idx = split['train_idx'], split['val_idx']

                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Scale inside the fold to avoid leakage (same logic as
                # ensemble_repeated_cross_validation).
                if scale_data:
                    scaler_cls = StandardScaler if scaler_type == 'standard' else MinMaxScaler
                    if continuous_features is not None:
                        scaler = ColumnTransformer(
                            [('continuous', scaler_cls(), continuous_features)],
                            remainder='passthrough'
                        )
                        feature_names = continuous_features + [
                            c for c in X_train.columns if c not in continuous_features
                        ]
                        X_train = pd.DataFrame(scaler.fit_transform(X_train),
                                               columns=feature_names, index=X_train.index)
                        X_val = pd.DataFrame(scaler.transform(X_val),
                                             columns=feature_names, index=X_val.index)
                    else:
                        scaler = scaler_cls()
                        X_train = pd.DataFrame(scaler.fit_transform(X_train),
                                               columns=X_train.columns, index=X_train.index)
                        X_val = pd.DataFrame(scaler.transform(X_val),
                                             columns=X_val.columns, index=X_val.index)

                model_params = dict(params)
                model_params['n_estimators'] = max_estimators
                if random_state is not None:
                    model_params['random_state'] = random_state

                fold_scores = {}
                try:
                    model = model_class(**model_params)
                    model.fit(X_train, y_train)

                    for stage_idx, stage_pred in enumerate(model.staged_predict(X_val), start=1):
                        if stage_idx in checkpoints:
                            fold_scores[stage_idx] = _ranking_metrics_from_pred(
                                stage_pred, y_train, y_val, metrics
                            )
                except Exception as e:
                    print(f"  Error in repeat {repeat_idx+1}, fold {fold_idx+1}: "
                          f"{type(e).__name__}: {str(e)}")

                for n_est in n_estimators_grid:
                    row = params.copy()
                    row['n_estimators'] = n_est
                    row['repeat_idx'] = repeat_idx
                    row['fold_idx'] = fold_idx
                    scores = fold_scores.get(n_est, {metric: np.nan for metric in metrics})
                    for metric in metrics:
                        row[metric] = scores.get(metric, np.nan)
                    all_results.append(row)

            if verbose:
                # Print every n_estimators checkpoint (not just the best) so
                # n_estimators reads as a grid axis in the console log --
                # this is the only surviving record of a combo if the run
                # gets interrupted before results_df is written out.
                combo_df = pd.DataFrame(all_results)
                combo_mask = np.ones(len(combo_df), dtype=bool)
                for k, v in params.items():
                    col = combo_df[k]
                    match = col.isna() if v is None else (col == v)
                    combo_mask &= match.to_numpy(dtype=bool)
                combo_df = combo_df[combo_mask]
                for metric in metrics:
                    per_n = combo_df.groupby('n_estimators')[metric].agg(['mean', 'std'])
                    per_n = per_n.reindex(n_estimators_grid)
                    best_n = per_n['mean'].idxmax() if per_n['mean'].notna().any() else None
                    print(f"  {metric} by n_estimators:")
                    for n_est, row in per_n.iterrows():
                        if pd.isna(row['mean']):
                            continue
                        marker = '  <- best' if n_est == best_n else ''
                        print(f"    n_estimators={n_est:<5} {row['mean']:.4f} "
                              f"± {row['std']:.4f}{marker}")
    except KeyboardInterrupt:
        interrupted_at = comb_idx + 1
        if verbose:
            print(f"\nInterrupted during combination {interrupted_at}/{total_combinations} "
                  f"-- returning results for the {interrupted_at - 1} combination(s) that "
                  f"finished, plus any folds already run on the one in progress.")

    results_df = pd.DataFrame(all_results)

    if results_df.empty:
        if verbose:
            print("\nInterrupted before any fold finished -- nothing to summarize.")
        return results_df, pd.DataFrame(), {metric: None for metric in metrics}

    param_cols = [c for c in results_df.columns
                  if c not in ['repeat_idx', 'fold_idx'] + list(metrics)]

    summary_rows = []
    for combo, group in results_df.groupby(param_cols, dropna=False):
        row = dict(zip(param_cols, combo if isinstance(combo, tuple) else [combo]))
        for metric in metrics:
            row[f'{metric}_mean'] = group[metric].mean()
            row[f'{metric}_std'] = group[metric].std()
            for repeat_idx in range(n_repeats):
                repeat_mask = group['repeat_idx'] == repeat_idx
                row[f'{metric}_repeat_{repeat_idx+1}_mean'] = group[repeat_mask][metric].mean()
            for repeat_idx in range(n_repeats):
                for fold_idx in range(n_splits):
                    mask = (group['repeat_idx'] == repeat_idx) & (group['fold_idx'] == fold_idx)
                    row[f'{metric}_repeat_{repeat_idx+1}_fold_{fold_idx+1}'] = (
                        group[mask][metric].values[0] if mask.any() else np.nan
                    )
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    best_params = {}
    for metric in metrics:
        mean_col = f'{metric}_mean'
        valid_results = summary_df[~summary_df[mean_col].isna()]
        if len(valid_results) == 0:
            if verbose:
                print(f"\nWarning: No valid results for {metric}")
            best_params[metric] = None
            continue

        best_idx = (valid_results[mean_col].idxmin() if 'brier' in metric.lower()
                    else valid_results[mean_col].idxmax())
        best_row = summary_df.loc[best_idx]
        best_params[metric] = {
            'params': {col: best_row[col] for col in param_cols},
            'mean': best_row[mean_col],
            'std': best_row[f'{metric}_std']
        }

    if verbose:
        print(f"\n{'='*80}")
        print(f"STAGED GRID SEARCH {'INTERRUPTED' if interrupted_at is not None else 'COMPLETE'}")
        print(f"{'='*80}")
        print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
        if interrupted_at is not None:
            print(f"Stopped during combination {interrupted_at}/{total_combinations} -- "
                  f"best parameters below are based on partial results only.")
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


def analyze_repeated_cv_stability(results_df, metrics, n_repeats):
    """
    Analyze the stability of repeated CV results across repetitions.

    Parameters:
    -----------
    results_df : DataFrame
        Detailed results from any repeated grid search function
    metrics : list
        List of metrics to analyze
    n_repeats : int
        Number of repetitions
    
    Returns:
    --------
    stability_df : DataFrame with stability metrics per hyperparameter combination
    """
    # Get unique hyperparameter combinations
    param_cols = [col for col in results_df.columns 
                  if col not in ['repeat_idx', 'fold_idx'] + metrics]
    
    stability_rows = []
    
    for _, group in results_df.groupby(param_cols):
        row = {col: group[col].iloc[0] for col in param_cols}
        
        for metric in metrics:
            # Calculate mean per repeat
            repeat_means = []
            for repeat_idx in range(n_repeats):
                repeat_mask = group['repeat_idx'] == repeat_idx
                repeat_means.append(group[repeat_mask][metric].mean())
            
            row[f'{metric}_mean'] = np.mean(repeat_means)
            row[f'{metric}_std_across_repeats'] = np.std(repeat_means)
            row[f'{metric}_cv_coefficient'] = np.std(repeat_means) / np.mean(repeat_means) if np.mean(repeat_means) != 0 else np.nan
            row[f'{metric}_range'] = max(repeat_means) - min(repeat_means)
        
        stability_rows.append(row)
    
    stability_df = pd.DataFrame(stability_rows)
    return stability_df


def plot_repeated_cv_results(results_df, metric, param_col, n_repeats, title_prefix=""):
    """
    Plot repeated CV results showing variability across repetitions.
    
    Parameters:
    -----------
    results_df : DataFrame
        Detailed results from repeated grid search
    metric : str
        Metric to plot
    param_col : str
        Parameter column to use for x-axis
    n_repeats : int
        Number of repetitions
    title_prefix : str
        Prefix for plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Get unique parameter values
    param_values = sorted(results_df[param_col].unique())
    
    # Plot 1: Mean ± std across all folds and repeats
    ax = axes[0]
    means = []
    stds = []
    for val in param_values:
        subset = results_df[results_df[param_col] == val]
        means.append(subset[metric].mean())
        stds.append(subset[metric].std())
    
    ax.errorbar(range(len(param_values)), means, yerr=stds, fmt='o-', capsize=5)
    ax.set_xticks(range(len(param_values)))
    ax.set_xticklabels([f'{v:.2e}' if isinstance(v, float) else str(v) for v in param_values], rotation=45)
    ax.set_xlabel(param_col)
    ax.set_ylabel(metric)
    ax.set_title(f'{title_prefix}{metric} vs {param_col} (Mean ± Std)')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Boxplot showing distribution across repeats
    ax = axes[1]
    data_for_boxplot = []
    labels = []
    for val in param_values:
        subset = results_df[results_df[param_col] == val]
        # Get mean per repeat
        repeat_means = []
        for repeat_idx in range(n_repeats):
            repeat_mask = subset['repeat_idx'] == repeat_idx
            repeat_means.append(subset[repeat_mask][metric].mean())
        data_for_boxplot.append(repeat_means)
        labels.append(f'{val:.2e}' if isinstance(val, float) else str(val))
    
    ax.boxplot(data_for_boxplot, labels=labels)
    ax.set_xlabel(param_col)
    ax.set_ylabel(f'{metric} (mean per repeat)')
    ax.set_title(f'{title_prefix}Distribution of {metric} across {n_repeats} repeats')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def compare_single_vs_repeated_cv(single_cv_results, repeated_cv_summary, metrics):
    """
    Compare results from single CV vs repeated CV.
    
    Parameters:
    -----------
    single_cv_results : DataFrame
        Results from single CV run
    repeated_cv_summary : DataFrame
        Summary from repeated CV
    metrics : list
        List of metrics to compare
    
    Returns:
    --------
    comparison_df : DataFrame with comparison statistics
    """
    comparison_rows = []
    
    for metric in metrics:
        single_mean_col = f'{metric}_mean' if f'{metric}_mean' in single_cv_results.columns else 'mean'
        single_std_col = f'{metric}_std' if f'{metric}_std' in single_cv_results.columns else 'std'
        
        repeated_mean_col = f'{metric}_mean'
        repeated_std_col = f'{metric}_std'
        
        if single_mean_col in single_cv_results.columns and repeated_mean_col in repeated_cv_summary.columns:
            comparison_rows.append({
                'metric': metric,
                'single_cv_best_mean': single_cv_results[single_mean_col].max() if 'brier' not in metric.lower() else single_cv_results[single_mean_col].min(),
                'single_cv_avg_std': single_cv_results[single_std_col].mean(),
                'repeated_cv_best_mean': repeated_cv_summary[repeated_mean_col].max() if 'brier' not in metric.lower() else repeated_cv_summary[repeated_mean_col].min(),
                'repeated_cv_avg_std': repeated_cv_summary[repeated_std_col].mean(),
            })
    
    comparison_df = pd.DataFrame(comparison_rows)
    return comparison_df
