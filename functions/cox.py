import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, auc as sklearn_auc, log_loss
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.metrics import concordance_index_ipcw
from sksurv.metrics import integrated_brier_score, brier_score
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv
import pandas as pd

def plot_coefficients(coefs):
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

def get_ipcw_weights_for_prauc(y_train, y_test, eval_time, max_weight=20.0):
    """
    IPCW weights for PR-AUC at a fixed eval_time.

    Differs from the usual ROC-AUC IPCW weighting in one important way:
    samples censored before eval_time are dropped entirely rather than
    downweighted. We have no way of knowing whether they'd have had the event
    by eval_time, so keeping them (even at low weight) just biases the PR
    curve. Events get weight 1/G(Ti); survivors past eval_time get 1/G(eval_time).

    Returns (valid_mask, y_binary, weights) — only entries with valid_mask=True
    carry a meaningful y_binary/weight.
    """
    from sksurv.nonparametric import kaplan_meier_estimator

    # censoring distribution G(t) = P(C > t)
    cens_time, cens_surv = kaplan_meier_estimator(
        ~y_train['event'],  # censoring indicator (True if censored)
        y_train['time']
    )
    
    weights = np.zeros(len(y_test))
    valid_mask = np.zeros(len(y_test), dtype=bool)
    y_binary = np.zeros(len(y_test), dtype=int)
    
    for i, (event, time) in enumerate(zip(y_test['event'], y_test['time'])):
        if event and time <= eval_time:
            # true positive: event happened by eval_time, weight 1/G(Ti)
            valid_mask[i] = True
            y_binary[i] = 1

            idx = np.searchsorted(cens_time, time) - 1
            g_ti = cens_surv[idx] if idx >= 0 else 1.0
            weights[i] = min(1.0 / g_ti, max_weight) if g_ti > 0 else 0.0

        elif time > eval_time:
            # true negative: still event-free at eval_time, weight 1/G(eval_time)
            valid_mask[i] = True
            y_binary[i] = 0

            idx = np.searchsorted(cens_time, eval_time) - 1
            g_eval = cens_surv[idx] if idx >= 0 else 1.0
            weights[i] = min(1.0 / g_eval, max_weight) if g_eval > 0 else 0.0

        # else: censored before eval_time -> excluded, valid_mask stays False
        # (unknowable whether they'd have had the event by eval_time)

    return valid_mask, y_binary, weights

def cox_model_evaluation(X_train, y_train, X_test, y_test, alpha, l1_ratio, metrics):
    """
    Fit a Coxnet model and score it on the test set for each metric in `metrics`.

    Metric strings follow one pattern throughout this module: 'c-index',
    'c-index-ipcw', 'int-brier-score', 'deviance', 'partial-likelihood',
    'r2-ibs', 'log-loss', plus time-indexed variants like 'brier-5y', 'auc-5y',
    'prauc-5y', 'r2-brier-5y', 'log-loss-5y' (suffixes: 6m/1y/2y/3y/4y/5y/10y/14y).
    Anything not computable at that horizon comes back as NaN rather than raising.

    Returns (results dict of metric -> score, list of features with nonzero coef).
    """
    model = CoxnetSurvivalAnalysis(alphas=[alpha], l1_ratio=l1_ratio, fit_baseline_model=True)
    model.fit(X_train, y_train)

    nonzero_features = [f for f, c in zip(model.feature_names_in_, model.coef_.flatten()) if c != 0]

    predicted_risk_scores = model.predict(X_test)

    # y_train/y_test are already sksurv structured arrays, nothing to convert

    results = {}
    
    for metric in metrics:
        if metric == 'c-index':
            score = concordance_index_censored(y_test['event'], y_test['time'], predicted_risk_scores)[0]
            results[metric] = score
            
        elif metric == 'c-index-ipcw':
            # Add tau parameter to avoid censoring warnings
            max_time = min(y_train['time'].max(), y_test['time'].max()) * 0.9
            score = concordance_index_ipcw(y_train, y_test, predicted_risk_scores, tau=max_time)[0]
            results[metric] = score
            
        elif metric == 'int-brier-score':
            # percentile rule for the eval grid: 5th-90th pct of train times
            surv_funcs = model.predict_survival_function(X_test)
            max_time = min(y_train['time'].max(), y_test['time'].max()) * 0.9

            times = np.percentile(y_train['time'], np.linspace(5, 90, 10))
            times = np.unique(times)
            times = times[(times > 0) & (times <= max_time)]

            if len(times) > 1:  # need at least 2 points for IBS
                predicted_survival_probabilities = np.asarray(
                    [[fn(t) for t in times] for fn in surv_funcs]
                )
                score = integrated_brier_score(y_train, y_test, predicted_survival_probabilities, times)
                results[metric] = score
            else:
                results[metric] = np.nan
        
        elif metric.startswith('brier-'):
            # Brier score at one horizon, e.g. 'brier-5y'
            surv_funcs = model.predict_survival_function(X_test)
            max_time = min(y_train['time'].max(), y_test['time'].max()) # * 0.9

            time_points = {
                'brier-6m': 0.5 * 365,
                'brier-1y': 1 * 365,
                'brier-2y': 2 * 365,
                'brier-3y': 3 * 365,
                'brier-4y': 4 * 365,
                'brier-5y': 5 * 365,
                'brier-10y': 10 * 365,
                'brier-14y': 14 * 365
            }
            
            if metric in time_points:
                time_point = time_points[metric]
                if time_point <= max_time:
                    try:
                        # For single time point, use brier_score (not integrated_brier_score)
                        predicted_survival_probabilities = np.asarray(
                            [fn(time_point) for fn in surv_funcs]
                        )
                        score = float(brier_score(y_train, y_test, predicted_survival_probabilities, time_point)[1][0])
                        results[metric] = score
                    except:
                        results[metric] = np.nan
                else:
                    results[metric] = np.nan
            else:
                results[metric] = np.nan
        
        elif metric == 'cumulative-dynamic-auc':
            # mean of the time-dependent AUC over the 10th-90th pct grid
            try:
                from sksurv.metrics import cumulative_dynamic_auc
                max_time = min(y_train['time'].max(), y_test['time'].max()) * 0.9
                times = np.percentile(y_train['time'], np.linspace(10, 90, 10))
                times = np.unique(times)
                times = times[(times > 0) & (times <= max_time)]

                if len(times) >= 2:
                    auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, predicted_risk_scores, times)
                    results[metric] = mean_auc
                else:
                    results[metric] = np.nan
            except Exception as e:
                results[metric] = np.nan
        
        elif metric.startswith('auc-'):
            # time-dependent AUC at one horizon
            try:
                from sksurv.metrics import cumulative_dynamic_auc
                max_time = min(y_train['time'].max(), y_test['time'].max()) # * 0.9
                
                time_points = {
                    'auc-6m': 0.5 * 365,
                    'auc-1y': 1 * 365,
                    'auc-2y': 2 * 365,
                    'auc-3y': 3 * 365,
                    'auc-4y': 4 * 365,
                    'auc-5y': 5 * 365,
                    'auc-10y': 10 * 365,
                    'auc-14y': 14 * 365
                }
                
                if metric in time_points:
                    time_point = time_points[metric]
                    if time_point <= max_time:
                        # cumulative_dynamic_auc still wants a list even for one point
                        auc_values, _ = cumulative_dynamic_auc(
                            y_train, y_test, predicted_risk_scores, [time_point]
                        )
                        results[metric] = auc_values[0]
                    else:
                        results[metric] = np.nan
                else:
                    results[metric] = np.nan
            except Exception as e:
                results[metric] = np.nan
        
        elif metric == 'r2-ibs':
            # Index of Prediction Proficiency: how much IBS improves vs. the
            # unconditional KM curve used as a "null model"
            try:
                surv_funcs = model.predict_survival_function(X_test)
                max_time = min(y_train['time'].max(), y_test['time'].max()) * 0.95
                times = np.percentile(y_train['time'], np.linspace(5, 90, 10))
                times = np.unique(times)
                times = times[(times > 0) & (times <= max_time)]

                if len(times) > 1:
                    model_surv_probs = np.asarray([[fn(t) for t in times] for fn in surv_funcs])
                    ibs_model = integrated_brier_score(y_train, y_test, model_surv_probs, times)

                    # same KM curve applied to every test subject, as the null
                    km_time, km_surv = kaplan_meier_estimator(
                        y_train['event'], y_train['time']
                    )
                    km_surv_at_times = np.interp(times, km_time, km_surv, left=1.0, right=km_surv[-1])
                    km_surv_probs = np.tile(km_surv_at_times, (len(y_test), 1))
                    ibs_null = integrated_brier_score(y_train, y_test, km_surv_probs, times)

                    if ibs_null > 0:
                        r2_ibs = 1.0 - (ibs_model / ibs_null)
                        results[metric] = r2_ibs
                    else:
                        results[metric] = np.nan
                else:
                    results[metric] = np.nan
            except Exception as e:
                results[metric] = np.nan
        
        elif metric.startswith('r2-brier-'):
            # same idea as r2-ibs but at a single horizon instead of integrated
            try:
                surv_funcs = model.predict_survival_function(X_test)
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
                        model_surv_probs = np.asarray([fn(time_point) for fn in surv_funcs])
                        bs_model = float(brier_score(y_train, y_test, model_surv_probs, time_point)[1][0])

                        km_time, km_surv = kaplan_meier_estimator(y_train['event'], y_train['time'])
                        km_surv_at_time = np.interp(time_point, km_time, km_surv, left=1.0, right=km_surv[-1])
                        km_surv_probs = np.full(len(y_test), km_surv_at_time)
                        bs_null = float(brier_score(y_train, y_test, km_surv_probs, time_point)[1][0])

                        results[metric] = 1.0 - (bs_model / bs_null) if bs_null > 0 else np.nan
                    else:
                        results[metric] = np.nan
                else:
                    results[metric] = np.nan
            except Exception as e:
                results[metric] = np.nan
        
        elif metric == 'cumulative-dynamic-prauc':
            # mean PR-AUC over the same time grid, reusing the IPCW helper above
            try:
                max_time = min(y_train['time'].max(), y_test['time'].max()) * 0.95
                times = np.percentile(y_train['time'], np.linspace(10, 90, 10))
                times = np.unique(times[(times > 0) & (times <= max_time)])
                
                if len(times) >= 2:
                    prauc_scores = []
                    for t in times:
                        valid_mask, y_binary, weights = get_ipcw_weights_for_prauc(y_train, y_test, t)
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
                results[metric] = np.nan
        
        elif metric.startswith('prauc-'):
            # PR-AUC at one horizon
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
                
                if metric in time_points:
                    time_point = time_points[metric]
                    if time_point <= max_time:
                        valid_mask, y_binary, weights = get_ipcw_weights_for_prauc(y_train, y_test, time_point)
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
                results[metric] = np.nan
                
        elif metric == 'deviance':
            # negative log partial likelihood, computed by hand from the risk
            # scores (lower = better fit). Same calc as 'partial-likelihood'
            # below, just negated -- kept as two metrics for convenience.
            risk_scores = model.predict(X_test)

            sort_idx = np.argsort(y_test['time'])
            sorted_time = y_test['time'][sort_idx]
            sorted_event = y_test['event'][sort_idx]
            sorted_risk = risk_scores[sort_idx]

            log_likelihood = 0.0
            for i in range(len(sorted_time)):
                if sorted_event[i]:
                    # risk set = everyone still at risk at this event time
                    risk_set_mask = sorted_time >= sorted_time[i]
                    risk_set_scores = sorted_risk[risk_set_mask]

                    log_likelihood += sorted_risk[i] - np.log(np.sum(np.exp(risk_set_scores)))

            deviance_score = -log_likelihood
            results[metric] = deviance_score

        elif metric == 'partial-likelihood':
            # higher is better here (unlike deviance above)
            try:
                risk_scores = model.predict(X_test)

                sort_idx = np.argsort(y_test['time'])
                sorted_time = y_test['time'][sort_idx]
                sorted_event = y_test['event'][sort_idx]
                sorted_risk = risk_scores[sort_idx]

                log_likelihood = 0.0
                for i in range(len(sorted_time)):
                    if sorted_event[i]:
                        risk_set_mask = sorted_time >= sorted_time[i]
                        risk_set_scores = sorted_risk[risk_set_mask]

                        log_likelihood += sorted_risk[i] - np.log(np.sum(np.exp(risk_set_scores)))

                results[metric] = log_likelihood
            except Exception as e:
                results[metric] = np.nan

        elif metric == 'log-loss':
            # log loss averaged over a grid of times, treating 1-S(t) as the
            # predicted probability of the event by t
            try:
                surv_funcs = model.predict_survival_function(X_test)
                max_time = min(y_train['time'].max(), y_test['time'].max()) * 0.9
                times = np.percentile(y_train['time'], np.linspace(10, 90, 10))
                times = np.unique(times[(times > 0) & (times <= max_time)])

                if len(times) >= 2:
                    log_losses = []
                    for t in times:
                        y_binary = ((y_test['time'] <= t) & y_test['event']).astype(int)
                        pred_proba = 1.0 - np.array([fn(t) for fn in surv_funcs])
                        pred_proba = np.clip(pred_proba, 1e-15, 1 - 1e-15)  # avoid log(0)

                        # skip if this time slice only has one class
                        if len(np.unique(y_binary)) >= 2:
                            log_losses.append(log_loss(y_binary, pred_proba))
                    
                    if len(log_losses) > 0:
                        results[metric] = np.mean(log_losses)
                    else:
                        results[metric] = np.nan
                else:
                    results[metric] = np.nan
            except Exception as e:
                results[metric] = np.nan
        
        elif metric.startswith('log-loss-'):
            # same as 'log-loss' but pinned to one horizon
            try:
                surv_funcs = model.predict_survival_function(X_test)
                max_time = min(y_train['time'].max(), y_test['time'].max())
                
                time_points = {
                    'log-loss-6m': 0.5 * 365,
                    'log-loss-1y': 1 * 365,
                    'log-loss-2y': 2 * 365,
                    'log-loss-3y': 3 * 365,
                    'log-loss-4y': 4 * 365,
                    'log-loss-5y': 5 * 365,
                    'log-loss-10y': 10 * 365,
                    'log-loss-14y': 14 * 365
                }
                
                if metric in time_points:
                    time_point = time_points[metric]
                    if time_point <= max_time:
                        y_binary = ((y_test['time'] <= time_point) & y_test['event']).astype(int)
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
                results[metric] = np.nan
    
    return results, nonzero_features


def cox_cross_validation(X, y, cv_folds, alpha, l1_ratio, metrics, scaler_class, binary_features=None, continuous_features=None):
    """
    CV for a single (alpha, l1_ratio) pair. X should be passed in unscaled --
    scaling happens inside each fold here, not before, so no fold sees info
    from its own validation split.

    Returns a DataFrame: folds as rows, metrics as columns, plus Mean/Std rows.
    """

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    if scaler_class == 'standard':
        scaler = StandardScaler()
    elif scaler_class == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_class == 'mixed':
        scaler = ColumnTransformer([
            ('binary', MinMaxScaler(), binary_features),
            ('continuous', StandardScaler(), continuous_features)
        ])
    else:
        raise ValueError(f"Unknown scaling method: {scaler_class}")
    
    n_folds = cv_folds.get_n_splits()

    results = {metric: [] for metric in metrics}

    for fold_idx, (train_idx, val_idx) in enumerate(cv_folds.split(X, y['event'])):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y[train_idx]  # structured array, direct indexing works
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y[val_idx]

        # fit the scaler on this fold's train split only
        X_train_fold_scaled = scaler.fit_transform(X_train_fold)
        X_val_fold_scaled = scaler.transform(X_val_fold)

        # back to DataFrame so cox_model_evaluation still gets feature names
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
        
        fold_results, _ = cox_model_evaluation(
            X_train_fold_scaled, y_train_fold, X_val_fold_scaled, y_val_fold,
            alpha, l1_ratio, metrics
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

# Create summary DataFrames for each metric
def create_alpha_summary(results_dict, metric):
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
    if higher_is_better:
        best_row = summary_df.loc[summary_df['Mean'].idxmax()]
    else:
        best_row = summary_df.loc[summary_df['Mean'].idxmin()]
    
    print(f"Best alpha for {metric_name}: {best_row['Alpha']:.4f}")
    print(f"Score: {best_row['Mean']:.4f} ± {best_row['Std']:.4f}")
    return best_row['Alpha']



def plot_cox_results(results_df, metric, title_prefix=""):
    """Scatter of per-fold scores for `metric`, with mean +/- std band."""
    fold_data = results_df.loc[results_df.index.str.startswith('Fold_')]

    plt.figure(figsize=(10, 6))

    mean_value = results_df.loc['Mean', metric]
    std_value = results_df.loc['Std', metric]

    fold_values = fold_data[metric]
    fold_numbers = range(1, len(fold_values) + 1)
    
    plt.scatter(fold_numbers, fold_values, alpha=0.6, s=50, label=f'{metric} (individual folds)')
    plt.axhline(y=mean_value, color='red', linestyle='--', label=f'Mean: {mean_value:.3f}')
    plt.fill_between([0.5, len(fold_values) + 0.5], 
                     mean_value - std_value, mean_value + std_value, 
                     alpha=0.2, color='red', label=f'±1 Std: {std_value:.3f}')
    
    plt.xlabel('Fold')
    plt.ylabel(f'{metric}')
    plt.title(f'{title_prefix}{metric} Across Folds')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_cindex_model5(df):

    plt.figure(figsize=(10, 6))

    best_idx = df['mean'].idxmax()
    
    plt.errorbar(df['Alphas'], df['mean'], yerr=df['std'], fmt='o', capsize=5, label='C-index (mean)', color='tab:cyan')
    
    plt.errorbar(df['Alphas'][best_idx], df['mean'][best_idx], yerr=df['std'][best_idx], 
                 fmt='o', capsize=5, label='Best C-index', color='tab:pink', markersize=10)

    plt.xscale('log')  
    plt.ylabel('C-index (Mean)')
    plt.xlabel('Lambdas (log scale)')
    plt.title('C-index (Mean) for Different Lambdas (log scale)')
    plt.xticks(rotation=45, ha='right')  
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(False)  

    plt.legend()  
    plt.tight_layout()  
        
    plt.show()


def plot_ipcw_model5(df):

    plt.figure(figsize=(10, 6))

    best_idx = df['mean'].idxmax()
    
    plt.errorbar(df['Alphas'], df['mean'], yerr=df['std'], fmt='o', capsize=5, label='IPCW C-index (mean)', color='tab:cyan')
    
    plt.errorbar(df['Alphas'][best_idx], df['mean'][best_idx], yerr=df['std'][best_idx], 
                 fmt='o', capsize=5, label='Best IPCW C-index', color='tab:pink', markersize=10)

    plt.xscale('log')  
    plt.ylabel('IPCW C-index (Mean)')
    plt.xlabel('Lambdas (log scale)')
    plt.title('IPCW C-index (Mean) for Different Lambdas (log scale)')
    plt.xticks(rotation=45, ha='right')  
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(False)  

    plt.legend()  
    plt.tight_layout()  
        
    plt.show()


def plot_IBS_model5(df):

    plt.figure(figsize=(10, 6))
    
    plt.errorbar(df['Alphas'], df['mean'], yerr=df['std'], fmt='o', capsize=5, label='IBS (mean)')

    plt.xscale('log') 
    plt.ylabel('IBS (Mean)')
    plt.xlabel('Lambdas (log scale)')
    plt.title('IBS (Mean) for Different Lambdas (log scale)')
    plt.xticks(rotation=45, ha='right')  
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(False)  

    plt.tight_layout() 
        
    plt.show()


def cox_grid_search(X, y, cv_folds, 
                   alpha_range, l1_ratio_range,
                   metrics, scaler_class='standard',
                   binary_features=None, continuous_features=None,
                   verbose=True):
    """
    Grid search over (alpha, l1_ratio) for the Coxnet model, running
    cox_cross_validation at every combination. X should be unscaled -- see
    cox_cross_validation for why.

    metrics is typically something like ['c-index', 'c-index-ipcw',
    'int-brier-score', 'deviance']. scaler_class/binary_features/
    continuous_features get passed straight through to cox_cross_validation.

    Returns (results: nested {l1_ratio: {alpha: results_df}},
    best_params: {metric: {l1_ratio, alpha, mean, std}}, summary_df: long-form
    table of every combination x metric).
    """
    from itertools import product
    import time

    results = {}
    all_results = []

    total_iterations = len(l1_ratio_range) * len(alpha_range)
    current_iteration = 0

    start_time = time.time()

    if verbose:
        print(f"{'='*80}")
        print(f"COX MODEL GRID SEARCH")
        print(f"{'='*80}")
        print(f"L1 ratios: {l1_ratio_range}")
        print(f"Alphas: {len(alpha_range)} values from {min(alpha_range):.2e} to {max(alpha_range):.2e}")
        print(f"Total combinations: {total_iterations}")
        print(f"Scaler: {scaler_class}")
        print(f"{'='*80}\n")

    for l1_ratio in l1_ratio_range:
        results[l1_ratio] = {}

        for alpha in alpha_range:
            current_iteration += 1

            if verbose:
                elapsed = time.time() - start_time
                avg_time = elapsed / current_iteration if current_iteration > 0 else 0
                remaining = avg_time * (total_iterations - current_iteration)
                print(f"[{current_iteration}/{total_iterations}] "
                      f"l1={l1_ratio:.3f}, alpha={alpha:.2e} "
                      f"(elapsed: {elapsed/60:.1f}min, est. remaining: {remaining/60:.1f}min)")

            try:
                results_df = cox_cross_validation(
                    X, y, cv_folds, alpha, l1_ratio, metrics,
                    scaler_class, binary_features, continuous_features
                )

                results[l1_ratio][alpha] = results_df

                for metric in metrics:
                    all_results.append({
                        'l1_ratio': l1_ratio,
                        'alpha': alpha,
                        'metric': metric,
                        'mean': results_df.loc['Mean', metric],
                        'std': results_df.loc['Std', metric]
                    })
            except Exception as e:
                if verbose:
                    print(f"  Error: {str(e)}")
                # combination failed to fit -- record NaN so the grid stays complete
                for metric in metrics:
                    all_results.append({
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
        metric_data = metric_data.dropna(subset=['mean'])

        if len(metric_data) == 0:
            if verbose:
                print(f"\nWarning: No valid results for {metric}")
            best_params[metric] = {
                'l1_ratio': np.nan,
                'alpha': np.nan,
                'mean': np.nan,
                'std': np.nan
            }
            continue

        # brier/deviance: lower is better. everything else: higher is better
        if 'brier' in metric.lower() or 'deviance' in metric.lower():
            best_idx = metric_data['mean'].idxmin()
        else:
            best_idx = metric_data['mean'].idxmax()
        
        best_row = metric_data.loc[best_idx]
        best_params[metric] = {
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
            if not np.isnan(params['mean']):
                print(f"\n{metric}:")
                print(f"  l1_ratio: {params['l1_ratio']:.4f}")
                print(f"  alpha: {params['alpha']:.4e}")
                print(f"  Score: {params['mean']:.4f} ± {params['std']:.4f}")
        print(f"\n{'='*80}\n")
    
    return results, best_params, summary_df


def plot_cox_grid_search_results(summary_df, metric='c-index'):
    """Heatmap + line plot of a grid search metric over (alpha, l1_ratio); summary_df comes from cox_grid_search."""
    import matplotlib.pyplot as plt

    metric_data = summary_df[summary_df['metric'] == metric].copy()
    metric_data = metric_data.dropna(subset=['mean'])

    if len(metric_data) == 0:
        print(f"No valid data to plot for {metric}")
        return

    l1_ratio_values = sorted(metric_data['l1_ratio'].unique())
    n_plots = len(l1_ratio_values)

    if n_plots == 0:
        print(f"No valid l1_ratio values to plot for {metric}")
        return

    # only worth a heatmap if there's more than one l1_ratio to compare
    if n_plots > 1:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        ax = axes[0]
        pivot_data = metric_data.pivot_table(
            values='mean', 
            index='l1_ratio', 
            columns='alpha', 
            aggfunc='first'
        )
        
        im = ax.imshow(pivot_data, aspect='auto', cmap='viridis')
        ax.set_xlabel('Alpha')
        ax.set_ylabel('L1 Ratio')
        ax.set_title(f'{metric} Heatmap')
        
        ax.set_xticks(range(len(pivot_data.columns)))
        ax.set_xticklabels([f'{a:.2e}' for a in pivot_data.columns], rotation=45, ha='right')
        ax.set_yticks(range(len(pivot_data.index)))
        ax.set_yticklabels([f'{r:.3f}' for r in pivot_data.index])
        
        plt.colorbar(im, ax=ax, label='Mean Score')

        ax = axes[1]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for l1_ratio in l1_ratio_values:
        l1_data = metric_data[metric_data['l1_ratio'] == l1_ratio].sort_values('alpha')
        ax.errorbar(l1_data['alpha'], l1_data['mean'], yerr=l1_data['std'],
                   marker='o', capsize=3, label=f'l1={l1_ratio:.3f}')
    
    ax.set_xscale('log')
    ax.set_xlabel('Alpha (log scale)')
    ax.set_ylabel(f'{metric}')
    ax.set_title(f'{metric} vs Alpha')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()