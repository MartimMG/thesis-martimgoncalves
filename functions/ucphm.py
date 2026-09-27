import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sksurv.util import Surv
from sklearn.model_selection import StratifiedGroupKFold
from sksurv.metrics import (concordance_index_censored, concordance_index_ipcw, integrated_brier_score)
from sksurv.linear_model import CoxPHSurvivalAnalysis
from scipy import stats
from scipy.stats import norm

def univariate_cox_analysis(data_x, data_y, groups, n_splits):
    # fits a separate one-feature Cox model per column, so we get a rough
    # per-feature signal (C-index, IBS, HR) before any multivariate selection

    univariate_results = []
    event_indicator = [y[0] for y in data_y]  # for stratification
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=12)

    print(f"Starting univariate analysis of {len(data_x.columns)} features...")
    print(f"Using {n_splits}-fold stratified group cross-validation")
    print("-" * 60)

    for i, feature in enumerate(data_x.columns):
        print(f"[{i+1}/{len(data_x.columns)}] Analyzing: {feature}")

        fold_results = {
            "cindex_scores": [],
            "coefficients": [],
            "brier_scores": [],
        }

        for fold, (train_index, test_index) in enumerate(sgkf.split(data_x, event_indicator, groups)):
            try:
                x_train_single = data_x[feature].iloc[train_index].values.reshape(-1, 1)
                x_test_single = data_x[feature].iloc[test_index].values.reshape(-1, 1)

                y_train_surv = [data_y[i] for i in train_index]
                y_test_surv = [data_y[i] for i in test_index]

                event_train = [e for e, t in y_train_surv]
                time_train = [t for e, t in y_train_surv]
                event_test = [e for e, t in y_test_surv]
                time_test = [t for e, t in y_test_surv]

                # can't get an IBS estimate out of a fold with only one test time
                if np.min(time_test) == np.max(time_test):
                    print(f"     Skipping Brier score: single time point in fold {fold}")
                    continue

                y_train = Surv.from_arrays(event=event_train, time=time_train)
                y_test = Surv.from_arrays(event=event_test, time=time_test)

                cox_model = CoxPHSurvivalAnalysis()
                cox_model.fit(x_train_single, y_train)

                fold_results["coefficients"].append(cox_model.coef_[0])  # log-hazard ratio

                risk_scores = cox_model.predict(x_test_single)
                cindex = concordance_index_censored(event_test, time_test, risk_scores)[0]
                fold_results["cindex_scores"].append(cindex)

                surv_funcs = cox_model.predict_survival_function(x_test_single)
                # keep IBS times inside the observed range or sksurv complains
                times = np.linspace(
                    max(np.min(time_train), np.min(time_test)) + 1e-6,
                    min(np.max(time_train), np.max(time_test)) - 1e-6,
                    100
                )

                surv_preds = np.asarray([[fn(t) for t in times] for fn in surv_funcs])
                if np.isnan(surv_preds).any():
                    print(f"     Skipping Brier score: NaNs in survival predictions in fold {fold}")
                    continue

                ibs = integrated_brier_score(y_train, y_test, surv_preds, times)
                fold_results["brier_scores"].append(ibs)

            except Exception as e:
                print(f"     Warning: Model fitting failed in fold {fold}: {e}")
                continue

        if fold_results["cindex_scores"]:
            mean_cindex = np.mean(fold_results["cindex_scores"])
            std_cindex = np.std(fold_results["cindex_scores"])

            mean_brier = np.mean(fold_results["brier_scores"]) if fold_results["brier_scores"] else np.nan
            std_brier = np.std(fold_results["brier_scores"]) if fold_results["brier_scores"] else np.nan

            mean_coef = np.mean(fold_results["coefficients"])
            std_coef = np.std(fold_results["coefficients"])

            # HR + 95% CI from the fold-averaged coefficient
            hr = np.exp(mean_coef)
            hr_lower = np.exp(mean_coef - 1.96 * std_coef)
            hr_upper = np.exp(mean_coef + 1.96 * std_coef)

            # Wald z-test on the mean coefficient, using the across-fold spread as SE
            se_mean_coef = std_coef / np.sqrt(len(fold_results["coefficients"]))
            wald_z = mean_coef / se_mean_coef
            p_value = 2 * norm.sf(abs(wald_z)) # sf = survival function = 1 - cdf

            univariate_results.append({
                "Feature": feature,
                "C_Index_Mean": mean_cindex,
                "C_Index_Std": std_cindex,
                "Integrated_Brier_Score": mean_brier,
                "Brier_Score_Std": std_brier,
                "Coefficient_Mean": mean_coef,
                "Coefficient_Std": std_coef,
                "Hazard_Ratio": hr,
                "HR_95_CI_Lower": hr_lower,
                "HR_95_CI_Upper": hr_upper,
                "P_Value_of_HR": p_value, 
                "N_Successful_Folds": len(fold_results["cindex_scores"]),
            })

            print(f"     C-index: {mean_cindex:.4f} ± {std_cindex:.4f}")
            print(f"     Integrated Brier score: {mean_brier:.4f} ± {std_brier:.4f}")
            print(f"     HR: {hr:.4f} (95% CI: {hr_lower:.4f}–{hr_upper:.4f}), P-value: {p_value:.4f}") # Updated print
        else:
            print(f"     Failed to fit any models for {feature}")

    results_df = pd.DataFrame(univariate_results)
    return results_df


# only makes sense for the hazard-ratio metric (needs HR + CI columns)
def plot_hazard(dataset, filename, highlight_features=None, highlight_color='red', default_color='blue'):
    if highlight_features is None:
        highlight_features = []

    dataset_sorted = dataset.sort_values(by='Hazard_Ratio')

    dataset_sorted['yerr_lower'] = dataset_sorted['Hazard_Ratio'] - dataset_sorted['HR_95_CI_Lower']
    dataset_sorted['yerr_upper'] = dataset_sorted['HR_95_CI_Upper'] - dataset_sorted['Hazard_Ratio']
    yerr_values = dataset_sorted[['yerr_lower', 'yerr_upper']].values.T

    colors = [
        highlight_color if feat in highlight_features else default_color 
        for feat in dataset_sorted['Feature']
    ]

    plt.figure(figsize=(10, 6))

    for i, (x, y, yerr, c) in enumerate(zip(
        dataset_sorted['Feature'], 
        dataset_sorted['Hazard_Ratio'], 
        yerr_values.T, 
        colors
    )):
        plt.errorbar(x, y, 
                     yerr=[[yerr[0]], [yerr[1]]], 
                     fmt='o', 
                     capsize=5, 
                     color=c)
        
    plt.axhline(y=1, color='gray', linestyle='--', linewidth=1)

    plt.ylabel('Hazard Ratio')
    plt.xlabel('Feature')
    plt.title('Hazard Ratio for Different Features')
    plt.xticks(rotation=45, ha='right')  

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.grid(False)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def plot_cindex(dataset, filename, highlight_features=None, highlight_color='red', default_color='blue'):
    if highlight_features is None:
        highlight_features = []

    dataset_sorted = dataset.sort_values(by='C_Index_Mean')

    yerr_values = dataset_sorted['C_Index_Std'].values

    colors = [
        highlight_color if feat in highlight_features else default_color
        for feat in dataset_sorted['Feature']
    ]

    plt.figure(figsize=(10, 6))

    for x, y, yerr, c in zip(
        dataset_sorted['Feature'],
        dataset_sorted['C_Index_Mean'],
        yerr_values,
        colors
    ):
        plt.errorbar(x, y, 
                     yerr=[[yerr], [yerr]], 
                     fmt='o', 
                     capsize=5, 
                     color=c)

    plt.ylabel('C-Index (Mean)')
    plt.xlabel('Feature')
    plt.title('C-Index for Different Features')
    plt.xticks(rotation=45, ha='right')

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.grid(False)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def plot_brier(dataset, filename, highlight_features=None, highlight_color='red', default_color='blue'):
    if highlight_features is None:
        highlight_features = []

    dataset_sorted = dataset.sort_values(by='Integrated_Brier_Score')
    yerr_values = dataset_sorted['Brier_Score_Std'].values
    colors = [
        highlight_color if feat in highlight_features else default_color
        for feat in dataset_sorted['Feature']
    ]

    plt.figure(figsize=(10, 6))

    for x, y, yerr, c in zip(
        dataset_sorted['Feature'],
        dataset_sorted['Integrated_Brier_Score'],
        yerr_values,
        colors
    ):
        plt.errorbar(x, y, 
                     yerr=[[yerr], [yerr]], 
                     fmt='o', 
                     capsize=5, 
                     color=c)

    plt.ylabel('Integrated Brier Score (Mean)')
    plt.xlabel('Feature')
    plt.title('Integrated Brier Score for Different Features')
    plt.xticks(rotation=45, ha='right')

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.grid(False)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
