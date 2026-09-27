import os
import sys

from new_approaches.models.workstation.functions.preproc import build_manual_stratified_group_folds

# Ensure the script runs relative to its own directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from functions.repeated_cv import *

from sksurv.linear_model import CoxnetSurvivalAnalysis

# --- Load data ---
dataset = pd.read_csv("data.csv")

diagnosis_variables = []
with open("diagnosis.txt", "r") as file:
    for line in file:
        diagnosis_variables.append(line.strip())

treatment_variables = []
with open("treatment.txt", "r") as file:
    for line in file:
        treatment_variables.append(line.strip())

survival_dataset = dataset.drop('Unnamed: 0', axis=1)

feature_columns = treatment_variables + diagnosis_variables

# Cap survival days at 1825 and set event status to 0 for censored cases
survival_dataset.loc[survival_dataset['Survival days'] > 1826, 'Regional LN recurrence'] = 0
survival_dataset.loc[survival_dataset['Survival days'] > 1826, 'Survival days'] = 1826

# Prepare data
data_x = survival_dataset[feature_columns]
data_y = Surv.from_arrays(
    event=survival_dataset['Regional LN recurrence'],
    time=survival_dataset['Survival days']
)
groups = survival_dataset['PERSON_ID']

# Outcome-Length stratification: time bins only for recurrence cases
n_splits = 5
max_time_bins = 5
event_col = survival_dataset['Regional LN recurrence'].astype(int)
time_col = survival_dataset['Survival days']

strata_labels = None
chosen_bins = None
recurrence_mask = (event_col == 1)

for n_bins in range(max_time_bins, 1, -1):
    try:
        # Build quantile bins using only recurrence cases
        recurrence_time_bins = pd.qcut(
            time_col.loc[recurrence_mask], q=n_bins, labels=False, duplicates='drop'
        )

        # Non-recurrence rows share one label; recurrence rows get time-bin labels
        candidate_labels = pd.Series('0_noevent', index=survival_dataset.index, dtype='object')
        candidate_labels.loc[recurrence_mask] = (
            '1_' + recurrence_time_bins.astype(int).astype(str)
        ).values

        # Every stratum must have at least n_splits samples for stable balancing
        if candidate_labels.value_counts().min() >= n_splits:
            strata_labels = candidate_labels
            chosen_bins = int(recurrence_time_bins.nunique())
            break
    except ValueError:
        continue

if strata_labels is None:
    # Fallback to outcome-only stratification when recurrence-time bins are too sparse
    strata_labels = event_col.astype(str)
    print('Fallback: using outcome-only stratification (recurrence time bins too sparse).')
else:
    print(f'Using recurrence-only time stratification with {chosen_bins} time bins.')

# sgkf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=12)
# cv = sgkf.split(X=data_x, y=strata_labels, groups=groups)

# train_index, test_index = next(cv)

manual_splits, manual_fold_summary = build_manual_stratified_group_folds(
    event_series=event_col,
    group_series=groups,
    strata_series=strata_labels,
    keep_groups_for_events=False,
    n_splits=n_splits,
    random_state=12
)

# Keep variable name for compatibility in later cells
cv = manual_splits
train_index, test_index = manual_splits[1]

x_train = data_x.iloc[train_index]
y_train = data_y[train_index]

x_test = data_x.iloc[test_index]
y_test = data_y[test_index]

subsampling = survival_dataset.iloc[train_index].copy()

param_grid = {
    'n_estimators': [700, 1000, 1300],
    'max_depth': [7, 10, 15],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [1, 3],
    'max_features': ['sqrt']
}

metrics = ['c-index', 'int-brier-score', 'c-index-ipcw']

results_df, summary_df, best_params = cc_rsf_repeated_grid_search(
    X=x_train, y=y_train,
    param_grid=param_grid,
    metrics=metrics,
    n_controls_per_case_range=["Full"],
    model_name='CC-RSF',
    event_col_name='event',
    random_state=42, n_jobs=-1,
    n_splits=4, n_repeats=5, base_random_state=42,
    verbose=True
)

summary_df.to_csv('rsf.csv', index=False)
print("Done. Results saved to rsf.csv")