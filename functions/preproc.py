import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import shapiro, skew

#Separate categorical and numerical variables
def type_separation(data_code, treatment_variables):
    numerical_data = []
    categorical_data = []
    for i in range(len(data_code)):
        row = data_code.iloc[i]
        if row.iloc[0] in treatment_variables:
            if row.iloc[2] == 'Categorical':
                categorical_data.append(row.iloc[0])
            elif row.iloc[2] == 'Numerical':
                numerical_data.append(row.iloc[0])
    
    return numerical_data, categorical_data

#Numerical descriptive statistics functions
def get_counts(data_set, variables):
    return data_set[variables].count().to_dict()

def get_means(data_set, variables):
    return data_set[variables].mean().to_dict()

def get_std(data_set, variables):
    return data_set[variables].std().to_dict()

def get_max(data_set, variables):
    return data_set[variables].max().to_dict()

def get_median(data_set, variables):
    return data_set[variables].median().to_dict()

def get_min(data_set, variables):
    return data_set[variables].min().to_dict()

def get_percentiles(data_set, variables):
    percentiles = {}
    for var in variables:
        percentiles[var] = np.percentile(data_set[var].dropna(), [25, 50, 75])
    return percentiles

def numerical_descriptive_statistics(data_set, variables):
    stats = {
        'Counts': get_counts(data_set, variables),
        'Mean': get_means(data_set, variables),
        'Median': get_median(data_set, variables),
        'Sd': get_std(data_set, variables),
        'Max': get_max(data_set, variables),
        'Min': get_min(data_set, variables),
        'Percentiles': get_percentiles(data_set, variables)
    }
    df = pd.DataFrame(stats)
    percentiles_df = pd.DataFrame(df['Percentiles'].tolist(), 
                                 index=df.index,
                                 columns=['25th', '50th', '75th'])
    
    final_df = pd.concat([
        df.drop('Percentiles', axis=1),
        percentiles_df
    ], axis=1)

    return final_df

#Categorical descriptive statistics functions
def get_number_of_classes(data_set, variables):
    return data_set[variables].nunique().to_dict()

def get_top_class(data_set, variables):
    return data_set[variables].mode().to_dict()

def get_frequency_of_top_class(data_set, variables):
    return data_set[variables].apply(lambda x: x.value_counts().iloc[0]).to_dict()

def categorical_descriptive_statistics(data_set, variables):
    stats = {
        'Counts': get_counts(data_set, variables),
        'Number of Classes': get_number_of_classes(data_set, variables),
        'Top Class': get_top_class(data_set, variables),
        'Frequency of Top Class': get_frequency_of_top_class(data_set, variables)
    }
    return pd.DataFrame(stats)

#Visualization the distributions of categorical variables
def pie_chart(data_set, categorical_data, cols=2, figsize=(15, 10)):
    rows = int(np.ceil(len(categorical_data) / cols))
    plt.figure(figsize=figsize)
    
    for i, column in enumerate(categorical_data, 1):
        plt.subplot(rows, cols, i)
        counts = data_set[column].value_counts()
        counts.plot.pie(
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
            textprops={'fontsize': 8}
        )
        plt.title(f'Distribution of {column}', pad=20)
        plt.ylabel('')
    
    plt.tight_layout()
    plt.savefig("pie_charts.png", bbox_inches='tight', dpi=120)
    plt.close()
    
    return("Saved.")

#Visualization the distributions of numerical variables
def histogram(data_set, numerical_data, cols=2, figsize=(15, 10)):
    rows = int(np.ceil(len(numerical_data) / cols))
    plt.figure(figsize=figsize)

    for i, column in enumerate(numerical_data, 1):
        plt.subplot(rows, cols, i)

        data_set[column].hist(bins='auto', edgecolor='black') 
        plt.title(f'Histogram of {column}')
        plt.ylabel('Frequency')
        plt.xlabel('Value')

    plt.tight_layout()
    plt.savefig("histograms.png", bbox_inches='tight', dpi=120)
    plt.close()
    
    return("Saved.")

def box_plot(data_set, numerical_data, cols=2, figsize=(15, 10)):
    rows = int(np.ceil(len(numerical_data) / cols))
    plt.figure(figsize=figsize)

    for i, column in enumerate(numerical_data, 1):
        plt.subplot(rows, cols, i)
        data_set.boxplot(column=column)
        plt.title(column)
        plt.ylabel('Value')
        plt.xlabel('')
    
    plt.tight_layout()
    plt.savefig("boxplots.png", bbox_inches='tight', dpi=120)
    plt.close()
    
    return("Saved.")

# To see the missing values and their percentage
def missing_values(dataset, variables):
    missing_counts = dataset[variables].isnull().sum()
    
    missing_percentage = (missing_counts / len(dataset)) * 100
    
    summary_df = pd.DataFrame({
        'Column': dataset[variables].columns,
        'Missing Values': missing_counts,
        'Percentage Missing': missing_percentage
    }).reset_index(drop=True)
    
    return summary_df

def patient_missing_values(dataset, variables):
    missing_vaules = dataset[variables].isnull()
    rows_with_missing = missing_vaules.any(axis=1)
    
    patient_ids = dataset.iloc[rows_with_missing.values, 0]
    
    return patient_ids

# To replace the missing values with zero
def replace_missing_values(dataset, variables, value):
    dataset[variables] = dataset[variables].fillna(value)
    return print('Done')

# To replace the missing values with the mean of the column
def replace_missing_values_with_mean(dataset, variables):
    dataset[variables] = dataset[variables].fillna(dataset[variables].mean())
    return print('Done')

def simplify_variable(dataset, variables, max_value):
    value = max_value - 1
    for variable in variables:
        dataset[variable] = dataset[variable].apply(
            lambda x: str(x) if x < max_value else f'>{value}'
        )
    return print('Done')

# for these variables the first is if the treatment was done, and the second was when it was not done by patient refusal
def encode(data_set, variables, drop_first):
    encoder = OneHotEncoder(sparse_output=False, drop=drop_first)
    
    for variable in variables:
        encoded_data = encoder.fit_transform(data_set[[variable]])
        feature_names = encoder.get_feature_names_out([variable])
        encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=data_set.index)
        data_set = pd.concat([data_set, encoded_df], axis=1)
        data_set = data_set.drop(columns=[variable])
        
    return data_set

# regarding the biological drugs variable, the first is if the treatment was done, and the second was when it was Trastuzumab (0) or another drug (1)
def encode_bio_therapy(data_set, variable):
    data_set['Adjuvant therapy using biological drugs'] = data_set[variable].apply(lambda x: 1 if x in [1, 2] else 0)

    encoder = OneHotEncoder(sparse_output=False, drop='first')
    performed_data = data_set[data_set['Adjuvant therapy using biological drugs'] == 1]
    encoded_therapy_type = encoder.fit_transform(performed_data[[variable]])
    feature_names = ['Trastuzumab or other drugs']
    encoded_df = pd.DataFrame(encoded_therapy_type, columns=feature_names, index=performed_data.index)
    data_set = data_set.join(encoded_df, how='left')
    data_set['Trastuzumab or other drugs'] = data_set['Trastuzumab or other drugs'].fillna(0)
    data_set = data_set.drop(columns=[variable]) 

    return data_set


def one_hot_encoding(dataset, columns_to_encode, handle_unknown='ignore', drop_first=None):

    # Make a copy to avoid modifying the original DataFrame directly
    dataset_copy = dataset.copy()

    # Initialize the encoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown=handle_unknown, drop=drop_first)

    # Fit and transform the specified columns
    # We pass df_copy[columns_to_encode] to ensure only these columns are used for fitting
    encoded_array = encoder.fit_transform(dataset_copy[columns_to_encode])

    # Get the new column names
    # input_features should be the names of the columns passed to fit_transform
    new_column_names = encoder.get_feature_names_out(input_features=columns_to_encode)

    # Create a DataFrame from the encoded array
    encoded_df = pd.DataFrame(encoded_array, columns=new_column_names, index=dataset_copy.index)

    # Drop the original columns that were encoded
    df_processed = dataset_copy.drop(columns=columns_to_encode)

    # Concatenate the original (non-encoded) columns with the new encoded columns
    df_processed = pd.concat([df_processed, encoded_df], axis=1)

    return df_processed, new_column_names

def two_to_zero(dataset, variables):
    for variable in variables:
        dataset[variable] = dataset[variable].replace(2, 0)
    return 'Done'

def two_to_one(dataset, variables):
    for variable in variables:
        dataset[variable] = dataset[variable].replace(2, 1)
    return 'Done'

# to fill the empty numerical data with the median
def median_imputer(dataset, variables):
    for variable in variables:
        dataset[variable] = dataset[variable].fillna(dataset[variable].median())
    return 'Done'

# to create a column with the date of diagnosis
def date_of_diagnosis(dataset, variables):
    dataset['Date of diagnosis'] = pd.NaT

    for i in range(len(dataset)):
        try:
            date_found = False
            if pd.isna(dataset.loc[i, 'Date of diagnosis']):
                for col in variables:
                    if pd.notna(dataset.loc[i, col]): 
                        dataset.loc[i, 'Date of diagnosis'] = dataset.loc[i, col]
                        date_found = True
                        break 
                dataset.loc[i, 'Date of diagnosis'] = pd.to_datetime(dataset.loc[i, 'Date of diagnosis'], errors='coerce')
            
            if not date_found and pd.isna(dataset.loc[i, 'Date of diagnosis']):
                dataset.loc[i, 'Date of diagnosis'] = pd.NaT

        except Exception as e:
            print(f"Error processing row {i}: {e}")
            dataset.loc[i, 'Date of diagnosis'] = pd.NaT
            continue

    return dataset


# to remove rows with 0 in date of diagnosis
def remove_zeros_na(dataset, variables):
    for variable in variables:
        dataset = dataset[dataset[variable] != 0]
        dataset = dataset[dataset[variable] != '0']
        dataset = dataset[dataset[variable] != '0.0']
        dataset = dataset[dataset[variable].notna()]

    dataset.reset_index(drop=True, inplace=True)
    return dataset

def remove_na(dataset, variables):
    for variable in variables:
        dataset = dataset[dataset[variable].notna()]
    return dataset

# to convert to a standard date
def format_date(dataset, variables):
    
    for variable in variables:
        dataset[variable] = pd.to_datetime(dataset[variable], errors='coerce').dt.date

    return dataset

# to calculate the survival days
def survival_days(dataset, diagnosis, last_treatment):
    dataset = dataset.copy()
    
    dataset['Survival days'] = np.nan
    
    for i in range(len(dataset)):
        date1 = dataset.loc[i, diagnosis[0]]
        date2 = dataset.loc[i, last_treatment[0]]
            
        survival_days = (date2 - date1).days
            
        dataset.at[i, 'Survival days'] = survival_days
    
    return dataset

# to remove patients with negative survival days
def tte_negative_patients(dataset):
    indexes_tte_neg = []
    for i in range(len(dataset)):
        if dataset.iloc[i]['Survival days'] <= 0: 
            indexes_tte_neg.append(i)
    dataset = dataset.drop(indexes_tte_neg)
    return dataset

# to filter for only EBC (early breast cancer)
def filter_EBC(dataset):
    dataset = dataset[dataset['EBC'] == 1]
    return dataset

def patients_multiple_lesions_recurrent (dataset):
    multiple_lesions_dataset = dataset[dataset.duplicated('PERSON_ID', keep=False)]
    recurrence_multiple_lesions_dataset = multiple_lesions_dataset[multiple_lesions_dataset['Regional LN recurrence'] == 1]
    recurrence_multiple_lesions_dataset.to_excel('Multiple lesions and recurrence.xlsx', index=False)

def change_to_numeric(dataset, variables):
    dataset_copy = dataset.copy()  
    for variable in variables:
        dataset_copy[variable] = dataset_copy[variable].astype('float64')
    return print('Done')

def change_to_categorical(dataset, variables):
    dataset_copy = dataset.copy()  
    for variable in variables:
        dataset_copy[variable] = dataset_copy[variable].astype('category')
    return print('Done')


def test_normality(dataset, variables):
    results = []

    for variable in variables:
        data = dataset[variable].dropna()
        shapiro_stat, p_value = shapiro(data)
        skewness = skew(data)
        
        # Determine normality
        normal = "Probably normal" if p_value > 0.05 else "Probably not normal"

        # Detect outliers using IQR method
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        has_outliers = ((data < lower_bound) | (data > upper_bound)).any()
        
        results.append({
            'Variable': variable,
            'Skewness': skewness,
            'Shapiro p-value': p_value,
            'Normality': normal,
            'Has Outliers': has_outliers
        })

    return pd.DataFrame(results)

def log_transform(dataset, variables):
    for variable in variables:
        dataset[variable] = np.log1p(dataset[variable])
    return 'Done'

def minus_one(dataset, variables):
    for variable in variables:
        dataset[variable] = dataset[variable].fillna(-1)
    return dataset

def minus_one_and_new(dataset, variables):
    for variable in variables:
        # Create a flag column for missing values
        flag_col = f"{variable}_was_missing"
        # Identify missing values before filling
        was_missing = dataset[variable].isnull()
        # Fill missing values with -1
        dataset[variable] = dataset[variable].fillna(-1)
        # Flag where missing values were filled
        dataset[flag_col] = was_missing.astype(int)
    return dataset

def columns_with_missing(df: pd.DataFrame, columns):
    """
    Return:
      - a list of column names (from the given columns) that have missing values
      - a Series with number of missing values per column
    """
    col_list = list(columns)

    # Missing count for selected columns
    missing_counts = df[col_list].isnull().sum()

    return missing_counts[missing_counts > 0]

def one_hot_encode_with_missing_indicator(dataset, variables, drop_first=False):
    """
    One-hot encode variables that have missing values.

    drop_first=True also drops the smallest category (never the 404 placeholder,
    which is always numerically largest), so missing patients end up indistinguishable
    from that dropped reference category rather than getting their own all-zero row.

    Returns (dataset_copy, new_column_names).
    """
    dataset_copy = dataset.copy()
    all_new_columns = []
    drop = 'first' if drop_first else None

    for variable in variables:
        missing_mask = dataset_copy[variable].isnull()
        dataset_copy[variable] = dataset_copy[variable].fillna(404)  # placeholder so OHE has something to encode

        encoder = OneHotEncoder(sparse_output=False, drop=drop, handle_unknown='ignore')
        encoded_array = encoder.fit_transform(dataset_copy[[variable]])
        feature_names = encoder.get_feature_names_out([variable])
        encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=dataset_copy.index)

        # placeholder column shouldn't survive into the feature set
        col_404 = f"{variable}_404"
        col_404_float = f"{variable}_404.0"
        if col_404 in encoded_df.columns:
            encoded_df = encoded_df.drop(columns=[col_404])
        elif col_404_float in encoded_df.columns:
            encoded_df = encoded_df.drop(columns=[col_404_float])

        # missing rows get all-zero instead of a spurious 404 dummy
        encoded_df.loc[missing_mask, :] = 0

        all_new_columns.extend(encoded_df.columns.tolist())

        dataset_copy = dataset_copy.drop(columns=[variable])
        dataset_copy = pd.concat([dataset_copy, encoded_df], axis=1)

    return dataset_copy, all_new_columns

# function to fill the specified columns with the mode of the column
def fill_with_mode(dataset, variables):
    for variable in variables:
        mode_value = dataset[variable].mode(dropna=True).iloc[0]
        dataset[variable] = dataset[variable].fillna(mode_value)
    return 'Done'


def build_manual_stratified_group_folds(
    event_series,
    group_series,
    n_splits=5,
    random_state=12,
    strata_series=None,
    keep_groups_for_events=True,
):
    """Build folds with two-stage balancing and optional group relaxation for events.

    group_series keeps a given group in a single fold (patients with multiple
    lesions shouldn't leak across train/valid). strata_series is what gets balanced
    across folds -- defaults to event_series itself if not given, but you'd pass
    e.g. outcome-time bins here to also spread those evenly.

    keep_groups_for_events=False breaks group-integrity for event rows specifically
    (assigns them one at a time) while still respecting groups for censored rows --
    useful when event groups are so large/few that keeping them intact makes it
    impossible to balance event counts across folds.

    Returns (splits, summary) where splits is a list of (train_idx, valid_idx) and
    summary is a small DataFrame reporting how close each fold landed to its target
    event/sample counts.
    """
    rng = np.random.default_rng(random_state)

    event_series = pd.Series(event_series).reset_index(drop=True)
    group_series = pd.Series(group_series).reset_index(drop=True)
    if len(event_series) != len(group_series):
        raise ValueError('event_series and group_series must have the same length.')

    if strata_series is None:
        strata_series = event_series.astype(str)
    else:
        strata_series = pd.Series(strata_series).astype(str).reset_index(drop=True)
        if len(strata_series) != len(group_series):
            raise ValueError('strata_series and group_series must have the same length.')

    temp = pd.DataFrame({
        'group': group_series.to_numpy(),
        'event': event_series.to_numpy().astype(int),
        'strata': strata_series.to_numpy()
    })

    group_stats = temp.groupby('group').agg(n_samples=('event', 'size'), n_events=('event', 'sum'))
    n_groups = len(group_stats)
    if n_groups < 2:
        raise ValueError('Need at least 2 unique groups to build grouped folds.')

    if n_groups < n_splits:
        print(f'Warning: only {n_groups} unique groups available; reducing n_splits from {n_splits} to {n_groups}.')
        n_splits = n_groups

    total_samples = len(temp)
    target_size = total_samples / n_splits
    all_indices = np.arange(len(temp))
    fold_valid_lists = [[] for _ in range(n_splits)]
    fold_sizes = np.zeros(n_splits, dtype=int)
    fold_events = np.zeros(n_splits, dtype=int)

    event_rows = temp[temp['event'] == 1]
    event_strata_levels = sorted(event_rows['strata'].unique().tolist())
    strata_to_pos = {s: i for i, s in enumerate(event_strata_levels)}
    fold_event_strata_counts = np.zeros((n_splits, len(event_strata_levels)), dtype=int)
    total_event_strata_counts = event_rows['strata'].value_counts().reindex(event_strata_levels, fill_value=0).to_numpy(dtype=float)
    target_event_strata = total_event_strata_counts / n_splits if len(event_strata_levels) > 0 else np.array([])

    total_events = int(event_series.sum())
    target_events = total_events / n_splits if n_splits > 0 else 0
    base_e, rem_e = divmod(total_events, n_splits)
    target_events_int = np.array([base_e + (1 if i < rem_e else 0) for i in range(n_splits)], dtype=int)

    if keep_groups_for_events:
        # Stage 1A: assign event groups while keeping groups intact.
        event_group_strata = pd.crosstab(event_rows['group'], event_rows['strata']).reindex(columns=event_strata_levels, fill_value=0)
        event_groups = group_stats[group_stats['n_events'] > 0].index.to_numpy().copy()
        rng.shuffle(event_groups)
        event_groups = sorted(
            event_groups,
            key=lambda g: (int(group_stats.loc[g, 'n_events']), int(group_stats.loc[g, 'n_samples'])),
            reverse=True,
        )

        group_to_indices = temp.groupby('group').indices

        def score_event_group(fold_idx, g_size, g_events, g_event_strata):
            if len(event_strata_levels) > 0:
                strata_after = fold_event_strata_counts[fold_idx] + g_event_strata
                strata_err = np.sum(np.abs(strata_after - target_event_strata) / np.maximum(target_event_strata, 1.0))
            else:
                strata_err = 0.0
            event_err = abs((fold_events[fold_idx] + g_events) - target_events) / max(target_events, 1.0)
            size_err = abs((fold_sizes[fold_idx] + g_size) - target_size) / max(target_size, 1.0)
            return strata_err + 0.5 * event_err + 0.2 * size_err

        assigned_groups = set()
        for grp in event_groups:
            g_idx = np.asarray(group_to_indices[grp], dtype=int)
            g_size = int(len(g_idx))
            g_events = int(group_stats.loc[grp, 'n_events'])
            if len(event_strata_levels) > 0 and grp in event_group_strata.index:
                g_event_strata = event_group_strata.loc[grp, event_strata_levels].to_numpy(dtype=int)
            else:
                g_event_strata = np.zeros(len(event_strata_levels), dtype=int)

            scores = np.array([score_event_group(f, g_size, g_events, g_event_strata) for f in range(n_splits)])
            min_score = scores.min()
            candidate_folds = np.where(np.isclose(scores, min_score))[0]
            best_fold = int(candidate_folds[np.argmin(fold_sizes[candidate_folds])])

            fold_valid_lists[best_fold].extend(g_idx.tolist())
            fold_sizes[best_fold] += g_size
            fold_events[best_fold] += g_events
            if len(event_strata_levels) > 0:
                fold_event_strata_counts[best_fold] += g_event_strata
            assigned_groups.add(grp)

        # Stage 2A: assign remaining groups by size only.
        remaining_groups = [g for g in group_stats.index if g not in assigned_groups]
        rng.shuffle(remaining_groups)
        remaining_groups = sorted(remaining_groups, key=lambda g: int(group_stats.loc[g, 'n_samples']), reverse=True)
        for grp in remaining_groups:
            g_idx = np.asarray(group_to_indices[grp], dtype=int)
            best_fold = int(np.argmin(fold_sizes))
            fold_valid_lists[best_fold].extend(g_idx.tolist())
            fold_sizes[best_fold] += len(g_idx)
    else:
        # Stage 1B: assign event rows individually (ignore grouping for events).
        event_idx = event_rows.index.to_numpy(dtype=int)
        rng.shuffle(event_idx)

        if len(event_strata_levels) > 0:
            strata_counts = event_rows['strata'].value_counts().to_dict()
            rarity = {s: 1.0 / max(strata_counts.get(s, 1), 1) for s in event_strata_levels}
            event_idx = sorted(event_idx, key=lambda i: rarity.get(temp.at[i, 'strata'], 0.0), reverse=True)

        for idx in event_idx:
            label = temp.at[idx, 'strata']
            label_pos = strata_to_pos.get(label, None)
            # Prefer folds that are still below their integer event targets.
            under_target_folds = np.where(fold_events < target_events_int)[0]
            candidate_folds = under_target_folds if len(under_target_folds) > 0 else np.arange(n_splits)

            scores = []
            for f in candidate_folds:
                event_err = abs((fold_events[f] + 1) - target_events) / max(target_events, 1.0)
                size_err = abs((fold_sizes[f] + 1) - target_size) / max(target_size, 1.0)
                if label_pos is not None and len(event_strata_levels) > 0:
                    strata_err = abs((fold_event_strata_counts[f, label_pos] + 1) - target_event_strata[label_pos]) / max(target_event_strata[label_pos], 1.0)
                else:
                    strata_err = 0.0
                scores.append(strata_err + 0.8 * event_err + 0.1 * size_err)
            scores = np.array(scores)
            min_score = scores.min()
            local_best = np.where(np.isclose(scores, min_score))[0]
            tied_folds = candidate_folds[local_best]
            best_fold = int(tied_folds[np.argmin(fold_sizes[tied_folds])])

            fold_valid_lists[best_fold].append(int(idx))
            fold_sizes[best_fold] += 1
            fold_events[best_fold] += 1
            if label_pos is not None and len(event_strata_levels) > 0:
                fold_event_strata_counts[best_fold, label_pos] += 1

        # Stage 2B: assign only non-event rows by group.
        non_event_temp = temp[temp['event'] == 0]
        # Use original row indices (not groupby positional indices) to avoid
        # assigning wrong/duplicated rows across folds.
        non_event_group_to_indices = {
            grp: idx_series.to_numpy(dtype=int)
            for grp, idx_series in non_event_temp.groupby('group').groups.items()
        }
        non_event_groups = list(non_event_group_to_indices.keys())
        rng.shuffle(non_event_groups)
        non_event_groups = sorted(non_event_groups, key=lambda g: len(non_event_group_to_indices[g]), reverse=True)
        for grp in non_event_groups:
            g_idx = np.asarray(non_event_group_to_indices[grp], dtype=int)
            best_fold = int(np.argmin(fold_sizes))
            fold_valid_lists[best_fold].extend(g_idx.tolist())
            fold_sizes[best_fold] += len(g_idx)

    # Build index splits
    splits = []
    for f in range(n_splits):
        valid_idx = np.array(sorted(set(fold_valid_lists[f])), dtype=int)
        train_mask = np.ones(len(temp), dtype=bool)
        train_mask[valid_idx] = False
        train_idx = all_indices[train_mask]
        splits.append((train_idx, valid_idx))

    base_n, rem_n = divmod(total_samples, n_splits)
    target_sizes = np.array([base_n + (1 if i < rem_n else 0) for i in range(n_splits)], dtype=int)

    summary = pd.DataFrame({
        'fold': np.arange(1, n_splits + 1),
        'target_events': target_events_int,
        'assigned_events': fold_events,
        'target_samples': target_sizes,
        'assigned_samples': fold_sizes,
        'valid_event_rate': [event_series.iloc[val].mean() if len(val) > 0 else np.nan for _, val in splits]
    })

    # Add compact event-strata diagnostics to inspect stage-1 balancing.
    for idx, level in enumerate(event_strata_levels):
        safe_level = str(level).replace(' ', '_')
        base_s, rem_s = divmod(int(total_event_strata_counts[idx]), n_splits)
        target_s = np.array([base_s + (1 if i < rem_s else 0) for i in range(n_splits)], dtype=int)
        summary[f'target_event_strata_{safe_level}'] = target_s
        summary[f'assigned_event_strata_{safe_level}'] = fold_event_strata_counts[:, idx]

    return splits, summary

# Multiple imputation by chained equations (MICE), for the diagnostic block.
#
# Why not just fill_with_mode: the missingness analysis in new_data.ipynb showed the
# pathology fields go missing together in a perfectly nested structure -- ER missing
# (137) subset of PR missing (447) subset of HER2 missing (477), zero violations. Mode
# imputation is demonstrably wrong on data like this. Among patients with both ER and
# PR recorded, ER predicts PR at 96.2% concordance; 50 patients have PR missing but ER
# positive, and blind mode imputation would give all 50 the modal PR, contradicting
# their own ER status and manufacturing ER+/PR- tumours that mostly don't exist.
# Chained equations use ER to impute PR and avoid that. (The same nesting also makes
# the missingness indicators near-collinear, which is why encoding them as separate
# dummies gave an ill-conditioned Cox design earlier.)
#
# A plain sklearn IterativeImputer call would get two things wrong here. First,
# categorical fields need CLASSIFIERS not regressors -- IterativeImputer treats every
# column as continuous, so a 3-level field comes back with values like 1.73 that then
# have to be rounded, inventing an ordering that doesn't exist. Second, MAR only holds
# conditional on the variables actually in the imputation model, and missingness here
# depends strongly on diagnosis year, so year needs to go in via aux_variables or the
# assumption is silently violated. The outcome (event indicator + Nelson-Aalen
# cumulative hazard) should go in aux_variables too -- feels backwards but is standard
# for survival data (White & Royston 2009): imputing covariates without the outcome
# biases the covariate-outcome association toward the null.
def mice_impute(dataset, categorical_variables, continuous_variables,
                aux_variables=None, n_iter=5, random_state=0, m=1, return_all=False):
    """Impute missing values by chained equations, on a copy of dataset.

    categorical_variables get a RandomForestClassifier (draws from the predicted
    class distribution, not just argmax); continuous_variables get a
    RandomForestRegressor. aux_variables are extra predictors that are used but
    never themselves imputed or returned -- year of diagnosis, outcome, that kind
    of thing. n_iter is chained-equation sweeps per imputation, m is how many
    imputations to run.

    Returns a single completed DataFrame, or a list of m of them if return_all=True.
    Columns not being imputed are passed through untouched.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    aux_variables = list(aux_variables or [])
    targets = [c for c in list(categorical_variables) + list(continuous_variables)
               if dataset[c].isna().any()]
    if not targets:
        return [dataset.copy() for _ in range(m)] if return_all else dataset.copy()

    work_cols = list(categorical_variables) + list(continuous_variables) + aux_variables
    rng = np.random.RandomState(random_state)

    # Factorise categoricals to integer codes so they can predict one another;
    # `levels` maps back afterwards so the original labels/dtypes are restored.
    base = dataset[work_cols].copy()
    levels = {}
    for c in categorical_variables:
        cat = pd.Categorical(base[c])
        levels[c] = dict(enumerate(cat.categories))
        base[c] = pd.Series(cat.codes, index=base.index).astype(float)
        base.loc[base[c] == -1, c] = np.nan          # -1 is pandas' NaN sentinel

    completed = []
    for _ in range(m):
        work = base.copy()
        # Seed each gap with a random draw from that column's observed values. Seeding
        # at the mode instead is what collapses the variance of a single imputation.
        for c in targets:
            gap = work[c].isna()
            work.loc[gap, c] = rng.choice(base[c].dropna().values, size=int(gap.sum()))

        for _ in range(n_iter):
            for c in targets:
                gap = base[c].isna()
                predictors = [p for p in work.columns if p != c]
                if c in categorical_variables:
                    model = RandomForestClassifier(n_estimators=60, min_samples_leaf=5,
                                                   random_state=rng.randint(1 << 30))
                    model.fit(work.loc[~gap, predictors], base.loc[~gap, c])
                    proba = model.predict_proba(work.loc[gap, predictors])
                    work.loc[gap, c] = [rng.choice(model.classes_, p=row) for row in proba]
                else:
                    model = RandomForestRegressor(n_estimators=60, min_samples_leaf=5,
                                                  random_state=rng.randint(1 << 30))
                    model.fit(work.loc[~gap, predictors], base.loc[~gap, c])
                    work.loc[gap, c] = model.predict(work.loc[gap, predictors])

        out = dataset.copy()
        for c in categorical_variables:
            out[c] = work[c].map(levels[c]).astype(dataset[c].dtype)
        for c in continuous_variables:
            out[c] = work[c].astype(dataset[c].dtype)
        completed.append(out)

    return completed if return_all else completed[0]


def imputation_fmi(completed_datasets, variables):
    """Fraction of missing information per variable across imputations.

    A rough stability diagnostic: how much do the imputed values disagree between
    runs, relative to their spread? Values near 0 mean the chained equations are
    pinning the missing entries down consistently, so a single imputation loses
    little relative to pooling several. Report it rather than assuming it.
    """
    rows = []
    for c in variables:
        stacked = pd.concat([d[c] for d in completed_datasets], axis=1)
        codes = stacked.apply(lambda s: pd.Categorical(
            s, categories=sorted(pd.unique(stacked.values.ravel()))).codes)
        between = codes.var(axis=1, ddof=1).mean()
        within = codes.stack().var(ddof=1)
        rows.append({'variable': c,
                     'between_imputation_var': round(float(between), 5),
                     'total_var': round(float(within), 5),
                     'FMI_approx': round(float(between / within), 4) if within else 0.0})
    return pd.DataFrame(rows)
