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
    data_set['Adjuvant therapy with bio drugs'] = data_set[variable].apply(lambda x: 1 if x in [1, 2] else 0)

    encoder = OneHotEncoder(sparse_output=False, drop='first')
    performed_data = data_set[data_set['Adjuvant therapy with bio drugs'] == 1]
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
    """Missing-value counts for the given columns, only the ones that actually have any."""
    col_list = list(columns)

    missing_counts = df[col_list].isnull().sum()

    return missing_counts[missing_counts > 0]

def one_hot_encode_with_missing_indicator(dataset, variables):
    # one-hot encode variables that have missing values; returns (dataset_copy, new_column_names)
    dataset_copy = dataset.copy()
    all_new_columns = []
    
    for variable in variables:
        missing_mask = dataset_copy[variable].isnull()
        dataset_copy[variable] = dataset_copy[variable].fillna(404)  # placeholder value, dropped below

        encoder = OneHotEncoder(sparse_output=False, drop=None, handle_unknown='ignore')
        encoded_array = encoder.fit_transform(dataset_copy[[variable]])
        feature_names = encoder.get_feature_names_out([variable])
        encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=dataset_copy.index)

        # the 404 dummy itself shouldn't end up as a feature
        col_404 = f"{variable}_404"
        col_404_float = f"{variable}_404.0"
        if col_404 in encoded_df.columns:
            encoded_df = encoded_df.drop(columns=[col_404])
        elif col_404_float in encoded_df.columns:
            encoded_df = encoded_df.drop(columns=[col_404_float])

        # missing rows go all-zero instead of getting the 404 dummy
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