import numpy as np


def convert_columns_values_to_numeric(pd, df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def remove_outliers_percentile(df, columns, lower_percentile=0.01, upper_percentile=0.95):
    for column in columns:
        lower_bound = df[column].quantile(lower_percentile)
        upper_bound = df[column].quantile(upper_percentile)
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

def normalize_input(df):
    df_stats = df.describe().T
    df_norm = (df - df_stats['mean']) / df_stats['std']
    return df_norm

def robust_scaling(data, columns):
    df_scaled = data.copy()
    for column in columns:
        median = data[column].median()
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1
        df_scaled[column] = (data[column] - median) / iqr
    return df_scaled
