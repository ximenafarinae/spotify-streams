def convert_columns_values_to_numeric(pd, df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def remove_outliers(df, columns):
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

def normalize_input(df):
    df_stats = df.describe().T
    print(df_stats)
    df_norm = (df - df_stats['mean']) / df_stats['std']
    print(df_norm)
    return df_norm
