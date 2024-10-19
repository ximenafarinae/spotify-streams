def convert_to_numeric_types(df, pd):
    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            df[column] = pd.to_numeric(df[column], errors='coerce')