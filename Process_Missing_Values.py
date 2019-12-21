from sklearn_pandas import CategoricalImputer

# categorical_imputer
class Categorical_Imputer:
    # Imputing categorical data using the most frequent value

    # instance attribute
    def __init__(self, strategy):
        self.strategy = strategy

    # instance method
    def fit_transform(self, df: 'dataframe') -> 'dataframe':
        # Fill in missing categorical values using most frequent value
        # instantiate CategoricalImputer
        imputer = CategoricalImputer()
        # convert array to dataframe
        df_filled = df.apply(lambda x: imputer.fit_transform(x), axis=0)
        # return filled dataframe
        return df_filled


def NA_feature_table(df, percentage):
    # Percentage of missing values
    missing_val_percent = 100 * (df.isnull().sum() / len(df))
    features_with_NA = []
    features_to_fill = []
    features_to_drop = []
    for c in df.columns:
        if missing_val_percent[c] > 0:
            features_with_NA.append(c)
        if missing_val_percent[c] <= percentage:
            features_to_fill.append(c)
        else:
            features_to_drop.append(c)
    return features_with_NA, features_to_fill, features_to_drop, missing_val_percent
