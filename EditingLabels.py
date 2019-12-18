from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np



def lbl_feature_encoder(x, cols):
    for c in cols:
        # label encoding for the string values in the feature.
        lbl = LabelEncoder()
        lbl.fit(list(x[c].values))
        x[c] = lbl.transform(list(x[c].values))
    return x


def hashing_onehot_feature_encoder(x, cols):
    for c in cols:
        # label encoding for the string values in the feature.
        lbl = LabelEncoder()
        lbl.fit(list(x[c].values))
        x[c] = lbl.transform(list(x[c].values))
        # Converting the categorical variable into its one-hot encoding representation.
        # by casting our it into the built-in pandas Categorical data type.
        x[c] = pd.Categorical(x[c])
        # Drop the old categorical column
        x.drop(c, axis=1)
        # encoding a categorically feature into a set of features.
        # using the pandas method to convert categorical variables into dummy/indicator variables
        # with the get_dummies function.
        feature_dummies = pd.get_dummies(x[c], prefix=c)
        print(feature_dummies.shape, x.shape)
        # Concatenating the new hashed features to the original data-frame.
        x = x.join(feature_dummies)
    return x


def multiVal_feature_encoder(x, cols):
    for c in cols:
        mlb = MultiLabelBinarizer()
        fitteddata = mlb.fit_transform(x[c])
        x = x.join(pd.DataFrame(fitteddata, columns=mlb.classes_))
    return x


def feature_scaling(x, a, b):
    normalized_x = np.zeros((x.shape[0], x.shape[1]))
    for i in range(x.shape[1]):
        normalized_x[:, i] = ((x[:, i] - min(x[:, i])) /
                              (max(x[:, i]) - min(x[:, i]))) * (b - a) + a
    return normalized_x
