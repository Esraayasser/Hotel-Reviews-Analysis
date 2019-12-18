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


def feature_dummies_encoder(x, cols):
    for c in cols:
        # label encoding for the string values in the feature.
        lbl = LabelEncoder()
        lbl.fit(list(x[c].values))
        x[c] = lbl.transform(list(x[c].values))
        # Converting the categorical variable into its one-hot encoding representation.
        # by casting our it into the built-in pandas Categorical data type.
        x[c] = pd.Categorical(x[c])
        # encoding a categorically feature into a set of features.
        # using the pandas method to convert categorical variables into dummy/indicator variables
        # with the get_dummies function.
        feature_dummies = pd.get_dummies(x[c], prefix=c, drop_first=True)
        print(feature_dummies.shape, x.shape)
        # Concatenating the new hashed features to the original data-frame.
        x = x.join(feature_dummies)
    return x

def feature_hashing_encoder(x, cols):
    for c in cols:
        fh = FeatureHasher(n_features=10, input_type='string')
        sp = fh.fit_transform(x[c])
        df = pd.DataFrame(sp.toarray())
        df.columns = [c+str(i) for i in range(df.shape[1])]
        print(df.shape, x.shape)
        print(df)
        x = pd.concat([x, df], axis=1)
        x = x.drop(c, axis=1)
    return x


def multiVal_feature_encoder(x, cols):
    for c in cols:
        mlb = MultiLabelBinarizer()
        fitteddata = mlb.fit_transform(x[c].values.tolist())
        df =  pd.DataFrame(fitteddata, columns=mlb.classes_)
        print("yup,Here!")
        x = x.join(pd.DataFrame(fitteddata, columns=mlb.classes_))
        print(df)
    return x


def feature_scaling(x, a, b):
    normalized_x = np.zeros((x.shape[0], x.shape[1]))
    for i in range(x.shape[1]):
        normalized_x[:, i] = ((x[:, i] - min(x[:, i])) /
                              (max(x[:, i]) - min(x[:, i]))) * (b - a) + a
    return normalized_x
