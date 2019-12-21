from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np


def feature_lbl_encoder(x, cols):
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
        # Concatenating the new hashed features to the original data-frame.
        x = x.join(feature_dummies)
    return x


def feature_hashing_encoder(x, cols):
    for c in cols:
        fh = FeatureHasher(n_features=10, input_type='string')
        sp = fh.fit_transform(x[c])
        df = pd.DataFrame(sp.toarray())
        df.columns = [c + str(i) for i in range(df.shape[1])]
        x = pd.concat([x, df], axis=1)
        x = x.drop(c, axis=1)
    return x


def multiVal_feature_encoder(x, cols):
    for c in cols:
        mlb = MultiLabelBinarizer()
        fitteddata = mlb.fit_transform(x[c].values.tolist())
        tags_weight = [sum(fitteddata[:, i]) for i in range(fitteddata.shape[1])]
        tags_weight = [val / x.shape[0] for val in tags_weight]
        tags_weight = np.reshape(tags_weight, (1, len(tags_weight)))
        training_tags_weights = pd.DataFrame(tags_weight, columns=mlb.classes_)

        # fitteddata = pd.DataFrame(fitteddata, columns=mlb.classes_)
        """
        print(fitteddata.shape)
        pca = PCA(n_components=70)
        principalComponents = pca.fit_transform(fitteddata)
        fitteddata = pd.DataFrame(principalComponents, columns=['tag ' + i for i in range(70)])
        print(fitteddata)
        #x.drop(c, axis=1)
        #x = x.join(pd.DataFrame(fitteddata, columns=mlb.classes_))
        #x = feature_hashing_encoder(x, cols)
        """
        tags_weight = []
        for val in x[c]:
            for tag in val:
                weight = 0
                if tag in training_tags_weights.columns:
                    weight += training_tags_weights[tag]
            tags_weight.append(weight)
        x.update(pd.Series(tags_weight, name='Tags', index=range(len(x['Tags']))))
    return x


def feature_scaling(x, a, b):
    cols = x.columns
    x = np.array(x)
    normalized_x = np.zeros((x.shape[0], x.shape[1]))
    for i in range(x.shape[1]):
        normalized_x[:, i] = ((x[:, i] - min(x[:, i])) /
                              (max(x[:, i]) - min(x[:, i]))) * (b - a) + a
    return pd.DataFrame(normalized_x, columns=cols)
