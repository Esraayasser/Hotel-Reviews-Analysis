import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from EditingLabels import *

#Load Hotels data
data = pd.read_csv('Hotel_Reviews.csv')
#Drop the rows that contain missing values
data.dropna(how='any', inplace=True)
Hotel_data = data.iloc[:, :]

X = data.iloc[:, 1:11] #Features
Y = data['Reviewer_Score'] #Label
cols = ('Hotel_Name', 'Reviewer_Nationality')
X = Feature_Encoder(X, cols)

"""
#Replaces the values in a column in the dataframe with a specific value by the value that currently exists in it.
X.loc[X['Negative_Review'] == 'No Negative', 'Negative_Review'] = 0
X.loc[X['Negative_Review'] != 0, 'Negative_Review'] = 1

X.loc[X['Positive_Review'] == 'No Positive', "Positive_Review"] = 0
X.loc[X['Positive_Review'] != 0, "Positive_Review"] = 1
"""
X = X.drop(['Review_Date', 'Negative_Review', 'Positive_Review'], axis=1)

for c in X:
    print(c, X[c].values)

Y = np.expand_dims(Y, axis=1)
X = featureScaling(np.array(X), 0, 10)

#Get the correlation between the features
corr = Hotel_data.corr()
#Top 50% Correlation training features with the Value
top_feature = corr.index[corr['Reviewer_Score'] >= -1]
#Correlation plot
plt.subplots(figsize=(12, 8))
top_corr = Hotel_data[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()


#Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)

