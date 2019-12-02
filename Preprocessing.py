import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from EditingLabels import *


def preprocessing():
    # Load Hotels data
    data = pd.read_csv('Hotel_Reviews.csv')
    # Drop the rows that contain missing values
    data.dropna(how='any', inplace=True)
    hotel_data = data.iloc[:, :]
    x = data.iloc[:, 0:12]  # Features
    y = data['Reviewer_Score']  # Label
    # Encode string data
    cols = ('Hotel_Address', 'Hotel_Name', 'Reviewer_Nationality')
    x = feature_encoder(x, cols)
    """
    #Replaces the values in a column in the dataframe with a specific value by the value that currently exists in it.
    x.loc[x['Negative_Review'] == 'No Negative', 'Negative_Review'] = 0
    x.loc[x['Negative_Review'] != 0, 'Negative_Review'] = 1
    
    x.loc[x['Positive_Review'] == 'No Positive', "Positive_Review"] = 0
    x.loc[x['Positive_Review'] != 0, "Positive_Review"] = 1
    """
    x = x.drop(['Review_Date', 'Negative_Review', 'Positive_Review'], axis=1)
    
    '''for c in x:
        print(c, x[c].values)'''

    # Normalize data
    x = feature_scaling(np.array(x), 0, 10)
    
    # Get the correlation between all features
    corr = hotel_data.corr()

    # All Correlation training features with the Value
    correlated_features = corr.index[corr['Reviewer_Score'] >= -1]

    # Plot correlation
    plt.subplots(figsize=(12, 8))
    correlations = hotel_data[correlated_features].corr()
    sns.heatmap(correlations, annot=True)
    plt.show()

    # Split the data to training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, shuffle=True)
    y = np.expand_dims(y, axis=1)
    return x_train, x_test, y_train, y_test, x, y
