import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from EditingLabels import *
from IFTDF_StringsProcessing import *


#def preprocessing():
# Load Hotels data
data = pd.read_csv('Hotel_Reviews.csv')
# Drop the rows that contain missing values
#data.dropna(how='any', inplace=True)

x = data.iloc[:, :16]  # Features
y = data['Reviewer_Score']  # Label

# Extracting the hotel country
countries = ['Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola', 'Antigua', 'and', 'Barbuda', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'The', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bhutan', 'Bolivia', 'Bosnia', 'and', 'Herzegovina', 'Botswana', 'Brazil', 'Brunei', 'Bulgaria', 'Burkina', 'Faso', 'Burundi', 'Cabo', 'Verde', 'Cambodia', 'Cameroon', 'Canada', 'Central', 'African', 'Republic', 'Chad', 'Chile', 'China', 'Colombia', 'Comoros', 'Congo,', 'Democratic', 'Republic', 'of', 'the', 'Congo,', 'Republic', 'of', 'the', 'Costa', 'Rica', 'Côte', 'dIvoire', 'Croatia', 'Cuba', 'Cyprus', 'Czech', 'Republic', 'Denmark', 'Djibouti', 'Dominica', 'Dominican', 'Republic', 'Advertisement', 'East', 'Timor', '(Timor-Leste)', 'Ecuador', 'Egypt', 'El', 'Salvador', 'Equatorial', 'Guinea', 'Eritrea', 'Estonia', 'Eswatini', 'Ethiopia', 'Fiji', 'Finland', 'France', 'Gabon', 'The', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Grenada', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati', 'Korea', 'North', 'North', 'Korea', 'South', 'Korea', 'Kosovo', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Marshall', 'Islands', 'Mauritania', 'Mauritius', 'Mexico', 'Micronesia', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Burma', 'Myanmar', 'Burma', 'Namibia', 'Nauru', 'Nepal', 'Netherlands', 'New', 'Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'North', 'Macedonia', 'Norway', 'Oman', 'Pakistan', 'Palau', 'Panama', 'Papua', 'New', 'Guinea', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar', 'Romania', 'Russia', 'Rwanda', 'Saint', 'Kitts', 'and', 'Nevis', 'Saint', 'Lucia', 'Saint', 'Vincent', 'and', 'the', 'Grenadines', 'Samoa', 'San', 'Marino', 'Sao', 'Tome', 'and', 'Principe', 'Saudi', 'Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra', 'Leone', 'Singapore', 'Slovakia', 'Slovenia', 'Solomon', 'Islands', 'Somalia', 'South', 'Africa', 'Spain', 'Sri', 'Lanka', 'Sudan', 'Sudan,', 'South', 'Suriname', 'Sweden', 'Switzerland', 'Syria', 'Taiwan', 'Tajikistan', 'Tanzania', 'Thailand', 'Togo', 'Tonga', 'Trinidad', 'and', 'Tobago', 'Tunisia', 'Turkey', 'Turkmenistan', 'Tuvalu', 'Uganda', 'Ukraine', 'United', 'Arab', 'Emirates', 'United Kingdom', 'United States', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Vatican', 'City', 'Venezuela', 'Vietnam', 'Yemen', 'Zambia', 'Zimbabwe']
tmp_list = []
for hotel in x['Hotel_Address']:
    split = hotel.split(' ')
    if split[len(split) - 2] + ' ' + split[len(split) - 1] in countries:
        split = split[len(split) - 2] + ' ' + split[len(split) - 1]
    elif split[len(split) - 1] in countries:
        split = split[len(split) - 1]
    tmp_list.append(split)

x.update(pd.Series(tmp_list, name='Hotel_Address', index=range(len(x['Hotel_Address']))))

# Extracting the review year from the 'Review_Date' column.
tmp_list = [date.split('/')[2] for date in x['Review_Date']]
x.update(pd.Series(tmp_list, name='Review_Date', index=range(len(x['Review_Date']))))
"""
# Processing the Review strings using IF-TDF
with_null_values = sum([1 for s in x['Positive_Review'] if len(s) == 0])
if with_null_values == 0:
    String_Processing(x['Positive_Review'])
with_null_values = sum([1 for s in x['Negative_Review'] if len(s) == 0])
if with_null_values == 0:
    String_Processing(x['Negative_Review'])
"""
# Extracting the tags in the 'Tags' column.
tmp_list = [tag.split('[')[1] for tag in x['Tags']]
tmp_list = [tag.split(']')[0] for tag in tmp_list]
tmp_list = [tag.split(", ") for tag in tmp_list]
for i in range(len(tmp_list)):
    tmp_list[i] = [tag.split("' ")[1] for tag in tmp_list[i]]
    tmp_list[i] = [tag.split(" '")[0] for tag in tmp_list[i]]

x.update(pd.Series(tmp_list, name='Tags', index=range(len(x['Tags']))))

# Delete the word 'days' in the 'days_since_review' column.
tmp_list = [days.split(' ')[0] for days in x['days_since_review']]
x.update(pd.Series(tmp_list, name='days_since_review', index=range(len(x['days_since_review']))))
xx = x.iloc[:20, :]
print(xx)
cols = ('Hotel_Name', 'Reviewer_Nationality', 'Hotel_Address')
x = hashing_onehot_feature_encoder(x, cols)
xx = x.iloc[:20, :]
print(xx)
x = feature_scaling(np.array(x), 0, 10)
"""
    # Get the correlation between the features
    corr = hotel_data.corr()
    # Top 50% Correlation training features with the Value
    top_feature = corr.index[corr['Reviewer_Score'] >= -1]
    # Correlation plot
    plt.subplots(figsize=(12, 8))
    top_corr = hotel_data[top_feature].corr()
    sns.heatmap(top_corr, annot=True)
    plt.show()

    # Split the data to training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, shuffle=True)
    y = np.expand_dims(y, axis=1)
    
    return x_train, x_test, y_train, y_test, x, y
"""
