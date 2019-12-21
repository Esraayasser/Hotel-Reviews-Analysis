import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from EditingLabels import *
from FTIDF_StringsProcessing import *
from Process_Missing_Values import *
from HandlingCSVs import *

def calculate_correlation(data):
    # Get the correlation between the features
    corr = data.corr()
    # Top 50% Correlation training features with the Value
    top_features = corr.index[corr['Reviewer_Score'] >= -1]
    # Correlation plot
    plt.subplots(figsize=(12, 8))
    top_corr = data[top_features].corr()
    sns.heatmap(top_corr, annot=True)
    plt.show()


def preprocessing(file_name, testing, testing_features_to_drop, tags_weights, model):
    # Load Hotels data
    # Drop Empty rows in the end of the file
    data = read_csv(file_name)
    data = data.iloc[:414738, :]

    # list of feature names that has null values of percentage > 0.25 to drop them
    # and the features with null values of percentage <= 0.25
    features_with_NA, features_to_fill, features_to_drop, missing_val_percent = NA_feature_table(data, 0.25)

    if testing == True:
        # drop the features with a high percentage of missing values.
        for c in features_to_drop:
            print("Dropping", c, "With a null percentage of: ", missing_val_percent[c])
        print(data)
        data = data.drop(testing_features_to_drop, axis=1)

        # fill the features with a low percentage missing of missing values.
        cate_imputer = Categorical_Imputer('most_frequent')
        data[features_with_NA] = cate_imputer.fit_transform(data[features_with_NA])
    else:
        # drop the features with a high percentage of missing values.
        for c in features_to_drop:
            print("Dropping", c, "With a null percentage of: ", missing_val_percent[c])
        print(data)
        data = data.drop(features_to_drop, axis=1)

        # fill the features with a low percentage missing of missing values.
        cate_imputer = Categorical_Imputer('most_frequent')
        data[features_to_fill] = cate_imputer.fit_transform(data[features_to_fill])

    # Pre-processing the features

    # Extracting the hotel country
    if 'Hotel_Address' not in features_to_drop:
        countries = ['Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola', 'Antigua', 'and', 'Barbuda', 'Argentina',
                     'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'The', 'Bahamas', 'Bahrain', 'Bangladesh',
                     'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bhutan', 'Bolivia', 'Bosnia', 'and',
                     'Herzegovina', 'Botswana', 'Brazil', 'Brunei', 'Bulgaria', 'Burkina', 'Faso', 'Burundi', 'Cabo',
                     'Verde', 'Cambodia', 'Cameroon', 'Canada', 'Central', 'African', 'Republic', 'Chad', 'Chile',
                     'China', 'Colombia', 'Comoros', 'Congo,', 'Democratic', 'Republic', 'of', 'the', 'Congo,',
                     'Republic', 'of', 'the', 'Costa', 'Rica', 'Côte', 'dIvoire', 'Croatia', 'Cuba', 'Cyprus', 'Czech',
                     'Republic', 'Denmark', 'Djibouti', 'Dominica', 'Dominican', 'Republic', 'Advertisement', 'East',
                     'Timor', '(Timor-Leste)', 'Ecuador', 'Egypt', 'El', 'Salvador', 'Equatorial', 'Guinea', 'Eritrea',
                     'Estonia', 'Eswatini', 'Ethiopia', 'Fiji', 'Finland', 'France', 'Gabon', 'The', 'Gambia',
                     'Georgia', 'Germany', 'Ghana', 'Greece', 'Grenada', 'Guatemala', 'Guinea', 'Guinea-Bissau',
                     'Guyana', 'Haiti', 'Honduras', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq',
                     'Ireland', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati',
                     'Korea', 'North', 'North', 'Korea', 'South', 'Korea', 'Kosovo', 'Kuwait', 'Kyrgyzstan', 'Laos',
                     'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg',
                     'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Marshall', 'Islands',
                     'Mauritania', 'Mauritius', 'Mexico', 'Micronesia', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro',
                     'Morocco', 'Mozambique', 'Myanmar', 'Burma', 'Myanmar', 'Burma', 'Namibia', 'Nauru', 'Nepal',
                     'Netherlands', 'New', 'Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'North', 'Macedonia', 'Norway',
                     'Oman', 'Pakistan', 'Palau', 'Panama', 'Papua', 'New', 'Guinea', 'Paraguay', 'Peru', 'Philippines',
                     'Poland', 'Portugal', 'Qatar', 'Romania', 'Russia', 'Rwanda', 'Saint', 'Kitts', 'and', 'Nevis',
                     'Saint', 'Lucia', 'Saint', 'Vincent', 'and', 'the', 'Grenadines', 'Samoa', 'San', 'Marino', 'Sao',
                     'Tome', 'and', 'Principe', 'Saudi', 'Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra', 'Leone',
                     'Singapore', 'Slovakia', 'Slovenia', 'Solomon', 'Islands', 'Somalia', 'South', 'Africa', 'Spain',
                     'Sri', 'Lanka', 'Sudan', 'Sudan,', 'South', 'Suriname', 'Sweden', 'Switzerland', 'Syria', 'Taiwan',
                     'Tajikistan', 'Tanzania', 'Thailand', 'Togo', 'Tonga', 'Trinidad', 'and', 'Tobago', 'Tunisia',
                     'Turkey', 'Turkmenistan', 'Tuvalu', 'Uganda', 'Ukraine', 'United', 'Arab', 'Emirates',
                     'United Kingdom', 'United States', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Vatican', 'City',
                     'Venezuela', 'Vietnam', 'Yemen', 'Zambia', 'Zimbabwe']
        tmp_list = []
        for hotel in data['Hotel_Address']:
            split = hotel.split(' ')
            if split[len(split) - 2] + ' ' + split[len(split) - 1] in countries:
                split = split[len(split) - 2] + ' ' + split[len(split) - 1]
            elif split[len(split) - 1] in countries:
                split = split[len(split) - 1]
            tmp_list.append(split)

        data.update(pd.Series(tmp_list, name='Hotel_Address', index=range(len(data['Hotel_Address']))))

    # Extracting the review year from the 'Review_Date' column.
    if 'Review_Date' not in features_to_drop:
        tmp_list = [int(date.split('/')[2]) for date in data['Review_Date']]
        data.update(pd.Series(tmp_list, name='Review_Date', index=range(len(data['Review_Date']))))
        data['Review_Date'] = data['Review_Date'].astype(int)

    # Processing the Review strings using IF-TDF
    if 'Positive_Review' not in features_to_drop and 'Negative_Review' not in features_to_drop:
        tmp = String_Processing(data)
        data = data.drop(['Positive_Review', 'Negative_Review'], axis=1)
        data = data.join(tmp)

    # Extracting the tags in the 'Tags' column.

    tmp_list = [tag.split('[')[1] for tag in data['Tags']]
    tmp_list = [tag.split(']')[0] for tag in tmp_list]
    tmp_list = [tag.split(", ") for tag in tmp_list]
    for i in range(len(tmp_list)):
        tmp_list[i] = [tag.split("' ")[1] for tag in tmp_list[i]]
        tmp_list[i] = [tag.split(" '")[0] for tag in tmp_list[i]]

    data.update(pd.Series(tmp_list, name='Tags', index=range(len(data['Tags']))))
    data = multiVal_feature_encoder(data, ['Tags'])
    data['Tags'] = data['Tags'].astype(float)

    # Delete the word 'days' in the 'days_since_review' column.
    if 'days_since_review' not in features_to_drop:
        tmp_list = [int(days.split(' ')[0]) for days in data['days_since_review']]
        data.update(pd.Series(tmp_list, name='days_since_review', index=range(len(data['days_since_review']))))
        data['days_since_review'] = data['days_since_review'].astype(int)

    if model == 'c':
        # Encoding the y label to 3 classes 0,1,2
        tmp = []
        for val in data['Reviewer_Score']:
            val = val.lower()
            if 'high_reviewer_score' in val:
                tmp.append(0)
            elif 'intermediate_reviewer_score' in val:
                tmp.append(1)
            else:
                tmp.append(2)
        data.update(pd.Series(tmp, name='Reviewer_Score', index=range(len(data['Reviewer_Score']))))

        data['Reviewer_Score'] = data['Reviewer_Score'].astype(int)
        label = data['Reviewer_Score']
        data = data.drop(['Reviewer_Score'], axis=1)
    else:
        data = feature_lbl_encoder(data, ['Reviewer_Score'])

    cols = ['Hotel_Name', 'Reviewer_Nationality', 'Hotel_Address']

    data_hashing_encoder = feature_hashing_encoder(data, cols)
    data_dummies_encoder = feature_dummies_encoder(data, cols)
    data_lbl_encoder = feature_lbl_encoder(data, cols)

    data_hashing_encoder = feature_scaling(data_hashing_encoder, 0, 10)
    data_dummies_encoder = feature_scaling(data_dummies_encoder, 0, 10)
    data_lbl_encoder = feature_scaling(data_lbl_encoder, 0, 10)

    data_hashing_encoder = data_hashing_encoder.join(label)
    data_dummies_encoder = data_dummies_encoder.join(label)
    data_lbl_encoder = data_lbl_encoder.join(label)

    write_in_csv(data_hashing_encoder, 'data_hashing_encoder.csv')
    write_in_csv(data_dummies_encoder, 'data_dummies_encoder.csv')
    write_in_csv(data_lbl_encoder,'data_lbl_encoder.csv')

    return data_hashing_encoder, data_dummies_encoder, data_lbl_encoder

