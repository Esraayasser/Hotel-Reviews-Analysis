from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from HandlingCSVs import *
from Preprocessing import *
from PolynomialRegression import *
from LinearRegression import *
from sklearn import model_selection
from MultipleLinearRegression import *
import Testing
import KNeighborsClassifier
import SVM
import Adaboost
import DecisionTree
import pickle
import Models
import HandlingCSVs

model = input("Would you like to run the classification or the regression model? c/r : ")
if model == 'c':

    # Classification milestone2, original file : 'Hotel_Reviews_Milestone_2.csv'

    # Initialization
    models = []  # list of models
    models_names = []  # list of models names
    models_training_time = []  # list of models training time
    # A list of all the data's dataframe to use the models on
    data_list = []
    data_list_names = []

    x_train = []
    y_train = []  # list of training samples
    x_test = []
    y_test = []  # list of testing samples

    # Pre processing
    preprocess_test = input("Would you like to preprocess the test data? y/n : ")
    if preprocess_test == 'y':
        file_name = input("Please enter the file name : ")
        data_list = preprocessing(file_name, True, testing_features_to_drop=['lat', 'lng'], tags_weights=read_csv('tags_weights.csv'), model=model)
    else:
        # data_hashing_encoder.csv and data_lbl_encoder.csv
        while True:
            file_name = input("Please enter the file name : ")
            data_list_names.append(file_name.split('.')[0])
            data = read_csv(file_name).round(3)
            data_list.append(data)
            ok = input("Would you like to read another file? y/n : ")
            if ok == 'n':
                break

    use_pca = input("Would you like to use PCA on the data? y/n : ")
    if use_pca == 'y':
        n_comps = input("Please enter PCA's n_components: ")
        list_id = -1
        for data in data_list:
            list_id += 1
            data = data.drop(['Reviewer_Score'], axis=1)
            pca = PCA(n_components=int(n_comps))
            principalComponents = pca.fit_transform(data.iloc[:, :])
            print(pca.explained_variance_ratio_)
            # region PCA Cumulative Plotting
            plt.plot(np.cumsum(pca.explained_variance_ratio_))
            plt.title('PCA of ' + data_list_names[list_id])
            plt.xlabel('number of components')
            plt.ylabel('cumulative explained variance')
            plt.show()
            # endregion

    # Splitting each data file into train and test samples.
    # The index of the data file name that we're training on now.
    list_id = -1
    for data in data_list:
        list_id += 1
        # Dividing the data to x and y
        x = data.iloc[:, :]  # Features
        x.drop(['Additional_Number_of_Scoring', 'Review_Date', 'Average_Score', 'Total_Number_of_Reviews',
                'Total_Number_of_Reviews_Reviewer_Has_Given', 'Tags', 'days_since_review',
                'Hotel_Name0', 'Hotel_Name1', 'Hotel_Name2', 'Hotel_Name3', 'Hotel_Name4',
                'Hotel_Name5', 'Hotel_Name6', 'Hotel_Name7', 'Hotel_Name8', 'Hotel_Name9',
                'Reviewer_Nationality0', 'Reviewer_Nationality1', 'Reviewer_Nationality2',
                'Reviewer_Nationality3', 'Reviewer_Nationality4', 'Reviewer_Nationality5',
                'Reviewer_Nationality6', 'Reviewer_Nationality7', 'Reviewer_Nationality8',
                'Reviewer_Nationality9',
                'Hotel_Address0', 'Hotel_Address1', 'Hotel_Address2', 'Hotel_Address3',
                'Hotel_Address4', 'Hotel_Address5', 'Hotel_Address6',
                'Hotel_Address7', 'Hotel_Address8', 'Hotel_Address9',
                'Reviewer_Score'], axis=1)
        y = data['Reviewer_Score']
        # y = data.iloc[:100000, 40] # Label
        y = y.values.ravel()  # convert column vector to 1d array
        # y = np.expand_dims(y, axis=1)

        do = input("Would you like to train or test? train/test: ")
        if do == "test":
            models = Models.load_models()
            x_test = x
            y_test = y

        if do == "train":
            """
            # Split the data to training and testing sets
            x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.20, shuffle=False)
            print(data_list_names[list_id])
            print(x_train.head(3))
            print(y_train)
            """

            """
            for c in [1000, 100000, 1000000]:
                model, time = SVM.svm_models(x_train, y_train, c)
                models.append(model)
                models_names.append('SVM ' + str(c))
                models_training_time.append(time)
            """

            x_train = x
            y_train = y

            model, time = SVM.svm_models(x_train, y_train, 1000000)
            models.append(model)
            models_names.append('SVM 3')
            models_training_time.append(time)
            filename = 'svm3.sav'
            pickle.dump(model, open(filename, 'wb'))


            """
            KNeighborsClassifier.plot_different_k_values(x_train, y_train, x_test, y_test)
            model, time = KNeighborsClassifier.knn_train(x_train, y_train, k=20)
            models.append(model)
            models_names.append('KNN')
            models_training_time.append(time)
            """

            model, time = DecisionTree.decision_tree_model(x_train, y_train)
            models.append(model)
            models_names.append('Decision Tree')
            models_training_time.append(time)
            filename = 'dt.sav'
            pickle.dump(model, open(filename, 'wb'))

            model, time = Adaboost.adaboost_model(x_train, y_train)
            models.append(model)
            models_names.append('Adaboost Decision Tree')
            models_training_time.append(time)
            filename = 'ada.sav'
            pickle.dump(model, open(filename, 'wb'))

        # Testing Models
        models_accuracies, models_testing_time = Testing.model_testing(models, x_test, y_test)

        # Printing Models outputs
        for i in range(len(models)):
            print("Model name is {} - accuracy = {} and model time = {}".
                  format(models_names[i], models_accuracies[i], models_testing_time[i]))
        # Bar Plotting
        # Accuracies
        y_pos = np.arange(len(models_names))
        plt.bar(y_pos, models_accuracies, align='center', alpha=0.5)
        plt.xticks(y_pos, models_names)
        plt.ylabel('Accuracy')
        plt.title('Classification Accuracy')
        plt.show()

        y_pos = np.arange(len(models_training_time))
        plt.bar(y_pos, models_training_time, align='center', alpha=0.5)
        plt.xticks(y_pos, models_names)
        plt.ylabel('Time')
        plt.title('Total Training Time')
        plt.show()

        y_pos = np.arange(len(models_testing_time))
        plt.bar(y_pos, models_testing_time, align='center', alpha=0.5)
        plt.xticks(y_pos, models_names)
        plt.ylabel('Time')
        plt.title('Total Testing Time')
        plt.show()

"""
# MS1
else:
    linearAS_train_error, linearAS_test_error, modelAS = Linear_Regression_averageScore(x_train, x_test, y_train, y_test)
    PlotLinearModel(X[:, 1], Y, modelAS, 'Average Score', 'Reviewer Score')
    
    linearPRC_train_Error, LinearPRC_test_Error, modelPRC = Linear_Regression_PositiveReviewCount(x_train, x_test, y_train, y_test)
    PlotLinearModel(X[:, 6], Y, modelPRC, 'Positive Review Word Count', 'Reviewer Score')
    
    linearNRC_train_error, linearNRC_test_error, modelNRC = LinearRegressionNegativeReviewCount(x_train, x_test, y_train, y_test)
    PlotLinearModel(X[:, 4], Y, modelNRC, 'Negative Review Word Count', 'Reviewer Score')
    
    #multipleLR1_train_Error, multipleLR1_test_Error = Multiple_Linear_Regression1(x_train, x_test, y_train, y_test)
    multipleLR1_test_Error = Multiple_Linear_Regression1(x_train, x_test, y_train, y_test)
    multipleLR_PN_error = MultipleLinearRegressionPositiveNegative(x_train, x_test, y_train, y_test)
    
    poly1_error_train, poly1_error_test = polynomial_regression_all(x_train, x_test, y_train, y_test)
    poly2_error_train, poly2_error_test = polynomial_regression_top_three(x_train, x_test, y_train, y_test) """
