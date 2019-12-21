from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from HandlingCSVs import *
from Preprocessing import *
from PolynomialRegression import *
from LinearRegression import *
from sklearn.model_selection import train_test_split
from MultipleLinearRegression import *
import SVM
import Testing
import KNeighborsClassifier

model = input("Would you like to run the classification or the regression model? c/r : ")
if model == 'r':

    # Classification milestone2, original file : 'Hotel_Reviews_Milestone_2.csv'

    # Initialization
    models = []  # list of models
    models_names = []  # list of models names
    models_training_time = []  # list of models training time
    # A list of all the data's dataframe to use the models on
    data_list = []
    data_list_names = []

    X_train = []
    Y_train = []  # list of training samples
    X_test = []
    Y_test = []  # list of testing samples

    # Pre processing
    test = input("Would you like to preprocess the test data? y/n : ")
    if test == 'y':
        file_name = input("Please enter the file name : ")
        data_list = preprocessing(file_name, True, testing_features_to_drop=['lat', 'lng'], tags_weights=read_csv('review_tags'), model=model)
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
    if use_pca is True:
        list_id = -1
        for data in data_list:
            list_id += 1
            data = data.drop(['Reviewer_Score'], axis=1)
            pca = PCA(n_components=data.shape[1])
            principalComponents = pca.fit_transform(data.iloc[:, :])
            print(pca.explained_variance_ratio_)
            # region PCA Cumulative Plotting
            plt.plot(np.cumsum(pca.explained_variance_ratio_))
            plt.title('PCA of ' + data_list_names[list_id])
            plt.xlabel('number of components')
            plt.ylabel('cumulative explained variance')
            plt.show()
            # endregion

    #training
    # Splitting each data file into train and test samples.
    # The index of the data file name that we're training on now.
    if test == 'n':
        list_id = -1
        for data in data_list:
            list_id += 1
            # Dividing the data to x and y
            x = data.iloc[:, :]  # Features
            x = x.drop(['Reviewer_Score'], axis=1)
            y = data['Reviewer_Score']  # Label
            y = np.expand_dims(y, axis=1)
            # Split the data to training and testing sets
            X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.20, shuffle=True)
            print(data_list_names[list_id])
            print(X_train.head(3))
            print(Y_train)
            # calling of the models
            # Training models

            svm_models, svm_times = SVM.svm_models(x_train, y_train)
            for i in range(0, len(svm_models)):
                models.append(svm_models[i])
                models_names.append('svm ' + str(i) + ' ' + data_list_names[list_id])
                models_training_time.append(svm_times[i])

            model, time = DecisionTree.decision_tree_model(x_train, y_train)
            models.append(model)
            models_names.append('decision-tree' + ' ' + data_list_names[list_id])
            models_training_time.append(time)

            model, time = Adaboost.adaboost_model(x_train, y_train)
            models.append(model)
            models_names.append('adaboost-decision-tree' + ' ' + data_list_names[list_id])
            models_training_time.append(time)
			
			# KNeighborsClassifier.plot_different_k_values(x_train, y_train, x_test, y_test)

			model, time = KNeighborsClassifier.knn_train(x_train, y_train, k=3)
			models.append(model)
			models_names.append('KNN')
			models_training_time.append(time)
            
    if test == 'y':
        # Testing Models
        models_accuracies, models_testing_time = Testing.model_testing(models, X_test, Y_test)

    # Printing Models outputs
    for i in range(len(models)):
        print("Model name is {} - accuracy = {} and model time = {}".
              format(models_names[i], models_accuracies*[i], models_testing_time[i]))
"""
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
