import SVM
import Testing

# Initialization
models = []  # list of models
models_names = []  # list of models names
models_training_time = []  # list of models training time

# Pre processing
X_train = [], y_train = []  # list of training samples
X_test = [], y_test = []  # list of testing samples

# Training models
model, time = SVM.svm_one_versus_one_training(X_train, y_train)
models.append(model)
models_names.append('svm-one-versus-one')
models_training_time.append(time)

model, time = SVM.svm_one_versus_rest_training(X_train, y_train)
models.append(model)
models_names.append('svm-one-versus-all')
models_training_time.append(time)

# Testing Models
models_accuracies, models_testing_time = Testing.model_testing(models, X_test, y_test)

# Printing Models outputs
for i in range(len(models)):
    print("Model name is {} - accuracy = {} and model time = {}".
          format(models_names[i], models_accuracies*[i], models_testing_time[i]))
