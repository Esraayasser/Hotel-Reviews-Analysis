import pickle


def load_models():
    models = list()
    models.append(pickle.load(open('svm3.sav', 'rb')))
    models.append(pickle.load(open('dt.sav', 'rb')))
    models.append(pickle.load(open('ada.sav', 'rb')))
    return models


def save_models(models, models_names):
    for i in len(models):
        filename = models_names[i] + '.sav'
        pickle.dump(models[i], open(filename, 'wb'))