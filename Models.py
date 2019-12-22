import pickle


def load_models():
    models = list()
    models.append(pickle.load(open('svm3.sav', 'rb')))
    models.append(pickle.load(open('dt.sav', 'rb')))
    models.append(pickle.load(open('ada.sav', 'rb')))
    return models

