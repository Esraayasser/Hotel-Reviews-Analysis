import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import heapq
import pandas as pd
import numpy as np
import re

def String_Processing(strings_list):
    stop_words = set(stopwords.words('english'))
    strings = [s for s in strings_list]
    text = []
    for i in range(len(strings)):
        strings[i] = strings[i].lower()
        strings[i] = re.sub(r'\s+', ' ', strings[i])
        strings[i] = re.sub(r'\W', ' ', strings[i])

    for str in strings_list:
        token = nltk.word_tokenize(str)
        token = [w for w in token if not w in stop_words]
        text.append(" ".join(token))

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), use_idf=True)
    fitted_vectorizer = tfidf_vectorizer.fit(text)
    tfidf_vectorizer_vectors = fitted_vectorizer.transform(text)

    final_col = []
    print(tfidf_vectorizer_vectors.shape[0])
    for i in range(tfidf_vectorizer_vectors.shape[0]):
        df = pd.DataFrame(tfidf_vectorizer_vectors[i].T.todense(), index=tfidf_vectorizer.get_feature_names(),
                          columns=["tfidf"])
        ss = df.sum(axis = 0, skipna = True)
        final_col.append(ss['tfidf'])
    return final_col
