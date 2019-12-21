import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import re


def String_Processing(x):
    # A list of the full review of each sample in the dataset.
    strings_list = []
    text = []

    # NLP on the reviews.
    # Getting a list of the english stopwords
    stop_words = set(stopwords.words('english'))
    for i in range(x.shape[0]):
        # Concatenating the positive and the negative reviews to get the full review fo each sample.
        string = x['Positive_Review'][i] + x['Negative_Review'][i]
        # Converting all the review's words to lower case words to facilitate it's processing.
        string = string.lower()
        # Remove any special characters.
        string = re.sub(r'\s+', ' ', string)
        string = re.sub(r'\W', ' ', string)
        strings_list.append(string)

    for str in strings_list:
        # Using the NLP NLTK libirary to get the tokens of each string.
        token = nltk.word_tokenize(str)
        # Removing the stop words from the string tokens.
        token = [w for w in token if not w in stop_words]
        # Revert the tokens to strings.
        text.append(" ".join(token))

    # Intializing the TfidfVectorizer method by stating that the words combination will be each in rang of [1 3] words per process by using IDF.
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), use_idf=True)
    # Applying the TF-IDF on the reviews string to get the TF-wieghts matrix.
    fitted_vector = tfidf_vectorizer.fit_transform(text)
    # Summing the TF-weights for all words for each review string.
    Reviews_value = fitted_vector.sum(axis=1)
    # converting the matrix to a data frame to facilitate maipulating it.
    Reviews_value = pd.DataFrame(Reviews_value, columns=['TF-IDF Review Value'])
    """# Saving Data into a csv file.
    Reviews_value.to_csv('review_tfidf.csv')
    !cp review_tfidf.csv drive/My\ Drive/"""

    return Reviews_value


