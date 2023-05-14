import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tagLemmatize import *
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer as Detok


nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


def get_CountVector(all_data, train_data, test_data):
    count_vectorizer = CountVectorizer()
    count_vectorizer = count_vectorizer.fit(all_data)
    x_train_data = count_vectorizer.transform(train_data)
    x_test_data = count_vectorizer.transform(test_data)
    return x_train_data, x_test_data


def remove_NLTK_Stop1(all_data):
    sw = stopwords.words("english")
    deto = Detok()

    all_cleaned = list()

    for article in all_data:
        word_tokens = word_tokenize(article)
        all_cleaned.append(deto.detokenize(
            [w for w in word_tokens if not w in sw]))
    return all_cleaned


def remove_NLTK_Stop3(all_data, train_data, test_data):
    sw = stopwords.words("english")
    deto = Detok()

    all_cleaned = list()
    train_cleaned = list()
    test_cleaned = list()

    for article in all_data:
        word_tokens = word_tokenize(article)
        all_cleaned.append(deto.detokenize(
            [w for w in word_tokens if not w in sw]))

    for article in train_data:
        word_tokens = word_tokenize(article)
        train_cleaned.append(deto.detokenize(
            [w for w in word_tokens if not w in sw]))

    for article in test_data:
        word_tokens = word_tokenize(article)
        test_cleaned.append(deto.detokenize(
            [w for w in word_tokens if not w in sw]))

    return all_cleaned, train_cleaned, test_cleaned


def get_CountVector_NLTK_Stop(all_data, train_data, test_data):
    sw = stopwords.words('english')
    count_vectorizer = CountVectorizer(stop_words=sw)
    count_vectorizer = count_vectorizer.fit(all_data)
    x_train_data = count_vectorizer.transform(train_data)
    x_test_data = count_vectorizer.transform(test_data)
    return x_train_data, x_test_data


def get_TFIDF_Word(all_data, train_data, test_data):
    tfidf_vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vectorizer.fit(all_data)
    x_train_data = tfidf_vectorizer.transform(train_data)
    x_test_data = tfidf_vectorizer.transform(test_data)
    return x_train_data, x_test_data


def tag_and_lem_list(data_list):
    ret_list = []
    for d in data_list:
        ret_list.append(tag_and_lem(d))
    return ret_list
