from featureExtraction import *
from fakeNews import *


def basic_tests(train_data, train_labels, test_data, test_labels):
    clf = train_logistic_regression(train_data, train_labels)
    return test_classifier(clf, test_data, test_labels, "Logistic Regression: test data")


raw_data, labels = get_news_dataset()

raw_train_data, raw_test_data, train_labels, test_labels = split_data(raw_data, labels)


print("\nCount word only\n")

count_train_data, count_test_data = get_CountVector(raw_data, raw_train_data, raw_test_data)

count_basic_tests = basic_tests(count_train_data, train_labels, count_test_data, test_labels)


print("\nTFIDF word only\n")

tfidf_train_data, tfidf_test_data = get_TFIDF_Word(raw_data, raw_train_data, raw_test_data)

tfidf_basic_tests = basic_tests(tfidf_train_data, train_labels, tfidf_test_data, test_labels)


print("\nStop words removed + Count\n")

stop_raw_data = remove_NLTK_Stop1(raw_data)
stop_raw_train_data = remove_NLTK_Stop1(raw_train_data)
stop_raw_test_data = remove_NLTK_Stop1(raw_test_data)

stop_train_data, stop_test_data = get_CountVector_NLTK_Stop(stop_raw_data, stop_raw_train_data, stop_raw_test_data)

stop_basic_tests = basic_tests(stop_train_data, train_labels, stop_test_data, test_labels)


print("\nLemma + Count\n")

train_data1 = tag_and_lem_list(raw_train_data)
test_data1 = tag_and_lem_list(raw_test_data)

lemma_train_data, lemma_test_data = get_CountVector(raw_data, train_data1, test_data1)

lemma_basic_tests = basic_tests(lemma_train_data, train_labels, lemma_test_data, test_labels)


print("\nStop word removal + Lemma + Count\n")

train_data2 = tag_and_lem_list(stop_raw_train_data)
test_data2 = tag_and_lem_list(stop_raw_test_data)

stop_lemma_train_data, stop_lemma_test_data = get_CountVector(stop_raw_data, train_data2, test_data2)

stop_lemma_basic_tests = basic_tests(stop_lemma_train_data, train_labels, stop_lemma_test_data, test_labels)