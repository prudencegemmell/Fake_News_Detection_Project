import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def get_news_dataset():
    df_true = pd.read_csv("true.csv", usecols=["title"])
    df_true["Number of Words"] = df_true["title"].apply(lambda n: len(n.split()))
    df_true["Fake"] = 0

    df_fake = pd.read_csv("fake.csv", usecols=["title"])
    df_fake["Number of Words"] = df_fake["title"].apply(lambda n: len(n.split()))
    df_fake["Fake"] = 1

    header = ["title", "Number of Words", "Fake"]
    df_master = df_true.merge(df_fake, on=header, how="outer")

    ml_data = df_master["title"].values.tolist()
    ml_labels = df_master["Fake"].values.tolist()
    return ml_data, ml_labels


def split_data(data, labels):
    return train_test_split(data, labels, test_size=0.3, random_state=0, shuffle="true")


def train_logistic_regression(train_data, train_labels):
    return LogisticRegression(max_iter=10000).fit(train_data, train_labels)


def test_classifier(clf, test_data, test_labels, header):
    predicted = clf.predict(test_data)
    print(header)
    print(np.mean(predicted == test_labels))
