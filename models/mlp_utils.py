import tensorflow as tf
from tensorflow.keras import layers, Input, Sequential, Model
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score


def getData(train_path, test_path):
    
    #np.random.seed(2)

    train = pd.read_json(train_path)
    test = pd.read_json(test_path)

    train1 = train[train["has_spoiler"] == True] 
    train0 = train[train["has_spoiler"] == False]
    train = pd.concat([train1, train0[:100000]], axis=0)#.sample(frac=1).reset_index(drop=True)

    test_1 = test[test["has_spoiler"] == True]
    test_0 = test[test["has_spoiler"] == False]
    test = pd.concat([test_1, test_0[:25000]], axis=0)#.sample(frac=1).reset_index(drop=True)

    X_train, y_train = train.drop(columns=["has_spoiler"]), train["has_spoiler"]
    X_test, y_test = test.drop(columns=["has_spoiler"]), test["has_spoiler"]


    inputs_train = {
    "user_id": X_train["user_id"].values,
    "book_id": X_train["book_id"].values,
    "numerics": X_train[["rating", "n_votes", "n_comments"]].values
    }

    inputs_test = {
    "user_id": X_test["user_id"].values,
    "book_id": X_test["book_id"].values,
    "numerics": X_test[["rating", "n_votes", "n_comments"]].values
    }

    user_max = X_train.user_id.max()
    book_max = X_train.book_id.max()

    return inputs_train, y_train, inputs_test, y_test, user_max, book_max