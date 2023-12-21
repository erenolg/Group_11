import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
import numpy as np
import yaml

def read_config():

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config

def data_preprocessing():
    """
    1- Merges two datasets
    2- Reduces merged dataset while protecting samples with label-1
    3- Applies preprocessing
    """

    config = read_config() # read necessary information
    
    data_raw_path = config["data_raw"]
    data_parsed_path = config["data_parsed"]

    # determines how much we will reduce dataset,you can change from config file
    num_of_books = config["num_of_books"] 

    # read data
    data_with_raw = pd.read_json(data_raw_path, lines=True)
    data_with_parsed = pd.read_json(data_parsed_path, lines=True)
    new_df = pd.merge(data_with_parsed, data_with_raw, on=["review_id"])

    print("Datasets are merged..")

    new_df = new_df.rename(columns={"rating_x": "rating", "user_id_x": "user_id", "rating_x": "rating",
                       "book_id_x": "book_id",})
    selected_cols = ["user_id", "book_id", "review_sentences", "rating", 
                    "timestamp", "n_votes", "n_comments", "has_spoiler"]
    new_df = new_df[selected_cols]
    select_number = num_of_books 

    # we will drop the rows with book_id having the most label-0's (to balance dataset)
    chosen_books = new_df.groupby("book_id")["has_spoiler"].mean().sort_values(ascending=False).keys()[:select_number].tolist()
    reduced_df = new_df[new_df.book_id.isin(chosen_books)]
    reduced_df = reduced_df.reset_index(drop=True)

    print("New dataset is reduced..")

    train, test = train_test_split(reduced_df, test_size=0.2, random_state=42)

    train, mms_votes, mms_comments, le_user, le_book = train_preprocess(train)
    test = test_preprocess(test, mms_votes, mms_comments, le_user, le_book)

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    train.to_json("../data/train_preprocessed.json")
    test.to_json("../data/test.json")

    print("'train_preprocessed' and 'test' sets are ready.")

def train_preprocess(train):

    train["n_votes"] = train["n_votes"].apply(lambda x: 250+(x//100) if x>250 else x)
    train["n_comments"] = train["n_comments"].apply(lambda x: 100+(x//100) if x>100 else x)

    mms_votes = MinMaxScaler()
    mms_comments = MinMaxScaler()

    # OrdinalEncoder is used to handle unknown values in test set
    le_user = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=train.user_id.nunique()+1)
    le_book = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=train.book_id.nunique()+1)

    # Fix inconsistent values
    train["n_votes"] = train["n_votes"].apply(lambda x: max(0,x))
    train["n_comments"] = train["n_comments"].apply(lambda x: max(0,x))
    # convert label from bool to numeric
    train["has_spoiler"] = train["has_spoiler"].apply(lambda x: int(x))
    # minMaxScaling for n_votes and n_comments features
    train["n_votes"] =  mms_votes.fit_transform(train["n_votes"].values.reshape(-1,1))
    train["n_comments"] = mms_comments.fit_transform(train["n_comments"].values.reshape(-1,1))
    # label encoding for user_id (to convert into numeric)
    train["user_id"] = le_user.fit_transform(train["user_id"].values.reshape(-1,1)).astype(int)
    train["book_id"] = le_book.fit_transform(train["book_id"].values.reshape(-1,1)).astype(int)

    train["rating"] = train["rating"].apply(lambda x: x/5)

    return train, mms_votes, mms_comments, le_user, le_book

def test_preprocess(test, mms_votes, mms_comments, le_user, le_book):

    # Applies the steps applied to the training data to the test set 

    test["n_votes"] = test["n_votes"].apply(lambda x: 250+(x//100) if x>250 else x)
    test["n_comments"] = test["n_comments"].apply(lambda x: 100+(x//100) if x>100 else x)
    
    test["n_votes"] = test["n_votes"].apply(lambda x: max(0,x))
    test["n_comments"] = test["n_comments"].apply(lambda x: max(0,x))
    # convert label into numeric
    test["has_spoiler"] = test["has_spoiler"].apply(lambda x: int(x))
    # minMaxScaling for n_votes and n_comments features (only transform, not fit)
    test["n_votes"] =  mms_votes.transform(test["n_votes"].values.reshape(-1,1))
    test["n_comments"] = mms_comments.transform(test["n_comments"].values.reshape(-1,1))
    # label encoding for user_id (to convert into numeric)
    test["user_id"] = le_user.transform(test["user_id"].values.reshape(-1,1)).astype(int)
    test["book_id"] = le_book.transform(test["book_id"].values.reshape(-1,1)).astype(int)
    # scale rating feature 
    test["rating"] = test["rating"].apply(lambda x: x/5)
    
    return test