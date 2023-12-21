import pandas as pd
import numpy as np
np.random.seed(11)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


def read_dataset(train_path, test_path, max_words, balance_train=True, balance_test=False, only_lstm=False, only_mlp=False):
    train = pd.read_json(train_path)
    test = pd.read_json(test_path)

    # balance the train data if needed
    if balance_train:
        train_1 = train[train["has_spoiler"] == True]
        train_0 = train[train["has_spoiler"] == False]
        train = pd.concat([train_1, train_0[:100000]], axis=0).sample(frac=1, random_state=2).reset_index(drop=True)

    # balance the test data if needed
    if balance_test:
        test_1 = test[test["has_spoiler"] == True]
        test_0 = test[test["has_spoiler"] == False]
        test = pd.concat([test_1, test_0[:25000]], axis=0).sample(frac=1, random_state=2).reset_index(drop=True)

    # LSTM DATA
    reviews_train = train["review_sentences"]
    labels_train = train["has_spoiler"]
    reviews_train = [' '.join([sentence for _, sentence in review]) for review in reviews_train]

    reviews_test = test["review_sentences"]
    labels_test = test["has_spoiler"]
    reviews_test = [' '.join([sentence for _, sentence in review]) for review in reviews_test]

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(reviews_train)
    sequences = tokenizer.texts_to_sequences(reviews_train)

    max_sequence_length = max(len(seq) for seq in sequences)
    padded_train = pad_sequences(sequences, maxlen=max_sequence_length)
    sequences_test = tokenizer.texts_to_sequences(reviews_test)
    padded_test = pad_sequences(sequences_test, maxlen=max_sequence_length)

    lstm_data = (padded_train, labels_train, padded_test, labels_test, tokenizer, max_sequence_length)

    # MLP DATA

    mlp_Xtrain, mlp_ytrain = train.drop(columns=["has_spoiler"]), train["has_spoiler"]
    mlp_Xtest, mlp_ytest = test.drop(columns=["has_spoiler"]), test["has_spoiler"]

    inputs_train = {
    "user_id": mlp_Xtrain["user_id"].values,
    "book_id": mlp_Xtrain["book_id"].values,
    "numerics": mlp_Xtrain[["rating", "n_votes", "n_comments"]].values
    }

    inputs_test = {
    "user_id": mlp_Xtest["user_id"].values,
    "book_id": mlp_Xtest["book_id"].values,
    "numerics": mlp_Xtest[["rating", "n_votes", "n_comments"]].values
    }

    user_max = mlp_Xtrain.user_id.max()
    book_max = mlp_Xtrain.book_id.max()

    mlp_data = (inputs_train, mlp_ytrain, inputs_test, mlp_ytest, user_max, book_max)

    if only_lstm:
        return lstm_data
    elif only_mlp:
        return mlp_data
    else:
        return {"lstm_data": lstm_data, "mlp_data": mlp_data}


def embeddings(embeddings_path, embedding_dim, tokenizer, max_words):
    embeddings_index = {}

    with open(embeddings_path, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((max_words, embedding_dim))

    for word, i in tokenizer.word_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    return embedding_matrix


