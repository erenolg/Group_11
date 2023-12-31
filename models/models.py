import numpy as np
import tensorflow as tf
np.random.seed(11)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional,Dense, Dropout, SpatialDropout1D, BatchNormalization, Flatten, Concatenate
from tensorflow.keras import Input, Model


def LSTM_model(max_words, embedding_dim, embedding_matrix, max_seq_length):
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, weights=[embedding_matrix],
                        input_length=max_seq_length, trainable=False))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    # model.add(BatchNormalization())
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(16, return_sequences=False))
    # model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

def MLP_model(user_max, book_max, user_emb_size=24, book_emb_size=16):
    user_id_input = Input(shape=(1,), name="user_id")
    book_id_input = Input(shape=(1,), name="book_id")
    numerics_input = Input(shape=(3,), name="numerics")

    user_embedding = Embedding(input_dim=user_max + 3, output_dim=user_emb_size, input_length=1, name="user_embedding")(user_id_input)
    book_embedding = Embedding(input_dim=book_max + 3, output_dim=book_emb_size, input_length=1, name="book_embedding")(book_id_input)

    user_flattened = Flatten()(user_embedding)
    book_flattened = Flatten()(book_embedding)

    numerics = Sequential([
        Dense(40, activation="relu"),
        BatchNormalization(),
        Dropout(0.5)
    ])(numerics_input)

    concatenated = Concatenate()([user_flattened, book_flattened, numerics])

    out = BatchNormalization()(Dense(256, activation="relu")(concatenated))
    out = Dropout(0.5)(out)
    out = BatchNormalization()(Dense(64, activation="relu")(out))
    out = Dense(1, activation="sigmoid")(out)


    model = Model(inputs=[user_id_input, book_id_input, numerics_input], outputs=out)
    return model