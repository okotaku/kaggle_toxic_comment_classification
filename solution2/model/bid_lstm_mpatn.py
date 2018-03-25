# -*- coding: utf-8 -*-
from keras.models import Model
from keras.layers import Dense, Embedding, Input, concatenate
from keras.layers import LSTM, Bidirectional, Dropout, GlobalMaxPooling1D
from model.attention import Attention


def BidLstm(maxlen, max_features, embed_size, embedding_matrix):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix],
                  trainable=False)(inp)
    x = Bidirectional(LSTM(300, return_sequences=True, dropout=0.25,
                           recurrent_dropout=0.25))(x)
    x = Bidirectional(LSTM(300, return_sequences=True, dropout=0.25,
                           recurrent_dropout=0.25))(x)
    x1 = Attention(maxlen)(x)
    x2 = GlobalMaxPooling1D()(x)
    x = concatenate([x1, x2])
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.25)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)

    return model
