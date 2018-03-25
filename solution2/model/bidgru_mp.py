# -*- coding: utf-8 -*-
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import GRU, Dropout, Bidirectional, GlobalMaxPooling1D
from model.attention import Attention


def BidGru(maxlen, max_features, embed_size, embedding_matrix):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix],
                  trainable=False)(inp)
    x = Bidirectional(GRU(300, return_sequences=True, dropout=0.25,
                          recurrent_dropout=0.25))(x)
    x = Bidirectional(GRU(300, return_sequences=True, dropout=0.25,
                          recurrent_dropout=0.25))(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.25)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)

    return model
