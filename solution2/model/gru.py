from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import GRU, GlobalMaxPool1D, Dropout


def Gru(maxlen, max_features):
    embed_size = 128
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(inp)
    x = GRU(50, return_sequences=True)(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.3)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)

    return model
