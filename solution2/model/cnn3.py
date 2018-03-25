from keras.models import Model
from keras.layers import Dense, Embedding, Input, concatenate
from keras.layers import Dropout
from keras.layers import Conv1D, GlobalMaxPool1D


def Conv3(maxlen, max_features, embed_size, embedding_matrix):
    inp = Input(shape=(maxlen, ))
    emb = Embedding(max_features, embed_size, weights=[embedding_matrix],
                    trainable=False)(inp)
    x1 = Conv1D(filters=64, kernel_size=4, padding='same', activation='relu')(emb)
    x1 = Dropout(0.2)(x1)
    x1 = Conv1D(filters=128, kernel_size=4, padding='same', activation='relu')(x1)
    x1 = Dropout(0.2)(x1)
    x1 = Conv1D(filters=256, kernel_size=4, padding='same', activation='relu')(x1)
    x1 = GlobalMaxPool1D()(x1)

    x2 = Conv1D(filters=64, kernel_size=6, padding='same', activation='relu')(emb)
    x2 = Dropout(0.2)(x2)
    x2 = Conv1D(filters=128, kernel_size=6, padding='same', activation='relu')(x2)
    x1 = Dropout(0.2)(x1)
    x2 = Conv1D(filters=256, kernel_size=6, padding='same', activation='relu')(x2)
    x2 = GlobalMaxPool1D()(x2)

    x3 = Conv1D(filters=64, kernel_size=8, padding='same', activation='relu')(emb)
    x3 = Dropout(0.2)(x3)
    x3 = Conv1D(filters=128, kernel_size=8, padding='same', activation='relu')(x3)
    x3 = Dropout(0.2)(x3)
    x3 = Conv1D(filters=256, kernel_size=8, padding='same', activation='relu')(x3)
    x3 = GlobalMaxPool1D()(x3)

    x = concatenate([x1, x2, x3])
    x = Dropout(0.2)(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)

    return model
