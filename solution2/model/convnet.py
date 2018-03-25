from keras.models import Model
from keras.layers import Dense, Embedding, Input, Flatten
from keras.layers import LSTM, Bidirectional, Dropout
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPool1D
from keras.layers.merge import add


def ConvNet(maxlen, max_features, embed_size, embedding_matrix):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix],
                  trainable=False)(inp)
    skip = Conv1D(filters=100, kernel_size=1, padding='same', activation='relu')(x)
    x = Conv1D(filters=100, kernel_size=10, padding='same', activation='relu')(x)
    x = Conv1D(filters=100, kernel_size=10, padding='same', activation='relu')(x)
    x = add([x, skip])
    x = MaxPooling1D(pool_size=2)(x)
    skip = Conv1D(filters=100, kernel_size=1, padding='same', activation='relu')(x)
    x = Conv1D(filters=100, kernel_size=10, padding='same', activation='relu')(x)
    x = Conv1D(filters=100, kernel_size=10, padding='same', activation='relu')(x)
    x = add([x, skip])
    x = MaxPooling1D(pool_size=2)(x)
    skip = Conv1D(filters=100, kernel_size=1, padding='same', activation='relu')(x)
    x = Conv1D(filters=100, kernel_size=10, padding='same', activation='relu')(x)
    x = Conv1D(filters=100, kernel_size=10, padding='same', activation='relu')(x)
    x = add([x, skip])
    x = GlobalMaxPool1D()(x)
    #x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.25)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)

    return model
