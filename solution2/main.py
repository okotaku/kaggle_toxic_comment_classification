import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import text, sequence
np.random.seed(7)

from model.bid_lstm import BidLstm
from model.lstm import Lstm
from model.gru import Gru


if __name__ == "__main__":
    max_features = 20000
    maxlen = 100

    train = pd.read_csv("./data/train.csv")
    test = pd.read_csv("./data/test.csv")
    train = train.sample(frac=1)

    list_sentences_train = train["comment_text"].fillna("unknown").values
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult",
                    "identity_hate"]
    y = train[list_classes].values
    list_sentences_test = test["comment_text"].fillna("unknown").values

    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(list_sentences_train))
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
    X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
    X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

    model = BidLstm(maxlen, max_features)
    #model = Lstm(maxlen, max_features)
    #model = Gru(maxlen, max_features)
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    file_path = ".model.hdf5"
    ckpt = ModelCheckpoint(file_path, monitor='val_loss', verbose=1,
                           save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    model.fit(X_t, y, batch_size=512, epochs=10, validation_split=0.1,
              callbacks=[ckpt, early])

    model.load_weights(file_path)
    y_test = model.predict(X_te)
    sample_submission = pd.read_csv("./data/sample_submission.csv")
    sample_submission[list_classes] = y_test
    sample_submission.to_csv("sub.csv", index=False)
