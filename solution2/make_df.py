# -*- coding: utf-8 -*-
import pandas as pd
from keras.preprocessing import text, sequence


def make_df(train_path, test_path, max_features, maxlen, list_classes):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train = train.sample(frac=1)
    train_id = train["id"].values

    list_sentences_train = train["comment_text"].fillna("unknown").values
    y = train[list_classes].values
    list_sentences_test = test["comment_text"].fillna("unknown").values

    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(list_sentences_train))
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
    X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
    X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

    word_index = tokenizer.word_index

    return X_t, X_te, y, word_index, train_id
