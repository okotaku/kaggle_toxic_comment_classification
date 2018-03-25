#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy import sparse


if __name__ == "__main__":
    train = pd.read_csv('./data/train_with_convai.csv')
    test = pd.read_csv('./data/test_with_convai.csv')
    feats_to_concat = ['comment_text', 'toxic_level', 'attack', 'aggression']
    alldata = pd.concat([train[feats_to_concat], test[feats_to_concat]],
                        axis=0)
    alldata.comment_text.fillna('unknown', inplace=True)
    vect_words = TfidfVectorizer(max_features=50000, analyzer='word',
                                 ngram_range=(1, 1))
    vect_chars = TfidfVectorizer(max_features=20000, analyzer='char',
                                 ngram_range=(1, 3))
    all_words = vect_words.fit_transform(alldata.comment_text)
    all_chars = vect_chars.fit_transform(alldata.comment_text)
    train_new = train
    test_new = test
    train_words = all_words[:len(train_new)]
    test_words = all_words[len(train_new):]
    train_chars = all_chars[:len(train_new)]
    test_chars = all_chars[len(train_new):]

    feats = ['toxic_level', 'attack']
    train_feats = sparse.hstack([train_words, train_chars,
                                 alldata[feats][:len(train_new)]])
    test_feats = sparse.hstack([test_words, test_chars,
                                alldata[feats][len(train_new):]])

    col = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult',
           'identity_hate']

    preds = np.zeros((test_new.shape[0], len(col)))

    for i, j in enumerate(col):
        print('===Fit '+j)

        model = LogisticRegression(C=4.0, solver='sag')
        print('Fitting model')
        model.fit(train_feats, train_new[j])

        print('Predicting on test')
        preds[:, i] = model.predict_proba(test_feats)[:, 1]

    result_df = pd.DataFrame(preds, columns=col)
    pd.concat((test["id"], result_df), axis=1).to_csv("sublr.csv", index=False)
