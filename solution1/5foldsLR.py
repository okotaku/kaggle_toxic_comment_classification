#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold


if __name__ == "__main__":
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult',
                  'identity_hate']
    train_df = pd.read_csv('./data/train.csv')
    test_df = pd.read_csv('./data/test.csv')
    train_df.fillna('unknown', inplace=True)
    test_df.fillna('unknown', inplace=True)
    X = train_df.comment_text
    test_X = test_df.comment_text

    tfidf_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9,
                                strip_accents='unicode', use_idf=1,
                                smooth_idf=1, sublinear_tf=1)
    tfidf_vec.fit(X)
    train_tfidf = tfidf_vec.transform(X)
    test_tfidf = tfidf_vec.transform(test_X)

    folds = KFold(n_splits=5, shuffle=True, random_state=7)
    pred_test = np.zeros((len(test_X), len(label_cols)))
    for i, t in enumerate(label_cols):
        print(t)
        y = train_df.loc[:, [t]].values.reshape(-1)
        for train_idx, test_idx in folds.split(train_tfidf):
            xtr = train_tfidf[train_idx]
            ytr = y[train_idx]
            xval = train_tfidf[test_idx]
            yval = y[test_idx]
            model = LogisticRegression(C=9.0, class_weight='balanced')
            model.fit(xtr, ytr)
            pred_train = model.predict_proba(xtr)
            loss_train = log_loss(ytr, pred_train)
            pred_val = model.predict_proba(xval)
            loss_val = log_loss(yval, pred_val)
            pred_test[:, i] += model.predict_proba(test_tfidf)[:, 1]
            print("train loss:", loss_train, "test loss", loss_val)

    result_df = pd.DataFrame(pred_test/5, columns=label_cols)
    pd.concat((test_df["id"], result_df), axis=1).to_csv("sub.csv", index=False)
