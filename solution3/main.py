#!/usr/bin/env python
# -*- coding: utf-8 -*-
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold
np.random.seed(7)

from make_df import make_df
from make_glovevec import make_glovevec
from model.bid_lstm import BidLstm
from model.convaibidgru2l import BidGru


# set gpu usage
config = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list="0",
                                                  allow_growth=True))
session = tf.Session(config=config)
keras.backend.tensorflow_backend.set_session(session)

if __name__ == "__main__":
    max_features = 100000
    maxlen = 150
    embed_size = 300
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult",
                    "identity_hate"]

    xtr, xcontr, xte, xconte, y, word_index, trid = make_df("./data/train_with_convai.csv", "./data/test_with_convai.csv",
                                      max_features, maxlen, list_classes)
    embedding_vector = make_glovevec("./data/glove.840B.300d.txt",
                                     max_features, embed_size, word_index)
    folds = list(KFold(n_splits=10, shuffle=True,
                       random_state=7).split(xtr))
    early = EarlyStopping(monitor="val_loss", mode="min", patience=3)
    result = np.zeros((len(xte), len(list_classes)))
    ans = pd.DataFrame(columns=["id"] + list_classes)
    for i, (train_idx, test_idx) in enumerate(folds):
        x_tr = xtr[train_idx]
        xcon_tr = xcontr[train_idx]
        y_tr = y[train_idx]
        x_val = xtr[test_idx]
        xcon_val = xcontr[test_idx]
        y_val = y[test_idx]
        val_id = trid[test_idx]
        #model = BidLstm(maxlen, max_features, embed_size, embedding_vector)
        model = BidGru(maxlen, max_features, embed_size, embedding_vector)
        #model = Gru(maxlen, max_features, embed_size, embedding_vector)
        model.compile(loss='binary_crossentropy', optimizer='adam',
	              metrics=['accuracy'])
        file_path = ".model1.hdf5"
        ckpt = ModelCheckpoint(file_path, monitor='val_loss', verbose=2,
	                       save_best_only=True, mode='min')
        model.fit([x_tr, xcon_tr], y_tr, batch_size=64, epochs=15,
                  validation_data=([x_val, xcon_val], y_val),
	           callbacks=[ckpt, early], verbose=2)

        model.load_weights(file_path)
        y_test = model.predict([xte, xconte])
        result += y_test
        ans_ = pd.concat((pd.DataFrame(val_id, columns=["id"]),
                          pd.DataFrame(model.predict([x_val, xcon_val]), columns=list_classes)), axis=1)
        ans = pd.concat((ans, ans_), axis=0)
    sample_submission = pd.read_csv("./data/sample_submission.csv")
    sample_submission[list_classes] = result/10
    sample_submission.to_csv("sub1.csv", index=False)
    pd.DataFrame(ans, columns=["id"]+list_classes).to_csv("val1.csv", index=False)
