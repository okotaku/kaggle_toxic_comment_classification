import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


label = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

y = pd.read_csv("data/train.csv")

dic = {l: [] for l in label}
dic_test = {l: [] for l in label}
path = glob.glob("val/*.csv")

for i, p in enumerate(path):
    df = pd.read_csv(p)
    dftest = pd.read_csv(p.replace("val", "result").replace("_val", "").replace("_result", ""))
    for l in label:
        dic[l].append(df.loc[:, l].values)
        dic_test[l].append(dftest.loc[:, l].values)

y = pd.merge(df.drop(label, axis=1), y, on="id")

result = []
folds = KFold(n_splits=10, shuffle=True, random_state=7)
pred_test = np.zeros((len(np.array(dic_test[l]).T), len(label)))
#result = pd.DataFrame({"id": train_id})
train_roc, val_roc = [], []

for i, l in enumerate(label):
    print(l)
    x = np.array(dic[l]).T
    test_x = np.array(dic_test[l]).T
    y_ = y.loc[:, l].values.reshape(-1)
    for train_idx, test_idx in folds.split(x):
        xtr = x[train_idx]
        ytr = y_[train_idx]
        xval = x[test_idx]
        yval = y_[test_idx]
        model = LogisticRegression(C=9.0)
        model.fit(xtr, ytr)
        pred_train = model.predict_proba(xtr)[:, 1]
        loss_train = roc_auc_score(ytr, pred_train)
        pred_val = model.predict_proba(xval)[:, 1]
        loss_val = roc_auc_score(yval, pred_val)
        pred_test[:, i] += model.predict_proba(test_x)[:, 1]
        train_roc.append(loss_train)
        val_roc.append(loss_val)
        print("train loss:", loss_train, "test loss", loss_val)

sub = pd.read_csv("data/sample_submission.csv")
result_df = pd.DataFrame(pred_test/10, columns=label)
pd.concat((sub["id"], result_df), axis=1).to_csv("stack.csv", index=False)
