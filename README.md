# Solution1
- model: 5folds Rogistic LogisticRegression
  - feature: tfidf vector
  - LB 0.53

# Solution2
-  model: BidLstm
  - feature: 128 word embedding
  - valloss: 0.0495
  - LB: 0.52

- model: LSTM
  - feature: 128 word embedding
  - valloss: 0.0499
  - LB: 0.53

- model: Gru
  - feature: 128 word embedding
  - valloss: 0.0504
  - LB: 0.53

- three models ensemble
  - LB: 0.51
