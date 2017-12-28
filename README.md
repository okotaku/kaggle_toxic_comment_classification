# Solution1
- feature: tfidf vector
- model: 5folds Rogistic LogisticRegression
- LB 0.53

# Solution2
- feature: 128 word embedding
- model: BidLstm
- valloss: 0.0495
- LB: 0.52

- feature: 128 word embedding
- model: LSTM
- valloss: 0.0499
- LB: 0.53

- feature: 128 word embedding
- model: Gru
- valloss: 0.0504
- LB: 0.53

- three models ensemble
- LB: 0.51
