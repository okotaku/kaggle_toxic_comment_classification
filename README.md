# Solution1
ベーシックなモデルとして、TFIDFを特徴量としたロジスティック回帰を行なった。

各モデルAttentionを加え、word embedingはgloveの学習済みモデル(glove.840B.300d.zip)を使用。
https://github.com/stanfordnlp/GloVe

-  model: bidlstm
  - Private=0.9841, Public=0.9837

-  model: lstm
  - Private=0.9839, Public=0.9837


-  model: bidgru
  - Private=0.9839, Public=0.9839

-  model: gru
  - Private=0.9834, Public=0.9833

-  model: bidrwa
  - Private=0.9840, Public=0.9838

-  model: rwa
  - Private=0.9843, Public=0.9839

-  model: bidlstm2l
  - Private=0.9849, Public=0.9845

-  model: lstm2l
  - Private=0.9844, Public=0.9844

-  model: bidgru2l
  - Private=0.9847, Public=0.9843

-  model: gru2l
  - Private=0.9830, Public=0.9829

-  model: bidlstm3l
  - Private=0.9840, Public=0.9838

-  model: lstm3l
  - Private=0.9833, Public=0.9830

-  model: gru3l
  - Private=0.9810, Public=0.9810

-  model: bidgru3l
  - Private=0.9840, Public=0.9836

-  model: bidlstm2l_mp
  - Private=0.9849, Public=0.9845

-  model: bidlstm2l_ap
  - Private=0.9847, Public=0.9843

-  model: bidlstm2l_amp
  - Private=0.9848, Public=0.9846

-  model: bidlstm2l_mpatn
  - Private=0.9849, Public=0.9845

-  model: bidgru2l_mp
  - Private=0.9848, Public=0.9843

# Solution3
-  model: convaibidlstm2l
  - Private=0.9848, Public=0.9844

-  model: convailstm2l
  - Private=0.9844, Public=0.9844

-  model: convaibidgru2l
  - Private=0.9846, Public=0.9842

-  model: convaigru2l
  - Private=0.9831, Public=0.9829

# 参考文献
https://www.kaggle.com/tunguz/logistic-regression-with-words-and-char-n-grams
https://www.kaggle.com/yekenot/pooled-gru-fasttext
https://www.kaggle.com/chongjiujjin/capsule-net-with-gru
https://www.kaggle.com/ogrellier/lgbm-with-words-and-chars-n-gram
https://www.kaggle.com/eashish/bidirectional-gru-with-convolution
https://www.kaggle.com/yekenot/textcnn-2d-convolution
https://www.kaggle.com/fizzbuzz/bi-lstm-conv-layer-lb-score-0-9840
https://www.kaggle.com/michaelsnell/conv1d-dpcnn-in-keras
