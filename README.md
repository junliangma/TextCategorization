# DocumentFilter

## Description

単純ベイズ分類器を利用したドキュメントフィルタです．

- 英語と日本語に対応
- 不均衡データに対応
- 評価に F 値を使用

### Naive Bayes Classifier

#### Advantages

- 学習と予測を高速に行うことができる
- 各特徴量の影響を比較的単純に解釈できる

#### Disadvantages

- 特徴量の複雑な関係を捕捉できない

## Environments

- Python 3.5.0
- mecab-python3 0.7
- numpy 1.10.1
- PyStemmer 1.3.0
- pandas 0.17.0
- scikit-learn 0.17
- Mecab 0.996

## Usage

```
$ python3 tune.py LANG POS_LABEL TFILE EFILE
```

```
$ python3 predict.py IFILE EFILE OFILE
```

- POS_LABEL: 陽性ラベル
- LANG: 言語
- TFILE:  訓練データ
- EFILE: 分類器
- IFILE: 入力データ
- OFILE: 予測結果

## Datasets

- UCI Machine Learning Repository, ["SMS Spam Collection"](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection), 2012.

    - n: 5574
    - n_neg: 4827
    - n_pos: 747
    - best F1 score: 0.934

- [自作メンヘラデータセット](./trainingdata/menhera_dataset.tsv)

    - n: 573
    - n_neg: 89
    - n_pos: 484
    - best F1 score: 0.970

## Future works

- Graham 法の実装
- Robinson 法の実装
- Robinson-Fisher 法の実装
- 多クラス分類への対応
- 多言語環境への対応
- 閾値の導入
- SQLite の利用
- Cython の利用

## References

- P. Graham, ["Better Bayesian Filtering"](http://www.paulgraham.com/better.html), 2003.
- P. Graham, ["A Plan for SPAM"](http://www.paulgraham.com/spam.html), 2006.
- G. Robinson, "Spam Detection", 2006.
- T. Tabata, "SPAM mail filtering: commentary of Bayesian fileter", 2006.
- T.Segaran, "Programming Collective Intelligence: Building Smart Web 2.0 Applications", 2007.
- W. Richert, L. Coelho, ["Building Machine Learning Systems with Python"](http://github.com/luispedro/BuildingMachineLearningSystemsWithPython), 2013.
