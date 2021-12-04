# AIO2 TF-IDF Baseline

This is a very simple question answering system, which is developed as a lightweight baseline for AIO2 competition.

In training stage, the model builds a sparse matrix of TF-IDF features from the questions in training dataset.
In inference stage, the model predicts answers of unseen questions by finding the most similar training question to the input one by computing dot product of TF-IDF features.
Therefore, the model will not be able to predict completely unseen answers.

## Steps to experiment with the model

### Install requirements

```sh
$ pip install -r requirements.txt
```

### Train

```sh
$ python train.py \
--train_file <data dir>/aio_02_train.jsonl \
--output_dir model \
--pos_list 名詞 \
--stop_words でしょ う \
--max_features 10000
```

### Predict

```sh
$ python predict.py \
--model_dir model \
--test_file <data dir>/aio_02_dev_unlabeled_v1.0.jsonl \
--prediction_file <output dir>/predictions.jsonl
```

## Building Docker image

```sh
$ docker build -t aio2-tfidf-baseline .
```

Test locally:

```sh
$ docker run --rm -v "<input_dir>:/app/input" -v "<output_dir>:/app/output" aio2-tfidf-baseline bash ./submission.sh input/aio_02_dev_unlabeled_v1.0.jsonl output/predictions.jsonl
```
