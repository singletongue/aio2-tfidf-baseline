import argparse
import gzip
import json
import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

from utils.analyzer import JapaneseTextAnalyzer


def main(args):
    # load the training dataset
    print("Loading dataset")
    train_dataset = []
    with open(args.train_file) as f:
        for line in f:
            item = json.loads(line)
            train_dataset.append(item)

    # vectorize the questions by tf-idf
    print("Initializing vectorizer")
    analyzer = JapaneseTextAnalyzer(pos_list=args.pos_list, stop_words=args.stop_words)
    question_vectorizer = TfidfVectorizer(
        lowercase=args.do_lowercase, analyzer=analyzer, ngram_range=(1, 1), max_features=args.max_features
    )
    print("Vectoring questions")
    question_vectors = question_vectorizer.fit_transform(item["question"] for item in train_dataset)

    # output the created files
    print("Saving models to file")
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    train_dataset_file = os.path.join(args.output_dir, "train_dataset.json.gz")
    with gzip.open(train_dataset_file, "wt") as fo:
        json.dump(train_dataset, fo, ensure_ascii=False)

    question_vectorizer_file = os.path.join(args.output_dir, "question_vectorizer.pickle")
    with open(question_vectorizer_file, "wb") as fo:
        pickle.dump(question_vectorizer, fo)

    question_vectors_file = os.path.join(args.output_dir, "question_vectors.pickle")
    with open(question_vectors_file, "wb") as fo:
        pickle.dump(question_vectors, fo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True, help="Input training dataset file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to output model and data files.")
    parser.add_argument("--do_lowercase", action="store_true", help="Lowercase questions when indexing.")
    parser.add_argument("--pos_list", type=str, nargs="*", help="POS tags to filter question tokens.")
    parser.add_argument("--stop_words", type=str, nargs="*", help="Stop words to filter out from questions.")
    parser.add_argument("--max_features", type=int, default=10000, help="Maximum number of token features.")
    args = parser.parse_args()
    main(args)
