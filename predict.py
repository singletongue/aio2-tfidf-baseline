import argparse
import gzip
import json
import os
import pickle


def main(args):
    # load the processed train dataset and the question vectors
    train_dataset_file = os.path.join(args.model_dir, "train_dataset.json.gz")
    train_dataset = json.load(gzip.open(train_dataset_file, "rt"))

    question_vectorizer_file = os.path.join(args.model_dir, "question_vectorizer.pickle")
    question_vectorizer = pickle.load(open(question_vectorizer_file, "rb"))

    question_vectors_file = os.path.join(args.model_dir, "question_vectors.pickle")
    train_question_vectors = pickle.load(open(question_vectors_file, "rb"))

    # load each item in the test dataset and predict its answer
    with open(args.test_file) as f, open(args.prediction_file, "w") as fo:
        for line in f:
            item = json.loads(line)
            test_question_vector = question_vectorizer.transform([item["question"]])
            tfidf_scores = train_question_vectors.dot(test_question_vector.toarray().T)[:, None]
            top_score_index = tfidf_scores.argmax().item()
            pred_answer = train_dataset[top_score_index]["answers"][0]

            print(json.dumps({"qid": item["qid"], "prediction": pred_answer}, ensure_ascii=False), file=fo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Directory of trained model and data files.")
    parser.add_argument("--test_file", type=str, required=True, help="Input test dataset file.")
    parser.add_argument("--prediction_file", type=str, required=True, help="Output prediction file.")
    args = parser.parse_args()
    main(args)
