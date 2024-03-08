import pickle

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from dvclive import Live

from download import TITANIC_DATA_FOLDER
from train import FEATURES

EVAL_PATH = "eval"


def evaluate(model, live):
    """
    Dump all evaluation metrics and plots for given datasets.

    Args:
        model (sklearn.ensemble.RandomForestClassifier): Trained classifier.
        split (str): Dataset name.
        live (dvclive.Live): Dvclive instance.
        save_path (str): Path to save the metrics.
    """
    # Load data
    data = pd.read_csv(f"{TITANIC_DATA_FOLDER}/train.csv")
    split = "train"

    # Split data
    X = data.drop("Survived", axis=1)
    y = data["Survived"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test = pd.get_dummies(X_test[FEATURES])

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy}")
    print(classification_report(y_test, predictions))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, predictions)
    roc_auc = auc(fpr, tpr)

    if not live.summary:
        live.summary = {"accuracy": {}, "roc_auc": {}}
    live.summary["accuracy"][split] = accuracy
    live.summary["roc_auc"][split] = roc_auc

    # live.log_sklearn_plot(
    #     "roc",
    #     y_test.values,
    #     predictions,
    #     name="rf_classifier_evaluation.html",
    # )


def load_pickel(filename: str):
    with open(filename, "rb") as fd:
        return pickle.load(fd)


def main():
    # load training dataset
    train = load_pickel(f"{TITANIC_DATA_FOLDER}/output/train.pkl")

    # Evaluate train and test datasets.
    with Live(EVAL_PATH, dvcyaml=False) as live:
        evaluate(train, live)


if __name__ == "__main__":
    main()
