import pickle
import sys
from typing import List

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier

from download import TITANIC_DATA_FOLDER

FEATURES = ["Pclass", "Sex", "SibSp", "Parch"]


def get_df(file_name):
    """Read the input data file and return a data frame."""
    df = pd.read_csv(file_name)
    sys.stderr.write(f"The input data frame {file_name} size is {df.shape}\n")
    return df


def save_to_pickel(model, file_path: str):
    with open(file_path, "wb") as fd:
        pickle.dump(model, fd)


def generate_and_save_train_features(
    train_input: str, train_output: str, features: List[str]
):
    """Creates model and saves to file

    Args:
        train_input (str): Train input file name.
        train_output (str): Train output file name.
        features (List[str]): list of features
    """
    df_train = get_df(train_input)

    y = df_train["Survived"]
    X = pd.get_dummies(df_train[features])

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(X, y)

    save_to_pickel(model, train_output)
    print("saved train to pickel")


def main():
    # TODO add YAML params to model
    # params = yaml.safe_load(open("params.yaml"))["featurize"]

    generate_and_save_train_features(
        f"{TITANIC_DATA_FOLDER}/train.csv",
        f"{TITANIC_DATA_FOLDER}/output/train.pkl",
        FEATURES,
    )


if __name__ == "__main__":
    main()
