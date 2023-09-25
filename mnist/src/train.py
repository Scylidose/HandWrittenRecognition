import os
import argparse
import joblib
import pandas as pd
from sklearn import metrics, tree

import config
import model_dispatcher

def run(fold, model, save_model):
    print("Training starting...")

    data = pd.read_csv(config.TRAINING_FILE)

    train_data = data[data.kfold != fold].reset_index(drop=True)
    valid_data = data[data.kfold == fold].reset_index(drop=True)

    X_train = train_data.drop("class", axis=1).values
    y_train = train_data["class"].values

    X_valid = valid_data.drop("class", axis=1).values
    y_valid = valid_data["class"].values

    clf = model_dispatcher.models[model]
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_valid)

    accuracy = metrics.accuracy_score(y_valid, y_pred)
    print(f"Fold: {fold}, Accuracy: {accuracy}")

    if save_model:
        joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold", type=int, const=0, nargs='?')
    parser.add_argument("--model", type=str, const='rf', nargs='?')
    parser.add_argument("--save_model", type=str, const='false', nargs='?')

    args = parser.parse_args()

    run(fold=args.fold, model=args.model, save_model=args.save_model)

