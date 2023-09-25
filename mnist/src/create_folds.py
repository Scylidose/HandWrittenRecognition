import pandas as pd
from sklearn import model_selection
import argparse

import config

def create_fold(number_fold):
    data = pd.read_csv(config.DATA_FILE)

    data["kfold"] = -1

    data = data.sample(frac=1).reset_index(drop=True)

    kf = model_selection.KFold(n_splits=number_fold)

    for fold, (trn_, val_) in enumerate(kf.split(X=data)):
        data.loc[val_, 'kfold'] = fold

    data.to_csv(config.TRAINING_FILE, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold", type=int)

    args = parser.parse_args()

    create_fold(number_fold=args.fold)