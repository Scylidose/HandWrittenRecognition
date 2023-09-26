import pandas as pd
from sklearn import model_selection
import argparse


def create_kfold(db_input_path, db_output_path, number_fold):
    data = pd.read_csv(db_input_path)

    data["kfold"] = -1

    data = data.sample(frac=1).reset_index(drop=True)

    kf = model_selection.KFold(n_splits=number_fold)

    for fold, (trn_, val_) in enumerate(kf.split(X=data)):
        data.loc[val_, 'kfold'] = fold

    data.to_csv(db_output_path, index=False)


def create_stratified_kfold(db_input_path, db_output_path, target_name, number_fold):
    data = pd.read_csv(db_input_path)

    data["kfold"] = -1

    data = data.sample(frac=1).reset_index(drop=True)

    y = data[target_name].values

    kf = model_selection.StratifiedKFold(n_splits=number_fold)

    for fold, (trn_, val_) in enumerate(kf.split(X=data, y=y)):
        data.loc[val_, 'kfold'] = fold

    data.to_csv(db_output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--method", type=str, const='kfold', nargs='?')
    parser.add_argument("--target", type=str, nargs='?')

    parser.add_argument("--fold", type=int, const=5, nargs='?')

    args = parser.parse_args()

    if args.method == 'kfold':
        create_kfold(db_input_path=args.input_path, db_output_path=args.output_path, number_fold=args.fold)
    elif args.method =='stratified':
        create_stratified_kfold(db_input_path=args.input_path, db_output_path=args.output_path, target_name=args.target, number_fold=args.fold)