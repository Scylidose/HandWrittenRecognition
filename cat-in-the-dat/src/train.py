import pandas as pd

from sklearn import linear_model, metrics, preprocessing

import config

def run(fold):
    data = pd.read_csv(config.TRAINING_FOLDS_FILE)

    features = [
        f for f in data.columns if f not in ("id", "target", "kfold")
    ]

    for col in features:
        data.loc[:, col] = data[col].astype(str).fillna("NONE")

    train_data = data[data.kfold != fold].reset_index(drop=True)
    valid_data = data[data.kfold == fold].reset_index(drop=True)

    ohe = preprocessing.OneHotEncoder()

    full_data = pd.concat([train_data[features], valid_data[features]], axis=0)

    ohe.fit(full_data[features])

    X_train = ohe.transform(train_data[features])
    X_valid = ohe.transform(valid_data[features])

    model = linear_model.LogisticRegression()

    model.fit(X_train, train_data.target.values)

    y_valid_pred = model.predict_proba(X_valid)[:, 1]

    auc = metrics.roc_auc_score(valid_data.target.values, y_valid_pred)

    print(f"Fold: {fold}, AUC Score: {auc}")


if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)