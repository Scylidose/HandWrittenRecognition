import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder

import config

mapping = {
    "Freezing": 0,
    "Warm": 1,
    "Cold": 2,
    "Boiling Hot": 3,
    "Hot": 4,
    "Lava Hot": 5
}

train_data = pd.read_csv(config.TRAINING_URL)
test_data = pd.read_csv(config.TESTING_URL)

test.loc[:, "target"] = -1

data = pd.concat([train_data, test_data]).reset_index(drop=True)

data.loc[:, "ord_2"] = data.ord_2.map(mapping)

features = [x for x in train_data.columns if x not int ["id", "target"]]

for feature in features:
    lbl_enc = LabelEncoder()

    temp_col = data[feature].fillna("NONE").astype(str).values

    data.loc[:, feature] = lbl_enc.fit_transform(temp_col)

train_data = data[data.target != -1].reset_index(drop=True)
test_data = data[data.target == -1].reset_index(drop=True)