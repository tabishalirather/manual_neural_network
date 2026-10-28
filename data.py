import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def inputs():
    inputs = [
        [0, 0, 0, 0],  # n=0
        [0, 0, 0, 1],  # n=1
        [0, 0, 1, 0],  # n=2
        [0, 0, 1, 1],  # n=3
        [0, 1, 0, 0],  # n=4
        [0, 1, 0, 1],  # n=5
        [0, 1, 1, 0],  # n=6
        [0, 1, 1, 1],  # n=7
        [1, 0, 0, 0],  # n=8
        [1, 0, 0, 1],  # n=9
    ]
    return inputs


def labels():

    labels = [
        [1, 0],  # n=0 even
        [0, 1],  # n=1 odd
        [1, 0],  # n=2 even
        [0, 1],  # n=3 odd
        [1, 0],  # n=4 even
        [0, 1],  # n=5 odd
        [1, 0],  # n=6 even
        [0, 1],  # n=7 odd
        [1, 0],  # n=8 even
        [0, 1],  # n=9 odd
    ]
    return labels


iris_data = pd.read_csv('data/Iris.csv')
# print(iris_data)


def convert_names_to_labels(iris_data):
    name_col = iris_data["Species"]
#     print(name_col)

    uniq_names, uniq_lables = np.unique(name_col, return_inverse=True)
#     print((uniq_names, uniq_lables))
    hot_labels = np.eye(3)[uniq_lables]
#     print(hot_labels)
#     iris_data['hot_labels'] = list(hot_labels)
    X = iris_data.drop(['Species', 'Id'], axis=1)
    y = hot_labels
#     print(X)
    return [X, y]

#     iris_data = convert_names_to_labels()
#     print(iris_data)

    len(iris_data)


scaler = MinMaxScaler()
X, y = convert_names_to_labels(iris_data)
# test_data, train_data =

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, shuffle=True)

feature_cols = ['SepalLengthCm', 'SepalWidthCm',
                'PetalLengthCm', 'PetalWidthCm']
X_train_norm = X_train.copy()
X_train_norm[feature_cols] = scaler.fit_transform(X_train[feature_cols])


X_test_norm = X_test.copy()
X_test_norm[feature_cols] = scaler.transform(
    X_test_norm[feature_cols])


def iris_train_data():
    return X_train_norm.to_numpy()


def iris_test_data():
    return X_test_norm.to_numpy()


def return_train_labels():
    return y_train


def return_test_labels():
    return y_test
