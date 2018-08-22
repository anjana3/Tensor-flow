import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from tensorflow import keras
from docopt import docopt
import io
from sklearn.model_selection import train_test_split
# reading the dataset
import csv
import sys
import argparse
import pandas as pd


class Test(object):
    def split_data(self, csv_file, target_label, test_size):
        data = pd.read_csv(csv_file)
        X = data.drop(target_label, axis=1)
        y = data[target_label]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=5
        )
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        train_data = (X_train - mean) / std
        test_data = (X_test - mean) / std
        model = keras.Sequential(
            [
                keras.layers.Dense(
                    64, activation=tf.nn.relu, input_shape=(train_data.shape[1],)
                ),
                keras.layers.Dense(64, activation=tf.nn.relu),
                keras.layers.Dense(1),
            ]
        )

        optimizer = tf.train.RMSPropOptimizer(0.001)

        model.compile(loss="mse", optimizer=optimizer, metrics=["mae"])
        model.fit(
            X_train, y_train, epochs=500, validation_split=0.2, verbose=0
        )
        [loss, mae] = model.evaluate(test_data, y_test, verbose=0)
        print(mae, loss)
        print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))

        test_predictions = model.predict(X_test).flatten()

        print(test_predictions)
        return test_predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group()
    required_args.add_argument(
        "-i", "--csv_file", dest="csv_file", required=True)
    required_args.add_argument(
        "-t", "--target_label", dest="target_label", required=True
    )
    required_args.add_argument(
        "-s", "--test_size", dest="test_size", type=float, required=True)
    arguments = parser.parse_args()
    tr_obj = Test()
    tr_obj.split_data(arguments.csv_file,
                      arguments.target_label, arguments.test_size)
