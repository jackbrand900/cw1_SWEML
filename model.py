#!/usr/bin/env python3

import argparse
import csv
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import fbeta_score

AKI_COLUMN = "aki"
SEX_COLUMN = "sex"
CREATININE_PREFIX = "creatinine_result_"
TRAIN_PATH = "./training.csv"

def read_data(data_path):
    data = pd.read_csv(data_path)
    return data

def fill_empty(data, columns):
    filled_data = data.copy()
    for col in columns:
        filled_data[col].replace('', np.nan, inplace=True)
    return filled_data

def preprocess(data):
    creatinine_col_names = [col for col in data.columns if col.startswith(CREATININE_PREFIX)]
    processed_data = fill_empty(data, creatinine_col_names)

    # building features based on creatinine tests
    creatinine_col_data = processed_data[creatinine_col_names].astype(float)
    min_cre = creatinine_col_data.min(axis=1)
    max_cre = creatinine_col_data.max(axis=1)
    mean_cre = creatinine_col_data.mean(axis=1)
    processed_data['min_cre'] = min_cre
    processed_data['max_cre'] = max_cre
    processed_data['mean_cre'] = mean_cre

    processed_data['sex'] = data[SEX_COLUMN].map({'f': 0, 'm': 1})
    labels = []
    if AKI_COLUMN in processed_data.columns:
        labels = processed_data[AKI_COLUMN].map({'n': 0, 'y': 1}).copy()
        processed_data.drop(columns=[AKI_COLUMN], axis=1, inplace=True)
    return processed_data, labels

def build_model(num_layers, nodes_per_layer, input_shape, activation='relu', output_activation='sigmoid'):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(input_shape)))
    for _ in range(num_layers):
        model.add(tf.keras.layers.Dense(nodes_per_layer, activation=activation))
    model.add(tf.keras.layers.Dense(1, activation=output_activation))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def train(model, X, y):
    return model.fit(X, y, epochs=10)

def test(model, X, y, X_test, y_test):
    predicted_labels = model.predict(X_test)
    f3_score = fbeta_score(predicted_labels, y_test, beta=3)
    return f3_score

def write_labels(predicted_labels, output_path):
    label_df = pd.DataFrame({'aki': ['n' if label == 0 else 'y' for label in predicted_labels]})
    label_df.to_csv(output_path, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="test.csv")
    parser.add_argument("--output", default="aki.csv")
    flags = parser.parse_args()
    training_data = read_data(TRAIN_PATH)
    processed_training_data, training_labels = preprocess(training_data)
    model = build_model(3, 64, processed_training_data.shape[1])
    trained_model = train(model, processed_training_data, training_labels)

    preprocessed_test_data, _ = preprocess(read_data(flags.input))
    predicted_labels = trained_model.predict(preprocessed_test_data)

    f3_score = test(trained_model, processed_training_data, training_labels, preprocessed_test_data, predicted_labels)
    print(f"F3 score: {f3_score}")

    write_labels(predicted_labels, flags.output)

if __name__ == "__main__":
    main()