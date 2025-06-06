import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.api import layers, models
from keras.api.callbacks import EarlyStopping
from keras.api.metrics import AUC, F1Score, Precision, Recall
from keras.api.optimizers import Adam
from keras.api.utils import to_categorical
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import GridSearchCV, KFold, train_test_split

from scripts.ExperimentPipeline import ExperimentPipeline


class BaselineSentimentClassificationPipeline(ExperimentPipeline):
    def build_model(self, params):
        model = models.Sequential(name="sentiment_model")
        model.add(layers.Input(shape=(26, 300), name="Input"))
        model.add(layers.Flatten())
        model.add(
            layers.Dense(
                params.get("first_units", 200), activation="relu", name="layer_1"
            )
        )
        model.add(layers.Dropout(params.get("dropout_rate", 0.3), name="dropout_3"))
        model.add(layers.BatchNormalization(name="batch_normalization_3"))
        model.add(
            layers.Dense(
                params.get("second_units", 200), activation="relu", name="layer_2"
            )
        )
        model.add(layers.Dropout(params.get("dropout_rate", 0.3), name="dropout_4"))
        model.add(layers.BatchNormalization(name="batch_normalization_4"))
        model.add(layers.Dense(3, activation="softmax", name="sentiment"))

        model.compile(
            optimizer=Adam(learning_rate=params.get("learning_rate", 0.01)),
            loss="categorical_crossentropy",
            metrics=[
                "accuracy",
                Recall(),
                Precision(),
                AUC(multi_label=True),
                F1Score(average="weighted"),
            ],
        )
        return model

    def load_data(self):
        df_original = pd.read_csv("data/processed/processed_tweets.csv")
        padded_vectors = np.load("data/processed/padded_glove_vectors.npy")
        return (df_original, padded_vectors)

    def preprocess_data(self, data):
        df, padded_vectors = data
        df = df.dropna(subset=["Sentiment"]).copy()
        df["Sentiment"] = df["Sentiment"].map({-1: 0, 0: 1, 1: 2})
        X = padded_vectors[:800]
        y = df["Sentiment"].values
        return (np.array(X), np.array(y))

    def train_model(
        self, model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32
    ):
        y_train_enc = to_categorical(y_train, num_classes=3)
        y_val_enc = to_categorical(y_val, num_classes=3)

        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_enc))
        train_ds = train_ds.batch(batch_size, drop_remainder=True)

        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val_enc))
        val_ds = val_ds.batch(batch_size, drop_remainder=True)

        callback = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
        history = model.fit(
            X_train,
            y_train_enc,
            validation_data=(X_val, y_val_enc),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[callback],
            verbose=0,
        )
        return model, history

    def save_model(self, model, path):
        model.save(path)
        print(f"Model saved to {path}")

    def optimize_hyperparameters(self, data, param_grid):
        X, y = data

        y_train_enc = to_categorical(y, num_classes=3)

        def create_model(first_units, second_units, dropout_rate, learning_rate):
            params = {
                "first_units": first_units,
                "second_units": second_units,
                "dropout_rate": dropout_rate,
                "learning_rate": learning_rate,
            }
            return self.build_model(params)

        keras_clf = KerasClassifier(
            model=create_model, epochs=10, batch_size=32, verbose=0
        )
        grid = GridSearchCV(
            estimator=keras_clf,
            param_grid=param_grid,
            cv=5,
            scoring="f1_weighted",
            verbose=1,
        )
        grid_result = grid.fit(X, y_train_enc)
        best_params = grid_result.best_params_
        print("Best validation accuracy from GridSearch:", grid_result.best_score_)
        return best_params

    def cross_validation(self, data, params, n_folds=5, epochs=10, batch_size=32):
        X, y = data
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=None)
        fold_metrics = []
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            model = self.build_model(params)
            model, _ = self.train_model(
                model,
                X_train,
                y_train,
                X_val,
                y_val,
                epochs=epochs,
                batch_size=batch_size,
            )
            y_val_enc = to_categorical(y_val, num_classes=3)
            results = model.evaluate(X_val, y_val_enc, verbose=0)

            metrics = {
                "loss": results[0],
                "accuracy": results[1],
                "recall": results[2],
                "precision": results[3],
                "auc": results[4],
                "f1": results[5],
            }
            fold_metrics.append(metrics)
        avg_metrics = {
            key: np.mean([m[key] for m in fold_metrics])
            for key in fold_metrics[0].keys()
        }
        return avg_metrics


if __name__ == "__main__":
    os.makedirs("plots/baseline_sentiment_classification", exist_ok=True)

    param_grid = {
        "model__first_units": [200, 250, 300],
        "model__second_units": [100, 150, 200],
        "model__dropout_rate": [0.2, 0.3, 0.5],
        "model__learning_rate": [0.001, 0.01, 0.0001],
    }

    pipeline = BaselineSentimentClassificationPipeline()
    overall_avg_metrics, best_params = pipeline.run(
        n_runs=10, param_grid=param_grid, optimize_hyperparameters=False
    )

    with open(
        "results/baseline/sentiment_classification/results_metrics.txt", "w"
    ) as f:
        f.write("Overall Average Metrics Across Runs:\n")
        for key, value in overall_avg_metrics.items():
            f.write(f"{key}: {value}\n")
        f.write("\nBest Hyperparameters:\n")
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")
    print("Overall results saved to results_metrics.txt")

    X, y = pipeline.preprocess_data(pipeline.load_data())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=None, stratify=y
    )
    y_train_enc = to_categorical(y_train, num_classes=3)
    y_test_enc = to_categorical(y_test, num_classes=3)
    final_model = pipeline.build_model(best_params)

    final_model, final_history = pipeline.train_model(
        final_model, X_train, y_train, X_test, y_test, epochs=10, batch_size=32
    )

    metrics_to_plot = ["accuracy", "loss", "recall", "precision", "auc", "f1_score"]
    for metric in metrics_to_plot:
        plt.figure()
        if metric in final_history.history:
            plt.plot(final_history.history[metric])
            plt.title(f"Training {metric.capitalize()}")
            plt.xlabel("Epoch")
            plt.ylabel(metric.capitalize())
            plt.savefig(f"plots/baseline_sentiment_classification/{metric}_plot.png")
            plt.close()
            print(
                f"Saved plot for {metric} as plots/baseline_sentiment_classification/{metric}_plot.png"
            )
        else:
            print(f"Metric '{metric}' not found in training history.")

    evaluation = final_model.evaluate(X_test, y_test_enc, verbose=0)
    print(
        f"Test Loss: {evaluation[0]}, Accuracy: {evaluation[1]}, Recall: {evaluation[2]}, "
        f"Precision: {evaluation[3]}, AUC: {evaluation[4]}, F1-score: {evaluation[5]}"
    )
    predictions = final_model.predict(X_test).argmax(axis=1)
    matrix = confusion_matrix(y_test, predictions)
    print("Confusion Matrix:\n", matrix)
    disp = ConfusionMatrixDisplay(matrix)
    disp.plot()
    plt.savefig("plots/baseline_sentiment_classification/confusion_matrix.png")
    plt.show()

    final_model.save("models/sentiment_classification_baseline_model.keras")
