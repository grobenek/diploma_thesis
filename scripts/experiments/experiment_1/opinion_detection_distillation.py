import os

import numpy as np
import pandas as pd
from keras.api import layers, models
from keras.api.metrics import AUC, F1Score, Precision, Recall
from keras.api.optimizers import Adam

from scripts.experiments.experiment_1.DataDistillation import DataDistillation

df_original = pd.read_csv("data/processed/processed_tweets.csv")
df_original = df_original.dropna(subset=["Has opinion"])
df_original["Has opinion"] = df_original["Has opinion"].astype(int)

padded_vectors = np.load("data/processed/padded_glove_vectors.npy")[:800]
unlabeled_padded_vectors = np.load("data/processed/padded_glove_vectors.npy")[800:]


# model with optimized hyperparameters from baseline model
def create_model():
    model = models.Sequential(name="opinion_model")
    model.add(layers.Input(shape=(26, 300), name="Domain_1"))
    model.add(layers.Flatten())
    model.add(layers.Dense(250, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(200, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy", Recall(), Precision(), AUC(), F1Score(threshold=0.5)],
    )
    return model


X_labeled = padded_vectors
y_labeled = df_original["Has opinion"].values[:800]

num_main_runs = 10
num_experiments_per_run = 10

main_run_averages = []

for main_run in range(num_main_runs):
    print(f"\nMAIN RUN {main_run + 1}/{num_main_runs}\n")

    f1_scores = []

    for experiment_run in range(num_experiments_per_run):
        print(f"Running experiment {experiment_run + 1}/{num_experiments_per_run}")

        model = create_model()

        distillation = DataDistillation(
            model=model,
            labeled_data=(X_labeled, y_labeled),
            unlabeled_data=unlabeled_padded_vectors,
            epochs=10,
            pseudo_batch_size=1000,
            validation_split=0.2,
            is_multi_class=False,
            metric_to_evaluate_with="f1_score",
        )

        distillation.start()

        final_metric = distillation.get_final_teacher_metric()
        print(f"Experiment {experiment_run + 1} F1 Score: {final_metric}")
        f1_scores.append(final_metric)

    average_f1_score = np.mean(f1_scores)
    print(f"Average F1 Score for main run {main_run + 1}: {average_f1_score}")
    main_run_averages.append(average_f1_score)

final_overall_average = np.mean(main_run_averages)
print(
    f"\nFinal overall average F1 Score across {num_main_runs} main runs: {final_overall_average}"
)

# Save results
os.makedirs("results/experiments/experiment_1/opinion_detection/", exist_ok=True)
with open(
    "results/experiments/experiment_1/opinion_detection/results.txt", "w"
) as file:
    for idx, avg in enumerate(main_run_averages, start=1):
        file.write(f"Average F1 Score (Main Run {idx}): {avg}\n")
    file.write(f"\nFinal overall average F1 Score: {final_overall_average}\n")
