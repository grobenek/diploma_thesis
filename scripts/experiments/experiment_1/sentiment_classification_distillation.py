import numpy as np
import pandas as pd
from keras.api import layers, models
from keras.api.metrics import AUC, F1Score, Precision, Recall
from keras.api.optimizers import Adam
from keras.api.utils import to_categorical

from scripts.experiments.experiment_1.DataDistillation import DataDistillation

df_original = pd.read_csv("data/processed/processed_tweets.csv")
df_original = df_original.dropna(subset=["Sentiment"])
df_original["Sentiment"] = df_original["Sentiment"].map({-1: 0, 0: 1, 1: 2})
padded_vectors = np.load("data/processed/padded_glove_vectors.npy")[:800]
unlabeled_padded_vectors = np.load("data/processed/padded_glove_vectors.npy")[801:]


# model with optimized hyperparameters from baseline model
def create_model():
    model = models.Sequential(
        [
            layers.Input(shape=(26, 300), name="Input"),
            layers.Flatten(),
            layers.Dense(250, activation="relu", name="layer_1"),
            layers.Dropout(0.2, name="dropout_3"),
            layers.BatchNormalization(name="batch_normalization_3"),
            layers.Dense(100, activation="relu", name="layer_2"),
            layers.Dropout(0.2, name="dropout_4"),
            layers.BatchNormalization(name="batch_normalization_4"),
            layers.Dense(3, activation="softmax", name="sentiment"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=0.001),
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


overall_averages = []

for main_run in range(10):
    print(f"\n=== MAIN EXPERIMENT RUN {main_run + 1}/10 ===\n")
    f1_scores = []

    for experiment_run in range(10):
        print(f"Running inner experiment {experiment_run + 1}/10")

        model = create_model()

        distillation = DataDistillation(
            model,
            (padded_vectors, to_categorical(df_original["Sentiment"], 3)),
            unlabeled_padded_vectors,
            epochs=10,
            pseudo_batch_size=1000,
            validation_split=0.2,
            should_stratify=True,
            is_multi_class=True,
            metric_to_evaluate_with="f1_score",
        )

        distillation.start()

        final_metric = distillation.get_final_teacher_metric()
        print(f"Inner experiment {experiment_run + 1} F1 Score: {final_metric}")
        f1_scores.append(final_metric)

    average_f1_score = np.mean(f1_scores)
    print(f"Average F1 Score for main run {main_run + 1}: {average_f1_score}")
    overall_averages.append(average_f1_score)

final_overall_average = np.mean(overall_averages)
print(f"\nFinal overall average F1 Score across 10 main runs: {final_overall_average}")

# save metrics
with open(
    "results/experiments/experiment_1/sentiment_analysis/results.txt", "w"
) as file:
    for idx, avg in enumerate(overall_averages, start=1):
        file.write(f"Average F1 Score (Main Run {idx}): {avg}\n")
    file.write(f"\nFinal overall average F1 Score: {final_overall_average}\n")
