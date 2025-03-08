import numpy as np
from keras.api.callbacks import EarlyStopping
from keras.api.metrics import AUC, F1Score, Precision, Recall
from keras.api.models import clone_model
from keras.api.optimizers import Adam
from sklearn.model_selection import train_test_split
from typing import Tuple, Any


class DataDistillation:
    def __init__(
        self,
        model: Any,
        labeled_data: Tuple[np.ndarray, np.ndarray],
        unlabeled_data: np.ndarray,
        validation_split: float = 0.2,
        confidence_threshold: float = 0.9,
        random_state: int = None,
        batch_size: int = 32,
        pseudo_batch_size: int = 1000,
        epochs: int = 10,
        should_stratify: bool = False,
        is_multi_class: bool = False,
        metric_to_evaluate_with: str = "accuracy",
    ) -> None:
        """
        Initializes the DataDistillation process.

        Parameters:
            model: Initial teacher model.
            labeled_data: Tuple containing (X, y) for labeled training data.
            unlabeled_data: Unlabeled input data.
            validation_split: Proportion of labeled data to use for validation.
            confidence_threshold: Minimum confidence required for pseudo‑labeling.
            random_state: Random seed for splitting data.
            batch_size: Batch size for training.
            pseudo_batch_size: Batch size for processing unlabeled data.
            epochs: Number of epochs for training each model.
            should_stratify: Whether to stratify the train/validation split.
            is_multi_class: Whether the problem is multi‑class classification.
            metric_to_evaluate_with: Metric used to evaluate performance.
        """
        self.original_model = model
        x_labeled, y_labeled = labeled_data
        self.unlabeled_data = unlabeled_data
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.pseudo_batch_size = pseudo_batch_size
        self.epochs = epochs
        self.confidence_threshold = confidence_threshold
        self.is_multi_class = is_multi_class
        self.metric_to_evaluate_with = metric_to_evaluate_with

        x_train, x_val, y_train, y_val = train_test_split(
            x_labeled,
            y_labeled,
            test_size=validation_split,
            random_state=random_state,
            stratify=y_labeled if should_stratify else None,
        )

        # For binary classification, expand dims if labels are 1D.
        if np.array(y_train).ndim == 1:
            y_train = np.expand_dims(y_train, axis=-1)
            y_val = np.expand_dims(y_val, axis=-1)

        self.labeled_data = (x_train, y_train)
        self.validation_data = (x_val, y_val)
        self.use_one_hot = np.array(y_train).ndim > 1 and np.array(y_train).shape[1] > 1

        self.teacher_model = model

    def _clone_model(self, model: Any) -> Any:
        """
        Clones and compiles the given model with appropriate metrics.
        """
        binary_metrics = [
            "accuracy",
            Recall(),
            Precision(),
            AUC(),
            F1Score(threshold=0.5),
        ]
        multi_class_metrics = [
            "accuracy",
            Recall(),
            Precision(),
            AUC(multi_label=self.is_multi_class),
            F1Score(average="weighted"),
        ]
        cloned = clone_model(model)
        cloned.compile(
            optimizer=Adam(),
            loss=model.loss,
            metrics=multi_class_metrics if self.is_multi_class else binary_metrics,
        )
        return cloned

    def _train_model(self, model: Any) -> None:
        """
        Trains the given model on the labeled training data.
        """
        x_train, y_train = self.labeled_data
        print(f"Training model on labeled training data with shape: {x_train.shape}")
        early_stop = EarlyStopping(
            monitor="loss", patience=5, restore_best_weights=True
        )
        model.fit(
            x_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            callbacks=[early_stop],
        )

    def evaluate_model(self, model, eval_iteration=1):
        print(
            f"Evaluating model on internal validation set with metric {self.metric_to_evaluate_with}..."
        )
        x_val, y_val = self.validation_data
        results = model.evaluate(
            x_val, y_val, batch_size=self.batch_size, verbose=0, return_dict=True
        )

        key_to_find = f"{self.metric_to_evaluate_with}_{eval_iteration}"
        if self.metric_to_evaluate_with == "f1_score":
            key_to_find = "f1_score"

        if self.metric_to_evaluate_with == "accuracy":
            key_to_find = "accuracy"

        # Search keys in a case-insensitive manner.
        for key, value in results.items():
            if key.lower() == key_to_find.lower():
                return value
        raise ValueError(
            f"{self.metric_to_evaluate_with} metric not found in evaluation results: {results}"
        )

    def _generate_pseudo_labels(
        self, predictions: np.ndarray, data_batch: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        num_classes = self.teacher_model.output_shape[-1]
        if predictions.ndim > 1 and predictions.shape[-1] > 1:
            # Multi‑class or one‑hot binary classification.
            max_confidences = np.max(predictions, axis=1)
            mask = max_confidences > self.confidence_threshold
            int_labels = np.argmax(predictions[mask], axis=1)
            pseudo_labels = (
                np.eye(num_classes)[int_labels]
                if self.use_one_hot
                else np.expand_dims(int_labels, axis=-1)
            )
            data_batch = data_batch[mask]
        else:
            # Binary classification with a single probability output.
            preds_flat = predictions.flatten()
            # Identify confident positive and negative predictions.
            pos_mask = preds_flat >= self.confidence_threshold
            neg_mask = preds_flat <= (1 - self.confidence_threshold)
            # Combine masks.
            mask = pos_mask | neg_mask
            if not np.any(mask):
                # No samples meet the confidence criteria.
                return data_batch[:0], np.empty((0, 1))
            # Generate labels: if positive threshold passed, label as 1; else 0.
            confident_preds = preds_flat[mask]
            labels = np.where(confident_preds >= self.confidence_threshold, 1, 0)
            data_batch = data_batch[mask]
            pseudo_labels = (
                np.eye(2)[labels]
                if self.use_one_hot
                else np.expand_dims(labels, axis=-1)
            )
        return data_batch, pseudo_labels

    def start(self) -> None:
        """
        Runs the data distillation process, iteratively refining the teacher model.
        """
        print("Starting Data Distillation")
        # Initial teacher training and evaluation.
        self._train_model(self.teacher_model)
        teacher_metric = self.evaluate_model(self.teacher_model, eval_iteration=1)
        print(f"Teacher performance on validation set: {teacher_metric}")

        accumulated_x = None
        accumulated_y = None
        iteration = 0

        while self.unlabeled_data.shape[0] > 0:
            iteration += 1
            print(f"\nIteration {iteration}")

            # Determine current batch size.
            current_batch_size = min(
                self.pseudo_batch_size, self.unlabeled_data.shape[0]
            )
            batch_data = self.unlabeled_data[:current_batch_size]
            self.unlabeled_data = self.unlabeled_data[current_batch_size:]

            print(f"Teacher predicting on batch with shape: {batch_data.shape}")
            predictions = self.teacher_model.predict(
                batch_data, batch_size=self.batch_size, verbose=0
            )
            predictions = np.array(predictions)

            batch_data, pseudo_labels = self._generate_pseudo_labels(
                predictions, batch_data
            )
            print(f"Pseudo‑labeled samples in this iteration: {pseudo_labels.shape[0]}")

            # Accumulate pseudo‑labeled data.
            if accumulated_x is None:
                accumulated_x = batch_data
                accumulated_y = pseudo_labels
            else:
                accumulated_x = np.concatenate([accumulated_x, batch_data], axis=0)
                accumulated_y = np.concatenate([accumulated_y, pseudo_labels], axis=0)

            # Clone teacher to create a new student model.
            student_model = self._clone_model(self.teacher_model)

            # Combine original labeled data with accumulated pseudo‑labeled data.
            x_train, y_train = self.labeled_data
            x_combined = np.concatenate([x_train, accumulated_x], axis=0)
            y_combined = np.concatenate([y_train, accumulated_y], axis=0)

            print(f"Training student on combined data of shape: {x_combined.shape}")
            early_stop = EarlyStopping(
                monitor="loss", patience=5, restore_best_weights=True
            )
            student_model.fit(
                x_combined,
                y_combined,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=0,
                callbacks=[early_stop],
            )

            student_metric = self.evaluate_model(
                student_model, eval_iteration=iteration + 1
            )
            print(f"Student performance on validation set: {student_metric}")

            # Update teacher if student performance improves.
            if student_metric > teacher_metric:
                print("Student outperformed teacher. Updating teacher model.")
                self.teacher_model = student_model
                teacher_metric = student_metric
            else:
                print("Student did not outperform teacher. Stopping data distillation.")
                break

        self.final_teacher_metric = teacher_metric

        print("Data distillation process completed.")
        print(f"Final teacher performance on validation set: {teacher_metric}")

    def get_final_teacher_metric(self):
        return self.final_teacher_metric

    def get_validation_set(self):
        return self.validation_data

    def get_model(self):
        return self.teacher_model


# for test purposes
if __name__ == "__main__":
    from keras.api import Sequential
    from keras.api.layers import Dense, Flatten
    from keras.api.utils import to_categorical

    def create_model(input_shape):
        model = Sequential(
            [
                Flatten(input_shape=input_shape),
                Dense(128, activation="relu"),
                Dense(3, activation="softmax"),
            ]
        )
        model.compile(
            optimizer=Adam(),
            loss="categorical_crossentropy",
            metrics=[
                "accuracy",
                Recall(),
                Precision(),
                AUC(),
                F1Score(average="weighted"),
            ],
        )
        return model

    # example data
    np.random.seed(5)
    X_labeled = np.random.rand(2000, 28, 28)
    y_labeled = np.random.randint(0, 3, 2000)
    y_labeled = to_categorical(y_labeled, num_classes=3)

    X_unlabeled = np.random.rand(5000, 28, 28)

    base_model = create_model((28, 28))

    dd = DataDistillation(
        model=base_model,
        labeled_data=(X_labeled, y_labeled),
        unlabeled_data=X_unlabeled,
        validation_split=0.2,
        pseudo_batch_size=1000,
        random_state=5,
        metric_to_evaluate_with="f1_score",
        should_stratify=True,
        is_multi_class=True,
    )

    dd.start()
