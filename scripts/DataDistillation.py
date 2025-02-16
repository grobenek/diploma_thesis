import numpy as np
from keras.api import Sequential
from keras.api.layers import Dense, Flatten
from keras.api.models import clone_model
from keras.api.optimizers import Adam
from sklearn.model_selection import train_test_split


class DataDistillation:
    def __init__(
            self,
            model,
            labeled_data,
            unlabeled_data,
            validation_split=0.2,
            random_state=1,
            batch_size=32,
            pseudo_batch_size=50,
            epochs=10,
            should_stratify=False
    ):
        self.model = model
        x_labeled, y_labeled = labeled_data
        self.unlabeled_data = unlabeled_data
        self.validation_split = validation_split
        x_train, x_val, y_train, y_val = train_test_split(
            x_labeled,
            y_labeled,
            test_size=self.validation_split,
            random_state=random_state,
            stratify=y_labeled if should_stratify else None
        )
        self.labeled_data = (x_train, y_train)
        self.validation_data = (x_val, y_val)
        self.pseudo_batch_size = pseudo_batch_size
        self.batch_size = batch_size
        self.epochs = epochs

    def _clone_model(self, model):
        cloned_model = clone_model(model)
        cloned_model.compile(optimizer=Adam(), loss=model.loss, metrics=["accuracy"])
        return cloned_model
    
    def get_model(self ) :
     return self.teacher

    def train_model(self, model):
        print("Training model on labeled training data...")
        x_train, y_train = self.labeled_data
        model.fit(
            x_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
        )

    def evaluate_model(self, model):
        print("Evaluating model on internal validation set...")
        x_val, y_val = self.validation_data
        results = model.evaluate(x_val, y_val, batch_size=self.batch_size, verbose=0)
        return results

    def start(self):
        print("Starting Data Distillation")
        self.teacher = self.model
        self.train_model(self.teacher)
        teacher_results = self.evaluate_model(self.teacher)
        teacher_accuracy = (
            teacher_results[1] if isinstance(teacher_results, list) else teacher_results
        )
        print(f"Teacher performance on internal validation set: {teacher_accuracy}")

        accumulated_x = None
        accumulated_y = None
        iteration = 0

        while self.unlabeled_data.shape[0] != 0:
            iteration += 1
            print(f"Starting iteration {iteration}")

            current_batch_size = min(
                self.pseudo_batch_size, self.unlabeled_data.shape[0]
            )
            unlabeled_x_batch = self.unlabeled_data[:current_batch_size]
            self.unlabeled_data = self.unlabeled_data[current_batch_size:]

            # Generate pseudo‑labels using the teacher model.
            print("Teacher predicting next unlabeled data batch with shape:", unlabeled_x_batch.shape)
            preds_on_unlabeled_batch = self.teacher.predict(unlabeled_x_batch, batch_size=self.batch_size, verbose=0)
            preds_on_unlabeled_batch = np.array(preds_on_unlabeled_batch)  # Ensure predictions is a NumPy array.

            confidence_threshold = 0.7  # Adjust as needed
            if preds_on_unlabeled_batch.ndim > 1 and preds_on_unlabeled_batch.shape[-1] > 1:
                max_confidence = np.max(preds_on_unlabeled_batch, axis=1)
                confident_mask = max_confidence > confidence_threshold
                pseudo_labels = np.argmax(preds_on_unlabeled_batch[confident_mask], axis=1)
                unlabeled_x_batch = unlabeled_x_batch[confident_mask]
            else:
                confident_mask = preds_on_unlabeled_batch.flatten() > confidence_threshold
                pseudo_labels = np.ones(np.sum(confident_mask))
                unlabeled_x_batch = unlabeled_x_batch[confident_mask]

            print(f"Pseudo-labeled samples in this iteration: {current_batch_size}")

            # Accumulate pseudo‑labeled data.
            if accumulated_x is None:
                accumulated_x = unlabeled_x_batch
                accumulated_y = pseudo_labels
            else:
                accumulated_x = np.concatenate([accumulated_x, unlabeled_x_batch], axis=0)
                accumulated_y = np.concatenate([accumulated_y, pseudo_labels], axis=0)

            student = self._clone_model(self.teacher)

            # Combine labeled training data with the accumulated pseudo‑labeled data.
            x_train, y_train = self.labeled_data
            x_combined = np.concatenate([x_train, accumulated_x], axis=0)
            y_combined = np.concatenate([np.array(y_train), accumulated_y], axis=0)

            print(f"Training student on combined data with size {x_combined.shape}")
            student.fit(
                x_combined,
                y_combined,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=0,
            )

            student_results = self.evaluate_model(student)
            student_accuracy = (
                student_results[1]
                if isinstance(student_results, list)
                else student_results
            )
            print(f"Student performance on internal validation set: {student_accuracy}")

            if student_accuracy > teacher_accuracy:
                print("Student outperformed teacher. Updating teacher model.")
                self.teacher = student
                teacher_accuracy = student_accuracy
            else:
                print("Student did not outperform teacher. Stopping data distillation.")
                break

        print("Data distillation process completed.")
        print(
            f"Final teacher performance on internal validation set: {teacher_accuracy}"
        )


# for test purposes
if __name__ == "__main__":
    def create_model(input_shape):
        model = Sequential(
            [
                Flatten(input_shape=input_shape),
                Dense(64, activation="relu"),
                Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"]
        )
        return model


    # example data
    np.random.seed(5)
    X_labeled = np.random.rand(2000, 28, 28)
    y_labeled = np.random.randint(0, 2, 2000)

    X_unlabeled = np.random.rand(5000, 28, 28)

    base_model = create_model((28, 28))

    dd = DataDistillation(
        model=base_model,
        labeled_data=(X_labeled, y_labeled),
        unlabeled_data=X_unlabeled,
        validation_split=0.2,
        pseudo_batch_size=1000,
        random_state=5
    )

    dd.start()
