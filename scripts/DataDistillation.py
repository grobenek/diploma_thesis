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
        )
        self.labeled_data = (x_train, y_train)
        self.validation_data = (x_val, y_val)
        self.pseudo_batch_size = pseudo_batch_size
        self.batch_size = batch_size
        self.epochs = epochs

    def _clone_model(self, model):
        cloned_model = clone_model(model)
        cloned_model.build(model.input_shape)
        cloned_model.set_weights(self.model.get_weights())
        cloned_model.compile(optimizer=Adam(), loss=model.loss, metrics=["accuracy"])
        return cloned_model

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
        teacher = self.model
        self.train_model(teacher)
        teacher_results = self.evaluate_model(teacher)
        teacher_accuracy = (
            teacher_results[1] if isinstance(teacher_results, list) else teacher_results
        )
        print(f"Teacher performance on internal validation set: {teacher_accuracy}")

        accumulated_x = None
        accumulated_y = None
        iteration = 0

        while len(self.unlabeled_data) != 0:
            iteration += 1
            print(f"Starting iteration {iteration}")

            current_batch_size = min(
                self.pseudo_batch_size, self.unlabeled_data.shape[0]
            )
            x_batch = self.unlabeled_data[:current_batch_size]
            self.unlabeled_data = self.unlabeled_data[current_batch_size:]

            # Generate pseudo‑labels using the teacher model.
            preds = teacher.predict(x_batch, batch_size=self.batch_size, verbose=0)
            preds = np.array(preds)  # Ensure predictions is a NumPy array.

            if preds.ndim > 1 and preds.shape[-1] > 1:
                pseudo_labels = np.argmax(preds, axis=1)
            else:
                pseudo_labels = (preds.flatten() > 0.5).astype(np.int32)

            # TODO choose only pseudo-labeled data when greater as treshold
            # confidence_threshold = 0.7  # Adjust as needed
            # if preds.ndim > 1 and preds.shape[-1] > 1:
            #     max_confidence = np.max(preds, axis=1)
            #     pseudo_labels = np.where(max_confidence > confidence_threshold, np.argmax(preds, axis=1), -1)
            # else:
            #     pseudo_labels = np.where(preds.flatten() > confidence_threshold, 1, -1)
            print(f"Pseudo-labeled samples in this iteration: {current_batch_size}")

            # Accumulate pseudo‑labeled data.
            if accumulated_x is None:
                accumulated_x = x_batch
                accumulated_y = pseudo_labels
            else:
                accumulated_x = np.concatenate([accumulated_x, x_batch], axis=0)
                accumulated_y = np.concatenate([accumulated_y, pseudo_labels], axis=0)

            student = self._clone_model(teacher)

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
                teacher = student
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
    np.random.seed(1)
    X_labeled = np.random.rand(2000, 28, 28)
    y_labeled = np.random.randint(0, 2, 2000)

    X_unlabeled = np.random.rand(5000, 28, 28)

    base_model = create_model((28, 28))

    dd = DataDistillation(
        model=base_model,
        labeled_data=(X_labeled, y_labeled),
        unlabeled_data=X_unlabeled,
        validation_split=0.2,
        pseudo_batch_size=1000
    )

    dd.start()
