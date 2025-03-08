from abc import ABC, abstractmethod

import numpy as np


class ExperimentPipeline(ABC):
    @abstractmethod
    def build_model(self, params):
        """
        Build and compile the model using the provided hyperparameters.
        Returns:
            model (tf.keras.Model): Compiled model.
        """
        pass

    @abstractmethod
    def load_data(self):
        """
        Load raw data.
        Returns:
            data: Raw data.
        """
        pass

    @abstractmethod
    def preprocess_data(self, data):
        """
        Preprocess the raw data.
        Args:
            data: Data returned by load_data.
        Returns:
            processed_data: Data ready for training/evaluation.
        """
        pass

    @abstractmethod
    def train_model(
        self, model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32
    ):
        """
        Train the model on training data with a validation set.
        Returns:
            model (tf.keras.Model): Trained model.
        """
        pass

    @abstractmethod
    def save_model(self, model, path):
        """
        Save the trained model to disk.
        """
        pass

    @abstractmethod
    def optimize_hyperparameters(self, data, param_grid):
        """
        Optimize hyperparameters.
        Args:
            data: Processed data.
            param_grid (dict): Grid of hyperparameters.
        Returns:
            best_params (dict): The best hyperparameters found.
        """
        pass

    @abstractmethod
    def cross_validation(self, data, params, n_folds=10, epochs=10, batch_size=32):
        """
        Perform cross-validation using fixed hyperparameters.
        Args:
            data: Processed data.
            params (dict): Hyperparameters to use.
            n_folds (int): Number of CV folds.
        Returns:
            avg_metrics (dict): Average metrics over all folds.
        """
        pass

    def run(
        self,
        n_runs,
        param_grid=None,
        optimize_hyperparameters=True,
        optimal_hyperparameters=None,
    ):
        """
        Run the full pipeline:
            1. Load and preprocess data.
            2. Optimize hyperparameters once.
            3. For each experiment run, perform cross-validation and compute the average metrics.
            4. Compute overall average metrics across runs.

        Args:
            n_runs (int): Number of experiment runs.
            param_grid (dict): Grid of hyperparameters for optimization.
        Returns:
            overall_avg_metrics (dict): Average metrics across all experiment runs.
            best_params (dict): Optimized hyperparameters.
        """
        # Load and preprocess data
        raw_data = self.load_data()
        data = self.preprocess_data(raw_data)

        # Optimize hyperparameters just once
        if optimize_hyperparameters:
            print(f"Optimizing hyperparameters: {param_grid}")
            best_params = self.optimize_hyperparameters(data, param_grid)
            print("Optimized hyperparameters:", best_params)
        else:
            best_params = optimal_hyperparameters if optimal_hyperparameters else {}

        run_metrics = []
        for run in range(1, n_runs + 1):
            print(f"\n--- Starting experiment run {run}/{n_runs} ---")
            metrics = self.cross_validation(data, best_params, n_folds=5)
            run_metrics.append(metrics)
            print(f"Run {run} metrics: {metrics}")

        # Average metrics over runs
        overall_avg_metrics = {}
        for key in run_metrics[0].keys():
            overall_avg_metrics[key] = np.mean([metric[key] for metric in run_metrics])
        print("\nOverall average metrics across runs:", overall_avg_metrics)
        return overall_avg_metrics, best_params
