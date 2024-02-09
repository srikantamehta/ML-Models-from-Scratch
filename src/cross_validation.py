from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, RepeatedKFold

class CrossValidation:

    def __init__(self, config):
        self.config = config

    def random_partition(self, data, val_size=0.2, random_state=None):
        """
        Randomly partition the data into training and testing sets.

        :param data: The DataFrame to be partitioned.
        :param test_size: The proportion of the dataset to include in the test split (default is 0.2).
        :param random_state: The seed used by the random number generator (for reproducibility).
        :return: The training and testing DataFrames.
        """
        data = data.copy()
        train_data, val_data = train_test_split(data, test_size=val_size, random_state=random_state)
        return train_data, val_data
    
    def cross_validation(self, data, n_splits=10, n_repeats=1, random_state=None, stratify=True):
        """
        Perform stratified k-fold cross-validation, supporting repeated splits.

        :param data: The DataFrame to be used for cross-validation.
        :param n_splits: The number of folds (default is 10 for 10-fold cross-validation).
        :param n_repeats: The number of repetitions for cross-validation (default is 1).
                          Set to 5 for 5x2 cross-validation with n_splits=2.
        :param random_state: The seed used by the random number generator (for reproducibility).
        :return: A generator yielding train/test datasets for each fold and repeat.
        """
        data = data.copy()
        X = data.drop(columns=[self.config['target_column']])
        y = data[self.config['target_column']]

        if stratify:
            cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        else:
            cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

        for i, (train_index, test_index) in enumerate(cv.split(X, y if stratify else None)):
            # print(f"Fold {i}:")
            # print(f"  Train indices: {train_index}")
            # print(f"  Test indices: {test_index}")
            yield data.iloc[train_index], data.iloc[test_index]

    def hyperparameter_tuning(self, data, all_hyperparameters, train_model, test_model, calculate_average_performance, pick_best_hyperparameters, val_size=0.2, n_splits=2, n_repeats=5, random_state=42):
        """
        Perform hyperparameter tuning using cross-validation.

        :param data: The DataFrame representing the entire dataset.
        :param all_hyperparameters: A list containing all sets of hyperparameters to be tested.
        :param train_model: Function to train the model on training data.
        :param test_model: Function to test the model on validation data.
        :param calculate_average_performance: Function to calculate the average performance.
        :param pick_best_hyperparameters: Function to pick the best hyperparameters based on results.
        :param val_size: The size of the validation set in the random partition (default=0.2).
        :param n_splits: Number of splits for cross-validation (default=2).
        :param n_repeats: Number of repeats for cross-validation (default=5).
        :param random_state: Random seed for reproducibility (default=42).
        :return: The best hyperparameters and their performance.
        """
        # Step 1: Partition your data into training (80%) and validation (20%)
        data_train, data_val = self.random_partition(data, val_size=val_size, random_state=random_state)

        # Store the results for different hyperparameters
        hyperparameter_results = []

        # Perform cross-validation for each set of hyperparameters
        for hyperparams in all_hyperparameters:
            results = []

            # Step 2: Perform 5x2 cross-validation
            for data_fold_train, data_fold_test in self.cross_validation(data_train, n_splits=n_splits, n_repeats=n_repeats, random_state=random_state):

                # Step 2b: Hyperparameter tuning and model training/testing
                # Train on the train fold, test on the validation set (data_val)
                model = train_model(data_fold_train, hyperparams)
                result = test_model(model, data_val)

                # Collect and store results
                results.append(result)

            # Calculate and store the average performance for these hyperparameters
            average_performance = calculate_average_performance(results)
            hyperparameter_results.append((hyperparams, average_performance))

        # Step 3: Determine the best hyperparameters based on the cross-validation results
        best_hyperparameters, best_performance = pick_best_hyperparameters(hyperparameter_results)

        # Return the best hyperparameters and their performance
        return best_hyperparameters, best_performance