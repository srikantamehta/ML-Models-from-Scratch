from sklearn.model_selection import train_test_split, StratifiedKFold
import os

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
    
    def k_fold_cross_validation(self, data, k=10, random_state=None):
        """
        Generate stratified k-fold cross-validation datasets.

        :param data: The DataFrame to be used for cross-validation.
        :param k: The number of folds.
        :param random_state: The seed used by the random number generator (for reproducibility).
        :return: A generator yielding train/test datasets for each fold.
        """
        data = data.copy()
        X = data.drop(columns=[self.config['target_column']])
        y = data[self.config['target_column']]

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
        for train_index, test_index in skf.split(X, y):
            yield data.iloc[train_index], data.iloc[test_index]

    def k_2_cross_validation(self, data, k=5, random_state=None):
        """
        Perform 5x2 cross-validation.

        :param data: The DataFrame to be used for cross-validation.
        :param k: The number of times to perform 2-fold cross-validation (usually 5 for 5x2 cross-validation).
        :param random_state: The seed used by the random number generator (for reproducibility).
        :return: A generator yielding train/test datasets for each of the 2-fold cross-validations, repeated k times.
        """
        for i in range(k):
            # Randomly partition the data into 50% each for train and test, then swap them in the second iteration
            train_data, test_data = train_test_split(data, test_size=0.5, stratify=data[self.config['target_column']], random_state=random_state)
            yield train_data, test_data
            yield test_data, train_data  # Swap train and test for the second iteration of each 2-fold cross-validation

