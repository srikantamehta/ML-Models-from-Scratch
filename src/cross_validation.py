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
            
            yield data.iloc[train_index], data.iloc[test_index]
