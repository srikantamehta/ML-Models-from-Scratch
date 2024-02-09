import numpy as np
import pandas as pd

class KNN:
    """
    An optimized implementation of the K-Nearest Neighbors algorithm for both classification and regression.
    """
    def __init__(self, config):
        """
        Initializes the KNN model with configuration settings.

        Parameters:
            config (dict): Configuration settings, including the name of the target column.
        """
        self.config = config

    def calc_euclidian_distance(self, x1, x2):
        """
        Calculates the Euclidean distance between two points.

        Parameters:
            x1 (array-like): The first point.
            x2 (array-like): The second point.

        Returns:
            float: The Euclidean distance between the two points.
        """
        x1 = np.array(x1)
        x2 = np.array(x2)
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def k_nearest_neighbors(self, test_point, train_set, k):
        """
        Finds the k nearest neighbors of a given test point.

        Parameters:
            test_point (array-like): The test point.
            train_set (DataFrame): The training dataset.
            k (int): The number of nearest neighbors to find.

        Returns:
            list of tuples: A list containing the distance to the test point and the target value of each of the k nearest neighbors.
        """
        train_set_features = train_set.drop(columns=[self.config['target_column']])
        train_set_target = train_set[self.config['target_column']]

        # Calculate distances from the test point to all training points
        distances = [self.calc_euclidian_distance(test_point, train_set_features.iloc[index]) for index in range(len(train_set_features))]

        # Get indices of the k smallest distances
        nearest_indices = np.argsort(distances)[:k]

        # Return the k nearest neighbors (distance and target value)
        return [(distances[i], train_set_target.iloc[i]) for i in nearest_indices]

    def knn_classifier(self, test_set, train_set, k):
        """
        Classifies each instance in the test set based on the k nearest neighbors algorithm.

        Parameters:
            test_set (DataFrame): The test dataset.
            train_set (DataFrame): The training dataset.
            k (int): The number of nearest neighbors to use for classification.

        Returns:
            DataFrame: The test set with an additional column for the predicted class.
        """
        test_set_features = test_set.drop(columns=[self.config['target_column']])
        predictions = []

        for index, row in test_set_features.iterrows():
            neighbors = self.k_nearest_neighbors(row.values, train_set, k)
            classes = [neighbor[1] for neighbor in neighbors]
            predicted_class = max(set(classes), key=classes.count)
            predictions.append(predicted_class)

        test_set_with_predictions = test_set.copy()
        test_set_with_predictions['Predicted Class'] = predictions
        return test_set_with_predictions

    def knn_regression(self, test_set, train_set, k, gamma):
        """
        Performs KNN regression on a given test set using a Gaussian kernel for weighting.

        Parameters:
            test_set (DataFrame): The test dataset.
            train_set (DataFrame): The training dataset.
            k (int): The number of nearest neighbors to consider.
            gamma (float): The bandwidth parameter for the Gaussian kernel.

        Returns:
            DataFrame: The test set with an additional column for the predicted value.
        """
        test_set_features = test_set.drop(columns=[self.config['target_column']])
        predictions = []

        for index, row in test_set_features.iterrows():
            neighbors = self.k_nearest_neighbors(row.values, train_set, k)
            weights = [np.exp(-gamma * (dist ** 2)) for dist, _ in neighbors]
            total_weight = sum(weights)

            if total_weight > 0:
                weighted_sum = sum(weight * target for (dist, target), weight in zip(neighbors, weights))
                prediction = weighted_sum / total_weight
            else:
                prediction = np.mean([target for _, target in neighbors])

            predictions.append(prediction)

        test_set_with_predictions = test_set.copy()
        test_set_with_predictions['Predicted Value'] = predictions

        return test_set_with_predictions
