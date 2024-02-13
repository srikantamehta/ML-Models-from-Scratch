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

    def calc_euclidian_distance(self, X, Y):
        """
        Calculates the Euclidean distance between two sets of points.

        Parameters:
            X (ndarray): An array of points.
            Y (ndarray): A single point.

        Returns:
            ndarray: The Euclidean distances between X and Y.
        """
        X = np.array(X, dtype=float)
        Y = np.array(Y, dtype=float)
        return np.sqrt(np.sum((X - Y) ** 2, axis=1))


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
        train_set_features = train_set.drop(columns=[self.config['target_column']]).values
        train_set_target = train_set[self.config['target_column']].values

        distances = self.calc_euclidian_distance(train_set_features, np.array(test_point))

        nearest_indices = np.argsort(distances)[:k]
        
        return [(distances[i], train_set_target[i]) for i in nearest_indices]

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

    def condensed_knn_classification(self, train_set, k=1):
        """
        Generates a condensed training set from the given training set by iteratively adding instances
        that are misclassified by their nearest neighbor within the condensed set.

        The goal is to reduce the size of the training set while preserving or even improving the classification
        accuracy of the k-NN algorithm.

        Parameters:
            train_set (DataFrame): The original training dataset, including the target column.
            k (int): The number of nearest neighbors to consider, typically k=1 for condensed k-NN.

        Returns:
            DataFrame: The condensed version of the training set.
        """
        change = True
        condensed_set = train_set.sample(n=1)
        
        while change:
            change = False
            for index, row in train_set.drop(condensed_set.index).iterrows():
                nearest_neighbors = self.k_nearest_neighbors(row.drop(self.config['target_column']).values, condensed_set, k)
                nearest_neighbor_label = nearest_neighbors[0][1]
                if nearest_neighbor_label != row[self.config['target_column']]:
                    condensed_set = pd.concat([condensed_set, train_set.loc[[index]]])
                    change = True

        return condensed_set
    
    def condensed_knn_regression(self, train_set, epsilon, k=1):
        """
        Generates a condensed training set from the given training set by iteratively adding instances
        that are misclassified by their nearest neighbor within the condensed set.

        The goal is to reduce the size of the training set while preserving or even improving the classification
        accuracy of the k-NN algorithm.

        Parameters:
            train_set (DataFrame): The original training dataset, including the target column.
            k (int): The number of nearest neighbors to consider, typically k=1 for condensed k-NN.

        Returns:
            DataFrame: The condensed version of the training set.
        """
        change = True
        condensed_set = train_set.sample(n=1)
        
        while change:
            change = False
            for index, row in train_set.drop(condensed_set.index).iterrows():
                nearest_neighbors = self.k_nearest_neighbors(row.drop(self.config['target_column']).values, condensed_set, k)
                nearest_neighbor_label = nearest_neighbors[0][1]
                if abs(nearest_neighbor_label - row[self.config['target_column']]) > epsilon:
                    condensed_set = pd.concat([condensed_set, train_set.loc[[index]]])
                    change = True

        return condensed_set

    def edited_knn_classificaton(self, train_set, k=1):
        """
        Refines the training set by removing instances that are misclassified by their k nearest neighbors.
        This editing process aims to remove noisy instances or those that are near the decision boundary,
        potentially improving the k-NN classifier's accuracy and efficiency.

        Parameters:
            train_set (DataFrame): The original training dataset, including the target column.
            k (int): The number of nearest neighbors to consider for determining misclassification.

        Returns:
            DataFrame: The edited version of the training set, with potentially noisy instances removed.
        """
        isRemoved = True
        edited_set = train_set.copy()
        while isRemoved:
            isRemoved = False
            for index, row in edited_set.iterrows():
                features = row.drop(self.config['target_column']).values
                temp_set = edited_set.drop(index)
                nearest_neighbors = self.k_nearest_neighbors(features, temp_set, k)
                labels = [label for _, label in nearest_neighbors]
                predicted_class = max(set(labels), key=labels.count)
                if predicted_class != row[self.config['target_column']]:
                    edited_set = edited_set.drop(index)
                    isRemoved = True

        return edited_set
    
    def edited_knn_regression(self, train_set, epsilon, k=1):
        """
        Refines the training set by removing instances that are misclassified by their k nearest neighbors.
        This editing process aims to remove noisy instances or those that are near the decision boundary,
        potentially improving the k-NN classifier's accuracy and efficiency.

        Parameters:
            train_set (DataFrame): The original training dataset, including the target column.
            k (int): The number of nearest neighbors to consider for determining misclassification.

        Returns:
            DataFrame: The edited version of the training set, with potentially noisy instances removed.
        """
        isRemoved = True
        edited_set = train_set.copy()
        while isRemoved:
            isRemoved = False
            for index, row in edited_set.iterrows():
                features = row.drop(self.config['target_column']).values
                temp_set = edited_set.drop(index)
                nearest_neighbors = self.k_nearest_neighbors(features, temp_set, k)
                labels = [label for _, label in nearest_neighbors]
                predicted_class = max(set(labels), key=labels.count)
                if abs(predicted_class - row[self.config['target_column']]) > epsilon:
                    edited_set = edited_set.drop(index)
                    isRemoved = True

        return edited_set