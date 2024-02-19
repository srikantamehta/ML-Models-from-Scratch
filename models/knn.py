import numpy as np
import pandas as pd
from src.evaluation import Evaluation

class KNN:
    """
    An implementation of the K-Nearest Neighbors algorithm for both classification and regression.
    """
    def __init__(self, config):
        """
        Initializes the KNN model with configuration settings.

        Parameters:
            config (dict): Configuration settings, including the name of the target column.
        """
        self.config = config
        self.vdm = {}
        self.operation_count = 0

    def reset_operation_count(self):
        """
        Resets the operation counter to zero.
        """
        self.operation_count = 0
    
    def get_operation_count(self):
        """
        Returns the current value of the operation counter.
        """
        return self.operation_count

    def is_numeric(self, val):
        """
        Checks if the given individual value is numeric (int or float).
        """
        return isinstance(val, (int, float, np.integer, np.floating))

        
    def compute_vdm(self, X, y, num_classes, p=2):
        """
        Computes the Value Difference Metric (VDM) for all categorical features in the dataset.

        Parameters:
        - X: numpy.ndarray, the dataset features, where rows represent samples and columns represent features.
        - y: numpy.ndarray, the target variable array where each element corresponds to a class label for the respective sample in X.
        - num_classes: int, the total number of unique classes in the target variable.
        - p: int, the power parameter for VDM, typically set to 2 for squared difference.

        This method updates the `vdm` attribute of the class, which is a dictionary where keys are tuples
        of the form (feature_index, value_i, value_j) and values are the computed VDM values.
        """
        n_features = X.shape[1] # Number of features in the dataset
        for feature_index in range(n_features): # Iterate over each feature
            # Check if the feature is categorical (not numeric)
            if not self.is_numeric(X[0, feature_index]):
                # Get unique values for the categorical feature
                feature_values = np.unique(X[:, feature_index])
                # Compare every pair of values for the feature
                for value_i in feature_values:
                    for value_j in feature_values:
                        delta_sum = 0 # Initialize the sum of differences
                        # Compute the VDM for each class
                        for class_val in range(num_classes):
                            # Count occurrences of value_i and value_j for the current class
                            C_i_a = np.sum((X[:, feature_index] == value_i) & (y == class_val))
                            C_i = np.sum(X[:, feature_index] == value_i)
                            C_j_a = np.sum((X[:, feature_index] == value_j) & (y == class_val))
                            C_j = np.sum(X[:, feature_index] == value_j)
                            # Calculate ratios for value_i and value_j
                            ratio_i = C_i_a / C_i if C_i > 0 else 0
                            ratio_j = C_j_a / C_j if C_j > 0 else 0
                            # Add the p-power of absolute difference of ratios to delta_sum
                            delta_sum += abs(ratio_i - ratio_j) ** p
                            self.operation_count += 6  # For the operations in the loop
                        # Assign the computed VDM value to the dictionary
                        self.vdm[(feature_index, value_i, value_j)] = delta_sum ** (1/p)
                        self.operation_count += 1  # For the power operation

    def calc_distance(self, X, Y):
        """
        Calculates the distance between a single sample Y and each sample in a dataset X using a combination
        of Euclidean distance for numeric features and Value Difference Metric (VDM) for categorical features.

        Parameters:
        - X: numpy.ndarray, the dataset features where rows represent samples.
        - Y: numpy.ndarray, a single sample's features.

        Returns:
        - distances: numpy.ndarray, the array of distances from each sample in X to Y.

        This method also updates `operation_count` attribute to keep track of the number of operations performed.
        """
        distances = np.zeros(X.shape[0]) # Initialize distances array
        
        for feature_index in range(X.shape[1]): # Iterate over each feature
            feature_values = X[:, feature_index] # Get the column for the current feature
            # Check if the feature and the corresponding value in Y are numeric
            if self.is_numeric(Y[feature_index]) and all(self.is_numeric(x) for x in feature_values):
                numeric_values = np.array(feature_values, dtype=float)
                y_value = float(Y[feature_index])
                # Compute squared Euclidean distance component
                distances += (numeric_values - y_value) ** 2
                self.operation_count += 3 * X.shape[0]  # Distance calculations
            else:
                # Use VDM for categorical features
                for i, x_val in enumerate(X[:, feature_index]):
                    vdm_key = (feature_index, x_val, Y[feature_index])
                    distances[i] += self.vdm.get(vdm_key, 0)
                    self.operation_count += 1  # For VDM lookup
        return np.sqrt(distances) # Return the square root of summed distances

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

        distances = self.calc_distance(train_set_features, np.array(test_point))

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
        # Exclude the target column from the test set to obtain only the features
        test_set_features = test_set.drop(columns=[self.config['target_column']])
        predictions = []

        # Iterate over each instance in the test set
        for index, row in test_set_features.iterrows():
            # Find the k nearest neighbors of the current test instance from the training set
            neighbors = self.k_nearest_neighbors(row.values, train_set, k)
            classes = [neighbor[1] for neighbor in neighbors]
            # Determine the most common class among the neighbors as the prediction
            predicted_class = max(set(classes), key=classes.count)
            # Append the predicted class to the list of predictions
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
        # Exclude the target column from the test set to focus on the features
        test_set_features = test_set.drop(columns=[self.config['target_column']])
        predictions = []

        # Iterate over each instance in the test set
        for index, row in test_set_features.iterrows():
            neighbors = self.k_nearest_neighbors(row.values, train_set, k)
            # Calculate weights for each neighbor using the Gaussian kernel formula
            weights = [np.exp(-gamma * (dist ** 2)) for dist, _ in neighbors]
            # Sum of all weights for normalization
            total_weight = sum(weights)

            # If the total weight is not zero, compute the weighted average of the neighbors' target values
            if total_weight > 0:
                weighted_sum = sum(weight * target for (dist, target), weight in zip(neighbors, weights))
                prediction = weighted_sum / total_weight
            else:
                # In case of zero total weight, default to the mean of the neighbors' target values
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
  
    def edited_knn_classification(self, train_set, validation_set, k=1):
        """
        Edits the training set by removing instances that are misclassified by their nearest neighbors.
        
        Parameters:
            train_set (DataFrame): The original training dataset, including the target column.
            validation_set (DataFrame): The dataset used for validating the performance during the editing process.
            k (int): The number of nearest neighbors to consider for determining misclassification.

        Returns:
            DataFrame: The edited version of the training set, with potentially noisy instances removed.
        """
        best_loss = 1  # Initialize to one
        best_set = train_set.copy()
        
        while True:
            removal_indices = []
            for index, row in train_set.iterrows():
                features = row.drop(self.config['target_column']).values
                temp_set = train_set.drop(index)
                nearest_neighbors = self.k_nearest_neighbors(features, temp_set, k)
                labels = [label for _, label in nearest_neighbors]
                predicted_class = max(set(labels), key=labels.count)
                
                if predicted_class != row[self.config['target_column']]:
                    removal_indices.append(index)
            
            if removal_indices:
                print("Editing training set...")
                train_set = train_set.drop(removal_indices)
                predictions = self.knn_classifier(validation_set, train_set, k)['Predicted Class']
                current_loss = Evaluation.zero_one_loss(validation_set[self.config['target_column']], predictions)
                
                if current_loss < best_loss:  # Check if the current loss is lower (better)
                    print(f"Zero one loss improved from {best_loss} to {current_loss}. Conitnuing...")
                    best_loss = current_loss
                    best_set = train_set.copy()
                else:
                    print(f"Zero-one loss degraded from {best_loss} to {current_loss}. Stopping editing. ")
                    # Loss did not improve, stop and revert to best set
                    break
            else:
                # No instances marked for removal, stop
                break

        return best_set
  
    def edited_knn_regression(self, train_set, validation_set, epsilon, gamma, k=1):
        """
        Refines the training set by removing instances that have a significant difference 
        between their target value and the predicted value based on their k nearest neighbors.

        Parameters:
            train_set (DataFrame): The original training dataset, including the target column.
            validation_set (DataFrame): The dataset used for validating the performance during the editing process.
            epsilon (float): Threshold for considering prediction errors as significant.
            k (int): The number of nearest neighbors to consider for determining the predicted value.

        Returns:
            DataFrame: The edited version of the training set, with potentially noisy or outlier instances removed.
        """
        
        best_loss = float('inf')  # Initialize to a large value for comparison
        best_set = train_set.copy()

        while True:
            removal_indices = []
            for index, row in train_set.iterrows():
                
                features = row.drop(self.config['target_column']).values
                temp_set = train_set.drop(index)
                nearest_neighbors = self.k_nearest_neighbors(features, temp_set, k)
                predicted_value = np.mean([target for _, target in nearest_neighbors])  # Use mean for regression
                
                if abs(predicted_value - row[self.config['target_column']]) > epsilon:
                    removal_indices.append(index)

            if removal_indices:
                print("Editing training set...")
                train_set = train_set.drop(removal_indices)
                
                # Validate the edited set performance
                
                validation_targets = validation_set[self.config['target_column']]
                predictions = self.knn_regression(validation_set, train_set, k, gamma)['Predicted Value']
                current_loss = Evaluation.mean_squared_error(validation_targets, predictions)
                
                if current_loss < best_loss:
                    print(f"MSE improved from {best_loss} to {current_loss}. Conitnuing...")
                    best_loss = current_loss
                    best_set = train_set.copy()
                else:
                    print(f"Zero-one loss degraded from {best_loss} to {current_loss}. Stopping editing. ")
                    # Performance degraded, stop and revert to best set
                    break
            else:
                # No instances marked for removal, stop
                break

        return best_set