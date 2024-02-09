class KNN:
    """
    An implementation of the K-Nearest Neighbors algorithm for classification.

    Attributes:
        config (dict): Configuration settings, including the name of the target column.
    """

    def __init__(self, config):
        """
        Initializes the KNN classifier with the given configuration.

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
        distance = sum((x1_val - x2_val) ** 2 for x1_val, x2_val in zip(x1, x2))
        return distance ** 0.5

    def k_nearest_neighbors(self, test_point, train_set, k):
        """
        Finds the k nearest neighbors of a given test point.

        Parameters:
            test_point (array-like): The test point.
            train_set (DataFrame): The training dataset.
            k (int): The number of nearest neighbors to find.

        Returns:
            list of tuples: A list of tuples containing the distance to the test point and the class of each of the k nearest neighbors.
        """
        train_set_features = train_set.drop(columns=[self.config['target_column']])

        distances = [(self.calc_euclidian_distance(test_point, train_set_features.iloc[index]), 
                     train_set.iloc[index][self.config['target_column']])
                     for index in range(len(train_set_features))]
        k_nearest_neighbors = sorted(distances, key=lambda x: x[0])[:k]
        return k_nearest_neighbors

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
            neighbors = self.k_nearest_neighbors(row, train_set, k)
            classes = [neighbor[1] for neighbor in neighbors]
            predicted_class = max(set(classes), key=classes.count)  
            predictions.append(predicted_class)
        
        test_set_with_predictions = test_set.copy()
        test_set_with_predictions['Predicted Class'] = predictions
        return test_set_with_predictions

    def knn_regression(self):
        """
        Placeholder for KNN regression implementation.
        """
        pass
