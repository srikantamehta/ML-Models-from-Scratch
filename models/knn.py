import pandas as pd
import numpy as np

class KNN:

    def __init__(self, data, config):
        self.config = config
        self.data = data

    def calc_euclidian_distance(self, X, Y):
        # Ensure that X and Y are numpy arrays for element-wise operations
        X = np.array(X)
        Y = np.array(Y)
        sum_squared_diff = np.sum((X - Y) ** 2)
        distance = np.sqrt(sum_squared_diff)
        return distance

    def k_nearest_neighbors(self, X, k):
        distances = []

        for index, row in self.data.iterrows():
            # Drop the target column from the row and convert to numpy array
            point = row.drop(labels=[self.config['target_column']]).values
            # Ensure X is in the correct format (numpy array)
            X_array = np.array(X.drop(labels=[self.config['target_column']]).values)
            dist = self.calc_euclidian_distance(X_array, point)
            distances.append((dist, index))

        distances.sort(key=lambda x: x[0])
        nearest_neighbors = distances[:k]

        return [index for _, index in nearest_neighbors]

    def knn_classifier(self, k):
        # Iterate through each row of the DataFrame correctly
        for index, row in self.data.iterrows():
            # Call k_nearest_neighbors with the row for classification
            # Be sure to pass the row as a Series, not as part of a DataFrame
            neighbors = self.k_nearest_neighbors(row, k)
            print(neighbors)  # Or handle the neighbors list as needed

    def knn_regression(self):
        pass
