import pandas as pd
import numpy as np

class KNN:

    def __init__(self, config, test_set, train_set):
        self.config = config
        self.test_set = test_set
        self.train_set = train_set

    def calc_euclidian_distance(self, x1, x2):
        distance = 0
        for i in range(len(x1)):
            distance += (x1[i]-x2[i]) ** 2
        return distance ** 0.5

    def k_nearest_neighbors(self, test_point, k):
        
        train = self.train_set.drop(self.config['target_column'], axis=1)

        distances = []
        for index in range(len(train)):
            distances.append(self.calc_euclidian_distance(test_point, train.iloc[index]))

        sorted_distances = sorted(distances)
        k_nearest_neighbors = sorted_distances[:k]
        
        return k_nearest_neighbors

    def knn_classifier(self, test_set, k):
        
        test_set_drop_class = test_set.drop(self.config['target_column'], axis=1)

        for index in range(len(test_set_drop_class)):
            k_nearest_neighbors = self.k_nearest_neighbors(test_set_drop_class.iloc[index], k)            

        pass

    def knn_regression(self):
        pass
