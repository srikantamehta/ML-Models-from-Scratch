import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, config):
        self.config = config
        

    def load_data(self):
        """
        Load data from a file specified in the config.

        :return: Loaded DataFrame.
        """
        file_path = self.config['file_path']
        separator = self.config['separator']
        column_names = self.config['column_names']
        missing_value_representation = self.config['missing_values']

        data = pd.read_csv(file_path, sep=separator, names=column_names, header=None, na_values=missing_value_representation)
       
        return data

    def impute_missing_values(self,data):
        data = data.copy()
        for column in data.columns:
            # Check if the column is numerical or nominal
            if column in self.config['numeric_features']:
                # Impute missing values with the mean for numeric columns
                data[column].fillna(data[column].mean(), inplace=True)
            elif column in self.config['nominal_features']:
                # Impute missing values with the mode for nominal columns
                data[column].fillna(data[column].mode()[0], inplace=True)
        return data

    def encode_ordinal_features(self, data):
        """
        Encode ordinal features as integers based on the predefined order in the config.

        :param data: The DataFrame to process.
        :return: The DataFrame with ordinal features encoded as integers.
        """
        data = data.copy()
        ordinal_features = self.config.get('ordinal_features', {})
        for feature, order in ordinal_features.items():
            if feature in data.columns:
                # Create a mapping from category name to an integer based on the order defined in config
                mapping = {category: rank for rank, category in enumerate(order)}
                # Apply the mapping to the data
                data[feature] = data[feature].map(mapping)

        return data
    
    def encode_nominal_features(self, data):
        """
        Perform one-hot encoding on nominal (categorical) features.

        :param data: The DataFrame to process.
        :return: The DataFrame with nominal features replaced by their one-hot encoded counterparts.
        """
        data = data.copy()
        nominal_features = self.config.get('nominal_features', [])
        for feature in nominal_features:
            if feature in data.columns:
                # Perform one-hot encoding
                dummies = pd.get_dummies(data[feature], prefix=feature, dtype='int')
                # Drop the original nominal feature from the dataset
                data = pd.concat([data.drop(feature, axis=1), dummies], axis=1)
        
        return data
    
    def discretize_feature_equal_width(self, data, feature, bins):
        """
        Discretize a real-valued feature into equal-width bins.

        :param data: The DataFrame to process.
        :param feature: The feature to discretize.
        :param bins: The number of bins.
        :return: The DataFrame with the feature discretized.
        """
        data = data.copy()
        if feature in data.columns:
            data[feature] = pd.cut(data[feature], bins, labels=np.arange(bins), right=False)
        return data

    def discretize_feature_equal_frequency(self, data, feature, bins):
        """
        Discretize a real-valued feature into equal-frequency bins.

        :param data: The DataFrame to process.
        :param feature: The feature to discretize.
        :param bins: The number of bins.
        :return: The DataFrame with the feature discretized.
        """
        data = data.copy()
        if feature in data.columns:
            try:
                data[feature] = pd.qcut(data[feature], q=bins, labels=np.arange(bins), duplicates='drop')
            except ValueError as e:
                print(f"An error occurred during discretization of '{feature}': {e}. This may be due to too many bins for the number of unique values.")
                
        return data