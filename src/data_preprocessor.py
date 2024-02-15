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
        has_header = self.config.get('has_header', False)  # Default to False if not specified
        numeric_features = self.config['numeric_features']

        dtype_dict = {col: str for col in column_names if col not in numeric_features + [self.config['target_column']]}

        if has_header:
            data = pd.read_csv(file_path, sep=separator, na_values=missing_value_representation, dtype=dtype_dict)
        else:
            data = pd.read_csv(file_path, sep=separator, names=column_names, header=None, na_values=missing_value_representation, dtype=dtype_dict)
        
        return data
    
    def impute_missing_values(self,data):
        """
        Impute missing values in the DataFrame. Numeric features are filled with the mean of the column.
        Nominal features are filled with the mode (most frequent value) of the column.

        :param data: pandas DataFrame - The DataFrame to process.
        :return: pandas DataFrame - The DataFrame with missing values imputed.
        """      
        
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
    
    def discretize_feature_equal_width(self, data, features, bins):
        """
        Discretize real-valued features into equal-width bins.

        :param data: The DataFrame to process.
        :param features: The list of features to discretize.
        :param bins: The number of bins.
        :return: The DataFrame with the features discretized.
        """
        data = data.copy()
        for feature in features:
            if feature in data.columns:
                data[feature] = pd.cut(data[feature], bins, labels=np.arange(bins), right=False)
        return data

    def discretize_feature_equal_frequency(self, data, features, bins):
        """
        Discretize real-valued features into equal-frequency bins.

        :param data: The DataFrame to process.
        :param features: The list of features to discretize.
        :param bins: The number of bins.
        :return: The DataFrame with the features discretized.
        """
        data = data.copy()
        for feature in features:
            if feature in data.columns:
                try:
                    data[feature] = pd.qcut(data[feature], q=bins, labels=np.arange(bins), duplicates='drop')
                except ValueError as e:
                    print(f"An error occurred during discretization of '{feature}': {e}. This may be due to too many bins for the number of unique values.")
        return data
    
    def standardize_data(self, train_data, data, features=None):
        """
        Apply z-score standardization to specified numeric features of the data based on statistics from the training data.

        :param train_data: The DataFrame representing the training data.
        :param data: The DataFrame to be standardized (can be test data or any other data).
        :param features: The feature(s) to be standardized. Can be a list of feature names or a single feature name.
        :return: The standardized DataFrame.
        """
        train_data = train_data.copy()
        data = data.copy()
        
        if features is None:
            print("No feature(s) specified for standardization.")
            return data
        
        if not isinstance(features, list):
            features = [features]  # Convert a single feature to a list for consistent processing

        for feature in features:
            if feature not in train_data.columns or feature not in data.columns:
                print(f"Feature '{feature}' not found in the training or the data set.")
                continue  # Skip this feature and continue with the next
            
            # Compute mean and std from the training data
            mean = train_data[feature].mean()
            std = train_data[feature].std()

            if std != 0:
                # Standardize the data
                data[feature] = (data[feature] - mean) / std
            else:
                pass

        return data
    
    def log_transform(self, data):

        data = data.copy()
        transform = np.log1p(data[self.config['target_column']])
        data[self.config['target_column']] = transform

        return data
    
    def inverse_log_transform(self,data):

        transform = np.expm1(data)
               
        return transform



