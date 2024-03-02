import math
import pandas as pd
import numpy as np 
from src.evaluation import Evaluation

class DecisionTreeNode:

    def __init__(self, feature=None, threshold=None, value=None, is_leaf=False, children=None):
        self.feature = feature  # Feature on which to split 
        self.threshold = threshold  # Threshold for the split if the feature is numerical
        self.value = value  # Prediction value (for leaf nodes)
        self.is_leaf = is_leaf  # Boolean flag indicating if the node is a leaf node
        self.children = children if children is not None else {}  # Children nodes

    def add_child(self, key, node):
        
        self.children[key] = node

    def set_leaf_value(self, value):
      
        self.is_leaf = True
        self.value = value

class DecisionTree:

    def __init__(self, config, full_dataset=None) -> None:
        
        self.root = None
        self.config = config
        self.all_categories = self.calculate_all_categories(full_dataset) if full_dataset is not None else {}

    def calculate_all_categories(self, full_dataset):
        """
        Calculate all possible categories for each non-numerical column in the dataset,
        using config['numerical_features'] to determine which columns are considered numerical.
        """
        all_categories = {}
        # Determine categorical columns by excluding those listed as numerical in config
        categorical_columns = [col for col in full_dataset.columns if col not in self.config['numerical_features'] and col != self.config['target_column']]
        
        for column in categorical_columns:
            all_categories[column] = set(full_dataset[column].unique())
        return all_categories
    
    def calc_entropy(self, labels):

        labels_counts = labels.value_counts(normalize=True)
        entropy = 0
        for freq in labels_counts:
            if freq > 0:
                entropy -= freq*math.log2(freq)
        return entropy
        
    def calc_gain_ratio(self, labels, features):
        dataset_entropy = self.calc_entropy(labels)
        gain_details = {}  # This will store both gain ratios and thresholds for numerical features

        for feature in features.columns:
            if feature in self.config['numeric_features']:
                # Handle as numerical feature
                sorted_values = features[feature].sort_values().unique()
                thresholds = (sorted_values[:-1] + sorted_values[1:]) / 2  # Potential thresholds

                best_gain_ratio = -np.inf
                best_threshold = None
                for threshold in thresholds:
                    left_subset = labels[features[feature] <= threshold]
                    right_subset = labels[features[feature] > threshold]

                    expected_entropy = sum([
                        self.calc_entropy(subset) * len(subset) / len(labels) 
                        for subset in [left_subset, right_subset]
                    ])
                    
                    intrinsic_value = sum([
                        -len(subset) / len(labels) * math.log2(len(subset) / len(labels)) 
                        if len(subset) > 0 else 0
                        for subset in [left_subset, right_subset]
                    ])

                    info_gain = dataset_entropy - expected_entropy
                    gain_ratio = info_gain / intrinsic_value if intrinsic_value > 0 else 0

                    if gain_ratio > best_gain_ratio:
                        best_gain_ratio = gain_ratio
                        best_threshold = threshold

                gain_details[feature] = {'gain_ratio': best_gain_ratio, 'threshold': best_threshold}
            else:
                # Handle as categorical feature
                unique_values = features[feature].unique()
                expected_entropy, intrinsic_value = 0, 0
                for value in unique_values:
                    subset = labels[features[feature] == value]
                    subset_proportion = len(subset) / len(labels)
                    subset_entropy = self.calc_entropy(subset)
                    
                    expected_entropy += subset_proportion * subset_entropy
                    if subset_proportion > 0:
                        intrinsic_value -= subset_proportion * math.log2(subset_proportion)

                info_gain = dataset_entropy - expected_entropy
                gain_ratio = info_gain / intrinsic_value if intrinsic_value > 0 else 0

                gain_details[feature] = {'gain_ratio': gain_ratio, 'threshold': None}  # No threshold for categorical features

        return gain_details
    

    def calculate_mse(self, data, target, feature):
        sorted_values = data[feature].sort_values().unique()
        thresholds = (sorted_values[:-1] + sorted_values[1:]) / 2

        best_mse = np.inf
        best_threshold = None
        for threshold in thresholds:
            left_target = target[data[feature] <= threshold]
            right_target = target[data[feature] > threshold]

            left_mse = np.mean((left_target - left_target.mean())**2) if len(left_target) > 0 else 0
            right_mse = np.mean((right_target - right_target.mean())**2) if len(right_target) > 0 else 0

            total_mse = (len(left_target) * left_mse + len(right_target) * right_mse) / (len(left_target) + len(right_target))

            if total_mse < best_mse:
                best_mse = total_mse
                best_threshold = threshold

        return best_mse, best_threshold
    
    def select_feature_mse(self, data, target):
        mean_square_errors = {}
        for feature in data.columns:
            if feature in self.config['numeric_features']:
                mse, threshold = self.calculate_mse(data, target, feature)
                mean_square_errors[feature] = {'mse': mse, 'threshold': threshold}

        best_feature = min(mean_square_errors, key=lambda f: mean_square_errors[f]['mse'])
        best_mse = mean_square_errors[best_feature]['mse']
        best_threshold = mean_square_errors[best_feature]['threshold']

        return best_feature, best_mse, best_threshold
    
    def select_feature_gain_ratio(self, features, labels):
        gain_ratios = self.calc_gain_ratio(labels, features)
        
        # Correctly access the 'gain_ratio' value within the dictionaries for comparison
        best_feature = max(gain_ratios, key=lambda f: gain_ratios[f]['gain_ratio'])
        
        # Extract the best gain ratio and threshold for the best feature
        best_gain_ratio = gain_ratios[best_feature]['gain_ratio']
        best_threshold = gain_ratios[best_feature].get('threshold')  # Use .get() to safely handle absence of 'threshold'

        # Return the best feature, its gain ratio, and threshold (if applicable)
        return best_feature, best_gain_ratio, best_threshold

    def split_data(self, data, target, feature, threshold=None):

        subsets = {}

        if threshold is not None:
            # Numerical split
            left_mask = data[feature] <= threshold
            right_mask = ~left_mask

            left_data = data[left_mask]
            left_target = target[left_mask]

            right_data = data[right_mask]
            right_target = target[right_mask]

            subsets['left'] = (left_data, left_target)
            subsets['right'] = (right_data, right_target)
        else:
            # Categorical split
            for category in data[feature].unique():
                category_mask = data[feature] == category
                category_data = data[category_mask]
                category_target = target[category_mask]

                subsets[category] = (category_data, category_target)

        return subsets

    def build_classification_tree(self, data, target, depth=0):
        # Check for a pure split or if there are no features left
        if len(target.unique()) == 1:
            return DecisionTreeNode(value=target.iloc[0], is_leaf=True)
        elif data.empty or len(data.columns) == 0:
            # Return a leaf node with the most common target value
            return DecisionTreeNode(value=target.mode().iloc[0], is_leaf=True)
        
        # Select the best feature and threshold for splitting
        best_feature, best_gain_ratio, best_threshold = self.select_feature_gain_ratio(data, target)
        
        # If no gain is found (data cannot be split), return a leaf node
        if best_gain_ratio <= 0:
            return DecisionTreeNode(value=target.mode().iloc[0], is_leaf=True)
        
        # Split the dataset based on the best feature and threshold
        subsets = self.split_data(data, target, best_feature, best_threshold)
        
        # Create the decision node
        node = DecisionTreeNode(feature=best_feature, threshold=best_threshold)
        
        # Recursively build the tree for each subset
        for subset_key, (subset_data, subset_target) in subsets.items():
            # If splitting was numerical, subset_key will be 'left' or 'right'
            # If splitting was categorical, subset_key will be the category value
            
            # Drop the used feature for categorical split to avoid infinite recursion
            if best_threshold is None:  # Categorical feature
                subset_data = subset_data.drop(columns=[best_feature])
            
            child_node = self.build_classification_tree(subset_data, subset_target, depth + 1)
            node.add_child(subset_key, child_node)
        
        return node
    
    def build_regression_tree(self, data, target, depth=0):
        
        # Check  a pure split or if the dataset is too small
        if len(target.unique()) == 1 or data.empty or len(data.columns) == 0:
            return DecisionTreeNode(value=target.mean(), is_leaf=True)
        
        best_feature, best_mse, best_threshold = self.select_feature_mse(data, target)
       
        # If no valid split was found (indicated by None), or if the improvement is negligible
        if best_feature is None or best_mse <= 0 or best_mse == float('inf'):
            return DecisionTreeNode(value=target.mean(), is_leaf=True)
        
        subsets = self.split_data(data, target, best_feature, best_threshold)

        node = DecisionTreeNode(feature=best_feature, threshold=best_threshold)
        
        for subset_key, (subset_data, subset_target) in subsets.items():
            # Recursively build the tree for each subset
            child_node = self.build_regression_tree(subset_data, subset_target, depth + 1)
            node.add_child(subset_key, child_node)
        
        return node
    
    def prune(self, node, validation_data):
        """
        Prune the tree recursively starting from the given node.
        validation_data is a DataFrame used to evaluate pruning.
        """
        if node.is_leaf:
            return

        # Recursively attempt to prune child nodes first
        for key, child in node.children.items():
            self.prune(child, validation_data)

        # Evaluate error before pruning this node
        before_pruning_error = self.evaluate_error(validation_data)

        # Temporarily make this node a leaf
        original_children = node.children
        node.children = {}
        node.is_leaf = True
        node.value = self.determine_leaf_value(validation_data)

        # Evaluate error after pruning this node
        after_pruning_error = self.evaluate_error(validation_data)

        # Revert pruning if it increases error
        if before_pruning_error <= after_pruning_error:
            node.children = original_children
            node.is_leaf = False
        # Else, keep the node as a leaf and update its value based on validation data

    def evaluate_error(self, data):
        """
        Evaluate the tree's error on the given dataset.
        This should be implemented to handle both regression and classification error metrics.
        """
        predictions = self.predict(data.drop(columns=[self.config['target_column']]))
        true_values = data[self.config['target_column']]

        if self.config['task'] == 'regression':
            return Evaluation().mean_squared_error(true_values,predictions)  # Mean Squared Error
        elif self.config['task'] == 'classification':
            return Evaluation().zero_one_loss(true_values,predictions) # (predictions != true_values).mean()  # Misclassification rate

    def determine_leaf_value(self, data):
        """
        Determine the value of a leaf node based on the given data.
        This should handle both regression and classification tasks.
        """
        if self.config['task'] == 'regression':
            return data[self.config['target_column']].mean()
        elif self.config['task'] == 'classification':
            return data[self.config['target_column']].mode()[0]

    def traverse_tree(self, node, test_instance, depth=0):
        
        if node.is_leaf:
            return node.value

        # Handle numerical features
        if node.threshold is not None:
            if test_instance[node.feature] <= node.threshold:
                return self.traverse_tree(node.children['left'], test_instance, depth + 1)
            else:
                return self.traverse_tree(node.children['right'], test_instance, depth + 1)
        else:  # Handle categorical features
            next_node_key = test_instance[node.feature]
            if next_node_key in node.children:
                return self.traverse_tree(node.children[next_node_key], test_instance, depth + 1)

        return node.value


    def predict(self, test_instances):
        # Check if test_instances is a DataFrame
        if isinstance(test_instances, pd.DataFrame):
            # Use DataFrame.apply() for vectorized row-wise operation
            predictions = test_instances.apply(lambda row: self.traverse_tree(self.root, row), axis=1)
            return predictions
        else:
            raise ValueError("Invalid input format for test_instances. Expected a DataFrame.")
        

    