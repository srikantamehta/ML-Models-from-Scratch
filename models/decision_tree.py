import math
import pandas as pd
import numpy as np 
from src.evaluation import Evaluation

class DecisionTreeNode:
    """
    Represents a node in a decision tree.

    Attributes:
        feature (str): The name of the feature this node splits on, if it's not a leaf node.
        threshold (float): The threshold value for splitting if the feature is numerical. Used for binary splits.
        value: The prediction value assigned to this node if it is a leaf. This could be a class label (in classification)
               or a continuous value (in regression).
        is_leaf (bool): Flag indicating whether this node is a leaf (True) or decision node (False).
        children (dict): A dictionary of children nodes, with keys being the outcome of the split leading to each child.
                         For numerical features, typically uses keys like 'left' and 'right'. For categorical features,
                         the keys correspond to the feature values.
        parent (DecisionTreeNode): Reference to the parent node. `None` for the root node.
    """

    def __init__(self, feature=None, threshold=None, value=None, is_leaf=False, children=None, parent=None):
        """
        Initializes a new instance of the DecisionTreeNode.

        Parameters:
            feature (str): The feature name on which to split. None for leaf nodes.
            threshold (float): The threshold for splitting if the feature is numerical. None for categorical features.
            value: The value to predict if this is a leaf node.
            is_leaf (bool): Indicates if this node is a leaf. Defaults to False.
            children (dict): A dictionary of children nodes. Defaults to an empty dict if None is provided.
            parent (DecisionTreeNode): The parent node. Defaults to None.
        """
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.is_leaf = is_leaf
        self.children = children if children is not None else {}
        self.parent = parent

    def add_child(self, key, node):
        """
        Adds a child node to this node.

        Parameters:
            key: The key associated with the child node. For numerical splits, this could be 'left' or 'right'.
                 For categorical splits, it would be one of the categorical values of the splitting feature.
            node (DecisionTreeNode): The child node to add.
        """
        self.children[key] = node
        node.parent = self  # Sets this node as the parent of the newly added child node.

    def set_leaf_value(self, value):
        """
        Converts this node into a leaf node with the specified prediction value.

        Parameters:
            value: The value to be predicted by this leaf node.
        """
        self.is_leaf = True  # Mark this node as a leaf.
        self.value = value  # Set the prediction value for this leaf node.

class DecisionTree:
    """
    Represents a decision tree for classification or regression.

    Attributes:
        root (DecisionTreeNode): The root node of the decision tree.
        config (dict): Configuration parameters for the decision tree, including 'numeric_features' and 'target_column'.
        numeric_features (list): List of names of numeric features.
        categorical_features (list): List of names of categorical features, derived from the dataset excluding numeric features and the target column.
        all_categories (dict): A dictionary mapping each categorical feature to its unique categories.
    """

    def __init__(self, config, data) -> None:
        """
        Initializes a new instance of the DecisionTree.

        Parameters:
            config (dict): Configuration parameters for the decision tree. It must include 'numeric_features' (list of numeric feature names)
                           and 'target_column' (name of the target variable).
            data (pd.DataFrame): The dataset used for building the decision tree. 
        """
        self.root = None  # Initially, the tree has no nodes
        self.config = config  
        # Identify numeric features based on the config.
        self.numeric_features = config['numeric_features']
        # Determine categorical features by excluding numeric features and the target column from all columns.
        self.categorical_features = [feat for feat in data.columns if feat not in self.numeric_features and feat != config['target_column']]
        # Extract all unique categories for each categorical feature.
        self.all_categories = self.extract_all_categories(data, self.categorical_features)

    def extract_all_categories(self, data, categorical_features):
        """
        Extracts and returns all unique categories for each categorical feature.

        Parameters:
            data (pd.DataFrame): The dataset from which to extract categories.
            categorical_features (list): A list of names of the categorical features.

        Returns:
            dict: A dictionary mapping each categorical feature to a list of its unique categories.
        """
        all_categories = {}
        for feature in categorical_features:
            # For each categorical feature, store its unique values/categories.
            all_categories[feature] = data[feature].unique().tolist()
        return all_categories
    
    def calc_entropy(self, labels):
        """
        Calculates the entropy of a given set of labels, which measures the impurity of the set.

        Parameters:
            labels (pd.Series): A series of class labels.

        Returns:
            float: The calculated entropy value.
        """
        labels_counts = labels.value_counts(normalize=True)  # Get the frequency of each label.
        entropy = 0
        for freq in labels_counts:
            if freq > 0:  
                entropy -= freq * math.log2(freq)  # Calculate entropy using the formula.
        return entropy
        
    def calc_gain_ratio(self, labels, features):
        """
        Calculates the gain ratio for each feature in the dataset, used for deciding the best feature to split on.

        Parameters:
            labels (pd.Series): The target labels for the dataset.
            features (pd.DataFrame): The features of the dataset, from which the best feature to split on will be determined.

        Returns:
            dict: A dictionary with each feature as a key, and its corresponding best gain ratio and (for numerical features) the
                best threshold for splitting as values.
        """
        dataset_entropy = self.calc_entropy(labels)  # Calculate the overall entropy of the dataset.
        gain_details = {}  # Initialize a dictionary to store gain ratios and thresholds for numerical features.

        for feature in features.columns:
            if feature in self.config['numeric_features']:
                # Numerical feature handling.
                sorted_values = features[feature].sort_values().unique()  # Get unique, sorted values of the feature.
                # Calculate potential thresholds as midpoints between consecutive sorted values.
                thresholds = (sorted_values[:-1] + sorted_values[1:]) / 2

                best_gain_ratio = -np.inf
                best_threshold = None
                for threshold in thresholds:
                    # Split the dataset based on the threshold and calculate entropy for each subset.
                    left_subset = labels[features[feature] <= threshold]
                    right_subset = labels[features[feature] > threshold]

                    expected_entropy = sum([
                        self.calc_entropy(subset) * len(subset) / len(labels) 
                        for subset in [left_subset, right_subset]
                    ])
                    
                    # Calculate the intrinsic value.
                    intrinsic_value = sum([
                        -len(subset) / len(labels) * math.log2(len(subset) / len(labels)) 
                        if len(subset) > 0 else 0
                        for subset in [left_subset, right_subset]
                    ])

                    info_gain = dataset_entropy - expected_entropy  # Calculate the information gain from the split.
                    gain_ratio = info_gain / intrinsic_value if intrinsic_value > 0 else 0  # Adjust the gain by the intrinsic value.

                    if gain_ratio > best_gain_ratio:
                        best_gain_ratio = gain_ratio
                        best_threshold = threshold

                gain_details[feature] = {'gain_ratio': best_gain_ratio, 'threshold': best_threshold}
            else:
                # Categorical feature handling.
                unique_values = features[feature].unique()  # Get unique values/categories of the feature.
                expected_entropy, intrinsic_value = 0, 0
                for value in unique_values:
                    subset = labels[features[feature] == value]
                    subset_proportion = len(subset) / len(labels)
                    subset_entropy = self.calc_entropy(subset)
                    
                    expected_entropy += subset_proportion * subset_entropy
                    if subset_proportion > 0:
                        intrinsic_value -= subset_proportion * math.log2(subset_proportion)

                info_gain = dataset_entropy - expected_entropy  # Calculate the information gain from the split.
                gain_ratio = info_gain / intrinsic_value if intrinsic_value > 0 else 0  # Adjust the gain by the intrinsic value.

                gain_details[feature] = {'gain_ratio': gain_ratio, 'threshold': None}  # Note: No threshold for categorical features.

        return gain_details
    

    def calculate_mse(self, data, target, feature):
        """
        Calculates the mean squared error (MSE) for all possible binary splits of a given feature.
        
        This method iterates through all unique values of a specified feature, calculates the potential
        thresholds for splitting, and evaluates the MSE for each split. It identifies the threshold that
        results in the lowest MSE.

        Parameters:
            data (pd.DataFrame): The dataset containing the features.
            target (pd.Series): The target variable corresponding to the data.
            feature (str): The feature on which to calculate the MSE for potential splits.

        Returns:
            float: The lowest MSE achieved by the best binary split.
            float: The threshold value for the best binary split.
        """
        # Calculate possible thresholds as the midpoint between consecutive unique values of the feature
        sorted_values = data[feature].sort_values().unique()
        thresholds = (sorted_values[:-1] + sorted_values[1:]) / 2

        best_mse = np.inf
        best_threshold = None
        for threshold in thresholds:
            # Split the target variable based on the threshold
            left_target = target[data[feature] <= threshold]
            right_target = target[data[feature] > threshold]

            # Calculate MSE for each side of the split
            left_mse = np.mean((left_target - left_target.mean())**2) if len(left_target) > 0 else 0
            right_mse = np.mean((right_target - right_target.mean())**2) if len(right_target) > 0 else 0

            # Calculate the weighted total MSE of the split
            total_mse = (len(left_target) * left_mse + len(right_target) * right_mse) / (len(left_target) + len(right_target))

            # Update best MSE and threshold if this split is better
            if total_mse < best_mse:
                best_mse = total_mse
                best_threshold = threshold

        return best_mse, best_threshold
    
    def select_feature_mse(self, data, target):
        """
        Selects the best feature for splitting based on the mean squared error (MSE).

        Iterates through each numeric feature to calculate the MSE for the best possible binary split.
        The feature that results in the lowest MSE is chosen as the best feature for splitting.

        Parameters:
            data (pd.DataFrame): The dataset containing the features.
            target (pd.Series): The target variable corresponding to the data.

        Returns:
            str: The name of the best feature for splitting.
            float: The lowest MSE achieved among all features.
            float: The threshold value for the best binary split of the selected feature.
        """
        mean_square_errors = {}
        # Iterate through each numeric feature to calculate its best split MSE
        for feature in data.columns:
            if feature in self.config['numeric_features']:
                mse, threshold = self.calculate_mse(data, target, feature)
                mean_square_errors[feature] = {'mse': mse, 'threshold': threshold}

        # Select the feature with the lowest MSE
        best_feature = min(mean_square_errors, key=lambda f: mean_square_errors[f]['mse'])
        best_mse = mean_square_errors[best_feature]['mse']
        best_threshold = mean_square_errors[best_feature]['threshold']

        return best_feature, best_mse, best_threshold
    
    def select_feature_gain_ratio(self, labels, features):
        """
        Selects the feature with the highest gain ratio from a set of features.

        This method computes the gain ratio for each feature in the dataset to determine the best feature
        to split on.

        Parameters:
            features (pd.DataFrame): The features of the dataset, from which the best feature to split on will be determined.
            labels (pd.Series): The target labels for the dataset.

        Returns:
            str: The name of the feature that results in the highest gain ratio.
            float: The highest gain ratio achieved.
            float or None: The threshold value for the best split on the selected feature if it is numerical; None if the feature is categorical.
        """
        # Calculate gain ratios for all features
        gain_ratios = self.calc_gain_ratio(labels, features)
        
        # Find the feature with the maximum gain ratio
        best_feature = max(gain_ratios, key=lambda f: gain_ratios[f]['gain_ratio'])
        
        # Extract the highest gain ratio and the corresponding threshold (if applicable) for the best feature
        best_gain_ratio = gain_ratios[best_feature]['gain_ratio']
        # Use .get() to retrieve the threshold safely, returning None if it's not present (e.g., for categorical features)
        best_threshold = gain_ratios[best_feature].get('threshold')

        # Return the best feature based on gain ratio, the gain ratio itself, and the threshold for splitting
        return best_feature, best_gain_ratio, best_threshold

    def split_data(self, data, target, feature, threshold=None):
        """
        Splits the dataset based on a specified feature and threshold.

        This method divides the dataset into subsets based on the value of a given feature. For numerical features,
        it uses a threshold to create two subsets ('left' and 'right'). For categorical features, it creates a subset
        for each category of the feature.

        Parameters:
            data (pd.DataFrame): The dataset to be split, containing the features.
            target (pd.Series): The target values corresponding to the data.
            feature (str): The feature on which to split the data.
            threshold (float, optional): The threshold value for splitting numerical features. If None, the split
                                        is assumed to be categorical.

        Returns:
            dict: A dictionary of subsets where keys are subset identifiers ('left' and 'right' for numerical splits,
                or category names for categorical splits) and values are tuples of (subset_data, subset_target).
        """
        subsets = {}

        if threshold is not None:
            # Perform a binary split based on the threshold for numerical features.
            left_mask = data[feature] <= threshold
            right_mask = ~left_mask

            left_data = data[left_mask]
            left_target = target[left_mask]

            right_data = data[right_mask]
            right_target = target[right_mask]

            subsets['left'] = (left_data, left_target)
            subsets['right'] = (right_data, right_target)
        else:
            # For categorical features, create a subset for each unique category.
            all_feature_categories = self.all_categories[feature]
            used_categories = data[feature].unique()
            
            for category in all_feature_categories:
                category_mask = data[feature] == category
                category_data = data[category_mask]
                category_target = target[category_mask]
                # Assign each subset of data and target values to the corresponding category key.
                subsets[category] = (category_data, category_target) if category in used_categories else (data.copy(), target.copy())

        return subsets

    def build_classification_tree(self, data, target, depth=0):
        """
        Builds a classification decision tree using a recursive approach.

        This method recursively splits the data based on the feature that provides the highest gain ratio until
        it reaches a condition where no further splitting is beneficial or possible. The recursion stops when
        all data in a node belong to the same class (pure node), there are no features left to split on.

        Parameters:
            data (pd.DataFrame): The dataset used for building the decision tree, containing the features.
            target (pd.Series): The target variable for the classification.
            depth (int): The current depth of the tree. Default is 0 for the root.

        Returns:
            DecisionTreeNode: The root node of the constructed decision tree.
        """
        # Check if all targets are the same (pure split) or if there are no features left to split on
        if len(target.unique()) == 1:
            # All data in this subset belong to the same class, so return a leaf node with that class
            return DecisionTreeNode(value=target.iloc[0], is_leaf=True)
        elif data.empty or len(data.columns) == 0:
            # No features left to split on, return a leaf node with the most common target value
            return DecisionTreeNode(value=target.mode().iloc[0], is_leaf=True)
        
        # Select the best feature for splitting based on the highest gain ratio
        best_feature, best_gain_ratio, best_threshold = self.select_feature_gain_ratio(target, data)
        
        # If no informative gain is found, return a leaf node with the most common target value
        if best_gain_ratio <= 0:
            return DecisionTreeNode(value=target.mode().iloc[0], is_leaf=True)
        
        # Split the dataset based on the best feature and the calculated threshold
        subsets = self.split_data(data, target, best_feature, best_threshold)
        
        # Create a decision node based on the selected feature and threshold
        node = DecisionTreeNode(feature=best_feature, threshold=best_threshold)
        
        # Recursively build the tree for each resulting subset
        for subset_key, (subset_data, subset_target) in subsets.items():
            # For categorical features, remove the split feature from the dataset to prevent infinite recursion
            if best_threshold is None:  # Indicates a categorical feature split
                subset_data = subset_data.drop(columns=[best_feature])
            
            # Recursively build the tree for this subset and add the resulting node as a child
            child_node = self.build_classification_tree(subset_data, subset_target, depth + 1)
            node.add_child(subset_key, child_node)
        
        return node
    
    def build_regression_tree(self, data, target, depth=0):
        """
        Builds a regression decision tree using a recursive approach.

        This method recursively splits the data based on the feature and threshold that minimize the mean squared error (MSE)
        until it reaches a condition where no further splitting is beneficial or possible. The recursion stops when all
        data in a node have the same value, the dataset is too small, there are no features left to split on.

        Parameters:
            data (pd.DataFrame): The dataset used for building the regression tree, containing the features.
            target (pd.Series): The target variable for the regression.
            depth (int): The current depth of the tree. Default is 0 for the root.

        Returns:
            DecisionTreeNode: The root node of the constructed regression decision tree.
        """
        # Check if all target values are the same (pure split), the dataset is empty, or there are no features left
        if len(target.unique()) == 1 or data.empty or len(data.columns) == 0:
            # Return a leaf node with the mean of target values as the prediction value
            return DecisionTreeNode(value=target.mean(), is_leaf=True)
        
        # Select the best feature for splitting based on the lowest MSE
        best_feature, best_mse, best_threshold = self.select_feature_mse(data, target)
        
        # If no valid split improves the MSE, return a leaf node with the mean target value
        if best_feature is None or best_mse <= 0 or best_mse == float('inf'):
            return DecisionTreeNode(value=target.mean(), is_leaf=True)
        
        # Split the dataset based on the best feature and the calculated threshold
        subsets = self.split_data(data, target, best_feature, best_threshold)

        # Create a decision node based on the selected feature and threshold
        node = DecisionTreeNode(feature=best_feature, threshold=best_threshold)
        
        # Recursively build the tree for each resulting subset
        for subset_key, (subset_data, subset_target) in subsets.items():
            # Recursively build the tree for this subset and add the resulting node as a child
            child_node = self.build_regression_tree(subset_data, subset_target, depth + 1)
            node.add_child(subset_key, child_node)
        
        return node

    def prune(self, node, validation_data):
        """
        Prunes the decision tree recursively to improve its generalization capabilities.
        
        Parameters:
            node (DecisionTreeNode): The current node being evaluated for pruning.
            validation_data (pd.DataFrame): The validation dataset used to evaluate the impact of pruning.
                This dataset should not have been used in the training phase.
        
        Returns:
            None: This method modifies the tree in place and does not return any value.
        """
        if node.is_leaf:
            return

        # Filter validation data for each child node
        filtered_data = {}
        if node.threshold is not None:  # Numeric feature
            filtered_data['left'] = validation_data[validation_data[node.feature] <= node.threshold]
            filtered_data['right'] = validation_data[validation_data[node.feature] > node.threshold]
        else:  # Categorical feature
            for value in node.children.keys():
                filtered_data[value] = validation_data[validation_data[node.feature] == value]

        # Recursively prune child nodes first
        for key, child in node.children.items():
            self.prune(child, filtered_data.get(key, pd.DataFrame()))

        # After attempting to prune children, check if this node can be pruned
        # Skip pruning if there's no validation data for this node
        if validation_data.empty:
            return

        # Evaluate error before pruning this node
        before_pruning_error = self.evaluate_error(validation_data)

        # Temporarily make this node a leaf by simulating pruning
        original_state = (node.is_leaf, node.value, node.children)
        node.is_leaf = True
        node.children = {}
        # Use the most common target value in validation data for this node, or skip if no data
        node.value = validation_data[self.config['target_column']].mode()[0] if not validation_data.empty else None

        # Evaluate error after pruning
        after_pruning_error = self.evaluate_error(validation_data)

        # Revert pruning if it does not reduce error
        if after_pruning_error > before_pruning_error:
            node.is_leaf, node.value, node.children = original_state


    def evaluate_error(self, data):
        """
        Evaluate the tree's error on the given dataset.
        This method computes the error of the tree's predictions against the actual target values in the dataset,
        using mean squared error for regression tasks and zero-one loss for classification tasks.

        Parameters:
            data (pd.DataFrame): The dataset containing the features and target column.

        Returns:
            float: The calculated error of the tree. Returns infinity if the dataset is empty, indicating no improvement.
        """
        # Check if the dataset is empty
        if data.empty or len(data[self.config['target_column']]) == 0:
            # Returning infinity to signify that pruning or splitting on an empty dataset does not provide any benefit.
            return float('inf')  

        # Generate predictions for the dataset excluding the target column.
        predictions = self.predict(data.drop(columns=[self.config['target_column']]))
        # Extract the actual target values from the dataset.
        true_values = data[self.config['target_column']]

        # Calculate and return the error based on the task type.
        if self.config['task'] == 'regression':
            # For regression tasks, use mean squared error as the performance metric.
            return Evaluation().mean_squared_error(true_values, predictions)  
        elif self.config['task'] == 'classification':
            # For classification tasks, use zero-one loss (misclassification rate) as the performance metric.
            return Evaluation().zero_one_loss(true_values, predictions)  
    
        
    def determine_leaf_value(self, data):
        """
        Determines the value of a leaf node based on the given data, tailored to the type of task (regression or classification).

        For regression, the mean of the target values in the data is used as the leaf's value. For classification,
        the most common class (mode) among the target values is used.

        Parameters:
            data (pd.DataFrame): The subset of data reaching the leaf, containing the target column.

        Returns:
            The value to be assigned to the leaf node, either a mean (regression) or mode (classification) of the target column.
        """
        # Calculate the leaf node value based on the type of task.
        if self.config['task'] == 'regression':
            # For regression, return the mean of the target column.
            return data[self.config['target_column']].mean()
        elif self.config['task'] == 'classification':
            # For classification, return the most common class (mode) in the target column.
            return data[self.config['target_column']].mode()[0]

    def traverse_tree(self, node, test_instance, depth=0):
        """
        Traverses the decision tree to predict the output for a given test instance.

        This method recursively traverses the tree starting from the given node (typically the root),
        following the path determined by the feature values of the test instance until it reaches a leaf node,
        at which point it returns the value associated with that leaf.

        Parameters:
            node (DecisionTreeNode): The current node in the tree being traversed.
            test_instance (pd.Series): A single instance from the test dataset for which the prediction is being made.
            depth (int): The current depth in the tree. Default is 0 for the root node.

        Returns:
            The prediction value of the leaf node that the test instance falls into.
        """
        # Base case: if the current node is a leaf, return its value as the prediction.
        if node.is_leaf:
            return node.value

        # If the node is not a leaf, determine the path based on the feature's value.
        if node.threshold is not None:
            # For numerical features, compare the feature value to the threshold.
            if test_instance[node.feature] <= node.threshold:
                return self.traverse_tree(node.children['left'], test_instance, depth + 1)
            else:
                return self.traverse_tree(node.children['right'], test_instance, depth + 1)
        else:  
            # For categorical features, follow the path corresponding to the feature's value.
            next_node_key = test_instance[node.feature]
            if next_node_key in node.children:
                return self.traverse_tree(node.children[next_node_key], test_instance, depth + 1)

        # Fallback to the current node's value if the path cannot be followed (e.g., missing feature value).
        return node.value
    
        
    def predict(self, test_instances):
        """
        Predicts the target values for each instance in the test dataset.

        This method applies the decision tree to each instance in the test dataset to make predictions.
        It relies on the `traverse_tree` method to navigate the tree for each instance.

        Parameters:
            test_instances (pd.DataFrame): The test dataset containing instances for prediction.

        Returns:
            pd.Series: A series of predicted values corresponding to each test instance.

        Raises:
            ValueError: If the input format of test_instances is not a pandas DataFrame.
        """
        # Check if test_instances is a DataFrame
        if isinstance(test_instances, pd.DataFrame):
            # Use DataFrame.apply() for vectorized row-wise operation
            predictions = test_instances.apply(lambda row: self.traverse_tree(self.root, row), axis=1)
            return predictions
        else:
            raise ValueError("Invalid input format for test_instances. Expected a DataFrame.")
        
    def traverse_tree_verbose(self, node, test_instance, depth=0, traversal_path=""):
    
        indent = "  " * depth  # Indentation for visualizing the depth level

        if node.is_leaf:
            traversal_path += f"{indent}Reached leaf node with prediction: {node.value}\n"
            return traversal_path

        if node.threshold is not None:
            traversal_path += f"{indent}Decision node at depth {depth}: {node.feature} <= {node.threshold}?\n"
            if test_instance[node.feature] <= node.threshold:
                traversal_path += f"{indent}Yes, proceed to left child...\n"
                return self.traverse_tree_verbose(node.children['left'], test_instance, depth + 1, traversal_path)
            else:
                traversal_path += f"{indent}No, proceed to right child...\n"
                return self.traverse_tree_verbose(node.children['right'], test_instance, depth + 1, traversal_path)
        else:
            traversal_path += f"{indent}Decision node at depth {depth}: {node.feature} == {test_instance[node.feature]}?\n"
            next_node_key = test_instance[node.feature]
            if next_node_key in node.children:
                traversal_path += f"{indent}Proceeding to child node for value: {next_node_key}\n"
                return self.traverse_tree_verbose(node.children[next_node_key], test_instance, depth + 1, traversal_path)
            else:
                traversal_path += f"{indent}Value {next_node_key} not found among children, fallback to current node's value: {node.value}\n"
                return traversal_path
    
    def predict_verbose(self, test_instances):
        
        # Check if test_instances is a DataFrame
        if isinstance(test_instances, pd.DataFrame):
            # Use DataFrame.apply() for vectorized row-wise operation
            predictions = test_instances.apply(lambda row: self.traverse_tree_verbose(self.root, row), axis=1)
            
            # Print the predictions
            for pred in predictions:
                for line in pred.split("\n"):
                    print(line)
                print()  # Add an extra line break between predictions
            
            return predictions
        else:
            raise ValueError("Invalid input format for test_instances. Expected a DataFrame.")
        
    def print_tree(self, node=None, depth=0, prefix="Root"):
        """
        Recursively prints the structure of the decision tree starting from the given node.

        Parameters:
            node (DecisionTreeNode): The current node to print. If None, starts from the root.
            depth (int): The current depth in the tree, used for indentation. Default is 0 for the root.
            prefix (str): The prefix label to show before printing the node's details. Default is "Root".
        """
        if node is None:
            node = self.root

        indent = "  " * depth  # Indentation based on the depth of the node.
        if node.is_leaf:
            print(f"{indent}{prefix} - Leaf, value: {node.value}")
        else:
            if node.threshold is not None:
                # For numerical features
                print(f"{indent}{prefix} - Decision: {node.feature} <= {node.threshold}")
            else:
                # For categorical features (assuming children keys are the category values)
                print(f"{indent}{prefix} - Decision: {node.feature}")

            # Recursively print children nodes
            for key, child in node.children.items():
                child_prefix = f"{node.feature} {key if node.threshold is None else ('<= ' if key == 'left' else '> ')} {node.threshold if node.threshold is not None else ''}"
                self.print_tree(child, depth + 1, prefix=child_prefix)