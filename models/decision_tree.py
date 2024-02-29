import math
import numpy as np 

class DecisionTreeNode:

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, is_leaf=False):
        self.feature = feature  # Feature on which to split
        self.threshold = threshold  # Threshold for the split if the feature is numerical
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.value = value  # Prediction value if the node is a leaf
        self.is_leaf = is_leaf  # Boolean flag indicating if the node is a leaf node
    
    @staticmethod
    def createLeafNode(self, label):
        # Implementation to create a leaf node
        return DecisionTreeNode(value=label, is_leaf=True)

    @classmethod
    def createDecisionNode(cls, feature, threshold=None):
        # Implementation to create a decision node
        return cls(feature=feature, threshold=threshold) 

class DecisionTree:

    def __init__(self, config) -> None:
        
        self.config = config

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
    
    def select_feature_gain_ratio(self, labels, features):
        gain_ratios = self.calc_gain_ratio(labels, features)
        
        # Correctly access the 'gain_ratio' value within the dictionaries for comparison
        best_feature = max(gain_ratios, key=lambda f: gain_ratios[f]['gain_ratio'])
        
        # Extract the best gain ratio and threshold for the best feature
        best_gain_ratio = gain_ratios[best_feature]['gain_ratio']
        best_threshold = gain_ratios[best_feature].get('threshold')  # Use .get() to safely handle absence of 'threshold'
        
        # Return the best feature, its gain ratio, and threshold (if applicable)
        return best_feature, best_gain_ratio, best_threshold

    def build_classification_tree(self, data_train):

        pass
