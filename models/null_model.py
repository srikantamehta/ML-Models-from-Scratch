from statistics import mode, mean

class NullModelClassification:
    def __init__(self, config):
        self.config = config
        
    def naive_classifier(self, data):
        # Get the most common class label
        most_common_label = mode(data[self.config['target_column']])
        # Return this label for each instance in the dataset
        predictions = [most_common_label] * len(data)
        return predictions
    
class NullModelRegression:
    def __init__(self, config):
        self.config = config

    def naive_regression(self, data):
        # Compute the mean of the target column
        mean_value = mean(data[self.config['target_column']])
        # Return this mean value for each instance in the dataset
        predictions = [mean_value] * len(data)
        return predictions


