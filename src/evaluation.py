import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, mean_absolute_error, r2_score
from scipy.stats import pearsonr

class Evaluation:

    @staticmethod
    def zero_one_loss(y_true, y_pred):
        """
        Calculate the 0/1 loss for predictions.
        
        :param y_true: The ground truth labels.
        :param y_pred: The predicted labels.
        :return: The 0/1 loss.
        """
        # Calculate the number of misclassifications
        misclassifications = sum(1 for true, pred in zip(y_true, y_pred) if true != pred)
        # Calculate the 0/1 loss
        zero_one_loss = misclassifications / len(y_true)

        return zero_one_loss

    @staticmethod
    def mean_squared_error(y_true, y_pred):
        """
        Calculate the mean squared error for predictions using NumPy for efficiency.
        
        :param y_true: The ground truth labels or values (as a NumPy array or compatible format).
        :param y_pred: The predicted labels or values (as a NumPy array or compatible format).
        :return: The mean squared error.
        """
        mse = np.mean((y_true - y_pred) ** 2)

        return mse
    
    @staticmethod
    def precision(y_true, y_pred):
        """
        Calculate the weighted average precision for multi-class classification using scikit-learn.
        
        :param y_true: The ground truth labels.
        :param y_pred: The predicted labels.
        :return: The weighted average precision.
        """
        return precision_score(y_true, y_pred, average='weighted',zero_division=0)

    @staticmethod
    def recall(y_true, y_pred):
        """
        Calculate the weighted average recall for multi-class classification using scikit-learn.
        
        :param y_true: The ground truth labels.
        :param y_pred: The predicted labels.
        :return: The weighted average recall.
        """
        return recall_score(y_true, y_pred, average='weighted')
    
    @staticmethod
    def f1_score(y_true, y_pred):
        """
        Calculate the weighted average F1 score for multi-class classification using scikit-learn.
        
        :param y_true: The ground truth labels.
        :param y_pred: The predicted labels.
        :return: The weighted average F1 score.
        """
        return f1_score(y_true, y_pred, average='weighted')
    
    @staticmethod
    def mean_absolute_error(y_true, y_pred):
        """
        Calculate the mean absolute error for predictions.
        
        :param y_true: The ground truth labels or values.
        :param y_pred: The predicted labels or values.
        :return: The mean absolute error.
        """
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def r2_coefficient(y_true, y_pred):
        """
        Calculate the R-squared (coefficient of determination) for predictions.
        
        :param y_true: The ground truth values.
        :param y_pred: The predicted values.
        :return: The R-squared value.
        """
        return r2_score(y_true, y_pred)

    @staticmethod
    def pearsons_correlation(y_true, y_pred):
        """
        Calculate Pearson’s correlation coefficient between the true and predicted values.
        
        :param y_true: The ground truth values.
        :param y_pred: The predicted values.
        :return: Pearson’s correlation coefficient.
        """
        correlation, _ = pearsonr(y_true, y_pred)
        return correlation