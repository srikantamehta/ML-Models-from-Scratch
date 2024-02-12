import numpy as np

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
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        mse = np.mean((y_true - y_pred) ** 2)

        return mse
        