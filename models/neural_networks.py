import numpy as np

class LinearNetwork:

    def __init__(self, config) -> None:
        self.config = config
        self.W = None

    @staticmethod
    def softmax(z):
        exp_scores = np.exp(z)
        probs = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
        return probs
    
    @staticmethod
    def cross_entropy_loss(y, probs):
        # Compute the cross-entropy loss
        N = y.shape[0]
        cross_entropy_loss = -np.sum(y * np.log(probs)) / N
        return cross_entropy_loss
    
    def logistic_regression(self, X, y, epochs=1000, lr=0.01):
       
        # Add weights
        bias = np.ones((X.shape[0],1))
        X = np.hstack([bias,X])
        self.W = np.random.rand(X.shape[1],y.shape[1])/100

        losses = []

        for epochi in range(epochs):

            Scores = X @ self.W
            probs = self.softmax(Scores)
            error = y - probs
            gradient = X.T@error
            self.W = self.W+lr*gradient
            loss = self.cross_entropy_loss(y, probs)
            print(loss)
            losses.append(loss)

        return losses
    
    def linear_regression(self, X, y, epochs=1000, lr=0.01):

        # Add weights
        bias = np.ones((X.shape[0],1))
        X = np.hstack([bias,X])
        self.W = np.random.rand(X.shape[1],y.shape[1])/100

        losses = []

        for epochi in range(epochs):
            Scores = X @ self.W
            residuals =  y - Scores
            loss = np.mean(residuals**2)
            gradient = -2*X.T @ residuals / X.shape[0]
            self.W = self.W - lr*gradient
            print(loss)
            losses.append(loss)

        return losses
    
    def predict_logistic(self, X):
        # Add weights
        bias = np.ones((X.shape[0],1))
        X = np.hstack([bias,X])

        scores = X @ self.W
        probs = self.softmax(scores)
        predictions = np.argmax(probs,axis=1)

        return predictions

    def predict_linear(self, X):
        # Add weights
        bias = np.ones((X.shape[0],1))
        X = np.hstack([bias,X])

        predictions = X @ self.W
        
        return predictions