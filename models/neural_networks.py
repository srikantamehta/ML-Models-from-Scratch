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

        # Add bias
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
    
class FeedForwardNetwork:

    def __init__(self, config, n_input, n_hidden_1, n_hidden_2, n_output) -> None:
        self.config = config
        self.W_hidden_1 = np.random.rand(n_input,n_hidden_1)/100
        self.W_hidden_2 = np.random.rand(n_hidden_1,n_hidden_2)/100
        self.W_output = np.random.rand(n_hidden_2,n_output)/100

        # Bias vectors initialization
        self.b_hidden_1 = np.zeros((1, n_hidden_1))
        self.b_hidden_2 = np.zeros((1, n_hidden_2))
        self.b_output = np.zeros((1, n_output))

    def forward_pass(self, X):
        # First hidden layer
        Z1 = np.dot(X, self.W_hidden_1) + self.b_hidden_1
        A1 = np.tanh(Z1)  # Activation function

        # Second hidden layer
        Z2 = np.dot(A1, self.W_hidden_2) + self.b_hidden_2
        A2 = np.tanh(Z2)  # Activation function

        # Output layer
        Z_output = np.dot(A2, self.W_output) + self.b_output
        A_output = self.softmax(Z_output) 

        return A1, A2, A_output
    
    def backpropagation(self, X, y_true, lr):
        # Forward pass
        A1, A2, A_output = self.forward_pass(X)

        # Output layer error (delta)
        error_output = A_output - y_true  # For cross-entropy loss
        dW_output = np.dot(A2.T, error_output)
        db_output = np.sum(error_output, axis=0, keepdims=True)
        
        # Second hidden layer error (delta)
        error_hidden_2 = np.dot(error_output, self.W_output.T) * (1 - np.power(A2, 2))  # Derivative of tanh is (1 - tanh^2)
        dW_hidden_2 = np.dot(A1.T, error_hidden_2)
        db_hidden_2 = np.sum(error_hidden_2, axis=0, keepdims=True)
        
        # First hidden layer error (delta)
        error_hidden_1 = np.dot(error_hidden_2, self.W_hidden_2.T) * (1 - np.power(A1, 2))
        dW_hidden_1 = np.dot(X.T, error_hidden_1)
        db_hidden_1 = np.sum(error_hidden_1, axis=0, keepdims=True)
        
        # Update weights and biases
        self.W_output -= lr * dW_output
        self.b_output -= lr * db_output
        self.W_hidden_2 -= lr * dW_hidden_2
        self.b_hidden_2 -= lr * db_hidden_2
        self.W_hidden_1 -= lr * dW_hidden_1
        self.b_hidden_1 -= lr * db_hidden_1

    @staticmethod
    def cross_entropy_loss(y, probs):
        # Compute the cross-entropy loss
        N = y.shape[0]
        cross_entropy_loss = -np.sum(y * np.log(probs)) / N
        return cross_entropy_loss
    
    @staticmethod
    def softmax(z):
        exp_scores = np.exp(z)
        probs = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
        return probs
    
    @staticmethod
    def calculate_accuracy(y_true, y_pred):
        # Assuming y_true is one-hot encoded
        true_labels = np.argmax(y_true, axis=1)
        predicted_labels = np.argmax(y_pred, axis=1)
        accuracy = np.mean(predicted_labels == true_labels)
        return accuracy

    def train(self, X_train, y_train, epochs, lr):
        losses = []
        
        for epoch in range(epochs):
            self.backpropagation(X_train, y_train, lr)
        
            _, _, A_output = self.forward_pass(X_train)
            
            # Compute the loss
            loss = self.cross_entropy_loss(y_train, A_output)
            losses.append(loss)
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")
        
        # After the final epoch, calculate the final accuracy
        final_accuracy = self.calculate_accuracy(y_train, A_output)
        print(f"Final Accuracy: {final_accuracy}")
        
        return losses, final_accuracy
