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
    
    def logistic_regression(self, X_train, y_train, X_val, y_val, epochs=1000, lr=0.01, patience=np.inf):
        # Add bias
        bias_train = np.ones((X_train.shape[0], 1))
        X_train = np.hstack([bias_train, X_train])
        bias_val = np.ones((X_val.shape[0], 1))
        X_val = np.hstack([bias_val, X_val])
        
        self.W = np.random.rand(X_train.shape[1], y_train.shape[1]) / 100
        
        best_loss = np.inf
        patience_counter = 0
        losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training step
            Scores = X_train @ self.W
            probs = self.softmax(Scores)
            error = y_train - probs
            gradient = X_train.T @ error
            self.W = self.W + lr * gradient
            
            # Calculate training loss
            loss = self.cross_entropy_loss(y_train, probs)
            losses.append(loss)
            
            # Validation step
            val_Scores = X_val @ self.W
            val_probs = self.softmax(val_Scores)
            val_loss = self.cross_entropy_loss(y_val, val_probs)
            val_losses.append(val_loss)
            
            # Early stopping check
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter > patience:
                print(f"No improvement in validation loss for {patience} epochs, stopping training.")
                break
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, Training Loss: {loss}, Validation Loss: {val_loss}")
        
        return losses, val_losses

    
    def linear_regression(self, X_train, y_train, X_val, y_val, epochs=1000, lr=0.01, patience=np.inf):
        # Add bias to both training and validation sets
        bias_train = np.ones((X_train.shape[0], 1))
        X_train_bias = np.hstack([bias_train, X_train])
        bias_val = np.ones((X_val.shape[0], 1))
        X_val_bias = np.hstack([bias_val, X_val])
        
        self.W = np.random.rand(X_train_bias.shape[1], y_train.shape[1]) / 100
        
        best_val_MSE = np.inf
        patience_counter = 0
        train_MSEs = []
        val_MSEs = []
        
        for epoch in range(epochs):
            # Training step
            Scores = X_train_bias @ self.W
            residuals = y_train - Scores
            train_MSE = np.mean(residuals**2)
            train_MSEs.append(train_MSE)
            
            # Gradient calculation and weight update
            gradient = -2 * X_train_bias.T @ residuals / X_train_bias.shape[0]
            self.W -= lr * gradient
            
            # Validation step
            val_Scores = X_val_bias @ self.W
            val_residuals = y_val - val_Scores
            val_MSE = np.mean(val_residuals**2)
            val_MSEs.append(val_MSE)
            
            # Early stopping check
            if val_MSE < best_val_MSE:
                best_val_MSE = val_MSE
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter > patience:
                print(f"No improvement in validation MSE for {patience} epochs, stopping training.")
                break
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, Training MSE: {train_MSE}, Validation MSE: {val_MSE}")
        
        return train_MSEs, val_MSEs

    
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
        self.n_output = n_output

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

        if self.n_output == 1:
            A_output = Z_output
        else:
            A_output = self.softmax(Z_output) 

        return A1, A2, A_output
    
    def backpropagation(self, X, y_true, lr):
        # Forward pass
        A1, A2, A_output = self.forward_pass(X)

        # Output layer error (delta)
        error_output = A_output - y_true 
        dW_output = np.dot(A2.T, error_output)
        db_output = np.sum(error_output, axis=0)
        
        # Second hidden layer error (delta)
        error_hidden_2 = np.dot(error_output, self.W_output.T) * (1 - np.power(A2, 2))  # Derivative of tanh is (1 - tanh^2)
        dW_hidden_2 = np.dot(A1.T, error_hidden_2)
        db_hidden_2 = np.sum(error_hidden_2, axis=0)
        
        # First hidden layer error (delta)
        error_hidden_1 = np.dot(error_hidden_2, self.W_hidden_2.T) * (1 - np.power(A1, 2))
        dW_hidden_1 = np.dot(X.T, error_hidden_1)
        db_hidden_1 = np.sum(error_hidden_1, axis=0)
        
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
    
    def train(self, X_train, y_train, X_val, y_val, epochs, lr, patience=np.inf):
        metrics = []
        val_metrics = []
        best_val_metric = np.inf
        patience_counter = 0  # Counts the epochs since the last improvement in validation metric

        for epoch in range(epochs):
            self.backpropagation(X_train, y_train, lr)
        
            _, _, A_output = self.forward_pass(X_train)
            _, _, val_output = self.forward_pass(X_val)

            if self.n_output == 1:
                # Compute MSE for training and validation sets
                train_metric = np.mean((y_train - A_output)**2)
                val_metric = np.mean((y_val - val_output)**2)
            else: 
                # Compute the loss for training and validation sets
                train_metric = self.cross_entropy_loss(y_train, A_output)
                val_metric = self.cross_entropy_loss(y_val, val_output)

            metrics.append(train_metric)
            val_metrics.append(val_metric)

            # Print progress
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, Train MSE: {train_metric}, Val MSE: {val_metric}")

            # Check for improvement
            if val_metric < best_val_metric:
                best_val_metric = val_metric
                patience_counter = 0  # Reset counter if there's an improvement
            else:
                patience_counter += 1  # Increment counter if no improvement

            # Stop training if patience is exceeded
            if patience_counter > patience:
                print(f"No improvement in validation metric for {patience} epochs, stopping training.")
                break

        # Final metric evaluation
        if self.n_output == 1:
            final_metric = train_metric
        else:
            # After the final epoch, calculate the final accuracy for classification
            final_metric = self.calculate_accuracy(y_train, A_output)
            print(f"Final Accuracy: {final_metric}")
            
        return metrics, val_metrics, final_metric
