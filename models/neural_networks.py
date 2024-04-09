import numpy as np

class BaseNetwork:
    """
    Provides common utilities for neural network models including activation functions, 
    loss calculation, and accuracy measurement.
    """
    @staticmethod
    def softmax(z):
        """
        Applies softmax to input array `z`, converting logits to probabilities.

        Parameters:
        - z (np.array): Input logits.

        Returns:
        - np.array: Output probabilities.
        """
        # Prevent overflow/underflow by subtracting the max from each row
        z_max = np.max(z, axis=1, keepdims=True)
        exp_scores = np.exp(z - z_max)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    @staticmethod
    def cross_entropy_loss(y, probs):
        """
        Calculates cross-entropy loss from true labels `y` and predicted probabilities `probs`.

        Parameters:
        - y (np.array): True labels, one-hot encoded.
        - probs (np.array): Predicted probabilities.

        Returns:
        - float: Cross-entropy loss.
        """
        N = y.shape[0]
        # Add a small epsilon to the log to prevent log(0)
        cross_entropy_loss = -np.sum(y * np.log(probs + 1e-9)) / N
        return cross_entropy_loss

    @staticmethod
    def sigmoid(X):
        """
        Applies sigmoid activation function to input array `X`.

        Parameters:
        - X (np.array): Input array.

        Returns:
        - np.array: Transformed array.
        """
        # Clip X values to avoid overflow in exp
        X_clipped = np.clip(X, -50, 50)
        return 1 / (1 + np.exp(-X_clipped))
    
    @staticmethod
    def calculate_accuracy(y_true, y_pred):
        """
        Computes accuracy between true labels `y_true` and predictions `y_pred`.

        Parameters:
        - y_true (np.array): True labels, one-hot encoded.
        - y_pred (np.array): Predictions, logits or probabilities.

        Returns:
        - float: Accuracy score.
        """
        true_labels = np.argmax(y_true, axis=1)
        predicted_labels = np.argmax(y_pred, axis=1)
        accuracy = np.mean(predicted_labels == true_labels)
        return accuracy



class LinearNetwork(BaseNetwork):
    """
    Implements logistic regression and linear regression neural networks,
    extending the BaseNetwork class for linear models.
    
    Attributes:
    - config (dict): Configuration parameters for the network.
    - W (np.array): Weights matrix of the model.
    """

    def __init__(self, config) -> None:
        """
        Initializes the LinearNetwork with a given configuration.

        Parameters:
        - config (dict): Configuration parameters for the network.
        """
        super().__init__()
        self.config = config
        self.W = None
    
    def logistic_regression(self, X_train, y_train, X_val, y_val, epochs=1000, lr=0.01, patience=np.inf):
        """
        Trains the network using logistic regression on the provided training set
        and evaluates it on the validation set.

        Parameters:
        - X_train (np.array): Training data features.
        - y_train (np.array): Training data labels, one-hot encoded.
        - X_val (np.array): Validation data features.
        - y_val (np.array): Validation data labels, one-hot encoded.
        - epochs (int): Number of training epochs (default 1000).
        - lr (float): Learning rate (default 0.01).
        - patience (int): Patience for early stopping (default inf).

        Returns:
        - (list, list): Training and validation losses over epochs.
        """

        # Add a bias term to the training and validation data
        bias_train = np.ones((X_train.shape[0], 1))
        X_train = np.hstack([bias_train, X_train])
        bias_val = np.ones((X_val.shape[0], 1))
        X_val = np.hstack([bias_val, X_val])
        
        # Initialize weights randomly
        self.W = np.random.rand(X_train.shape[1], y_train.shape[1]) / 100
        
        best_loss = np.inf
        patience_counter = 0
        losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Compute model scores for the training data
            Scores = X_train @ self.W
            # Apply softmax to obtain probabilities
            probs = self.softmax(Scores)
            # Compute error as difference between actual labels and probabilities
            error = y_train - probs
            # Calculate gradient with respect to weights
            gradient = X_train.T @ error
            # Update weights using gradient ascent
            self.W = self.W + lr * gradient
            
            # Calculate training loss
            loss = self.cross_entropy_loss(y_train, probs)
            losses.append(loss)
            
            # Validation step: compute validation loss
            val_Scores = X_val @ self.W
            val_probs = self.softmax(val_Scores)
            val_loss = self.cross_entropy_loss(y_val, val_probs)
            val_losses.append(val_loss)
            
            # Early stopping condition
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter > patience:
                # print(f"No improvement in validation loss for {patience} epochs, stopping training.")
                break
            
            # if epoch % 100 == 0:
            #     print(f"Epoch {epoch}/{epochs}, Training Loss: {loss}, Validation Loss: {val_loss}")
        
        return losses, val_losses

    
    def linear_regression(self, X_train, y_train, X_val, y_val, epochs=1000, lr=0.01, patience=np.inf):
        """
        Trains the network using linear regression on the provided training set
        and evaluates it on the validation set.

        Parameters:
        - X_train (np.array): Training data features.
        - y_train (np.array): Training data labels.
        - X_val (np.array): Validation data features.
        - y_val (np.array): Validation data labels.
        - epochs (int): Number of training epochs (default: 1000).
        - lr (float): Learning rate (default: 0.01).
        - patience (int): Patience for early stopping (default: inf).

        Returns:
        - tuple: Tuple containing lists of training and validation mean squared errors (MSE) for each epoch.
        """

        # Add bias term to the training and validation data for intercept
        bias_train = np.ones((X_train.shape[0], 1))
        X_train_bias = np.hstack([bias_train, X_train])
        bias_val = np.ones((X_val.shape[0], 1))
        X_val_bias = np.hstack([bias_val, X_val])
        
        # Initialize weights randomly
        self.W = np.random.rand(X_train_bias.shape[1], y_train.shape[1]) / 100
        
        best_val_MSE = np.inf
        patience_counter = 0
        train_MSEs = []
        val_MSEs = []
        
        for epoch in range(epochs):
            # Compute model scores (predictions) for the training data
            Scores = X_train_bias @ self.W
            # Calculate residuals between actual and predicted values
            residuals = y_train - Scores
            # Calculate training mean squared error
            train_MSE = np.mean(residuals**2)
            train_MSEs.append(train_MSE)
            
            # Update weights using gradient descent
            gradient = -2 * X_train_bias.T @ residuals / X_train_bias.shape[0]
            self.W -= lr * gradient
            
            # Validation: compute MSE for validation set
            val_Scores = X_val_bias @ self.W
            val_residuals = y_val - val_Scores
            val_MSE = np.mean(val_residuals**2)
            val_MSEs.append(val_MSE)
            
            # Early stopping based on validation MSE
            if val_MSE < best_val_MSE:
                best_val_MSE = val_MSE
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter > patience:
                # print(f"No improvement in validation MSE for {patience} epochs, stopping training.")
                break
            
            # if epoch % 100 == 0:
            #     print(f"Epoch {epoch}/{epochs}, Training MSE: {train_MSE}, Validation MSE: {val_MSE}")
        
        return train_MSEs, val_MSEs

    
    def predict_logistic(self, X):
        """
        Makes predictions using the trained logistic regression model.

        Parameters:
        - X (np.array): Data features for prediction.

        Returns:
        - np.array: Predicted class labels for each input.
        """

        # Add bias term to the input data
        bias = np.ones((X.shape[0],1))
        X = np.hstack([bias,X])

        # Compute scores and apply softmax to get probabilities
        scores = X @ self.W
        probs = self.softmax(scores)
        # Predict the class with the highest probability
        predictions = np.argmax(probs,axis=1)

        return predictions

    def predict_linear(self, X):
        """
        Makes predictions using the trained linear regression model.

        Parameters:
        - X (np.array): Data features for prediction.

        Returns:
        - np.array: Predicted values for each input.
        """
        # Add bias term to the input data for intercept
        bias = np.ones((X.shape[0],1))
        X = np.hstack([bias,X])

        # Compute predictions as dot product of weights and features
        predictions = X @ self.W
        
        return predictions
    
class FeedForwardNetwork(BaseNetwork):
    """
    Represents a feedforward neural network with two hidden layers,
    extending the BaseNetwork.

    Attributes:
    - config (dict): Configuration settings for the network.
    - n_output (int): Number of neurons in the output layer.
    - W_hidden_1 (np.array): Weights of the first hidden layer.
    - W_hidden_2 (np.array): Weights of the second hidden layer.
    - W_output (np.array): Weights of the output layer.
    - b_hidden_1 (np.array): Biases of the first hidden layer.
    - b_hidden_2 (np.array): Biases of the second hidden layer.
    - b_output (np.array): Biases of the output layer.
    - best_W_hidden_1, best_W_hidden_2, best_W_output, best_b_hidden_1, best_b_hidden_2, best_b_output:
      Best weights and biases during training for early stopping.
    """

    def __init__(self, config, n_input, n_hidden_1, n_hidden_2, n_output) -> None:
        """
        Initializes the feedforward network with the specified architecture and parameters.

        Parameters:
        - config (dict): Configuration settings for the network.
        - n_input (int): Number of neurons in the input layer.
        - n_hidden_1 (int): Number of neurons in the first hidden layer.
        - n_hidden_2 (int): Number of neurons in the second hidden layer.
        - n_output (int): Number of neurons in the output layer.
        """
        
        super().__init__()
        self.config = config
        self.n_output = n_output

        # Initialize weights randomly for layers
        self.W_hidden_1 = np.random.rand(n_input,n_hidden_1)/100
        self.W_hidden_2 = np.random.rand(n_hidden_1,n_hidden_2)/100
        self.W_output = np.random.rand(n_hidden_2,n_output)/100

        # Initialize biases to zeros for layers
        self.b_hidden_1 = np.zeros((1, n_hidden_1))
        self.b_hidden_2 = np.zeros((1, n_hidden_2))
        self.b_output = np.zeros((1, n_output))

        # Save best weights and biases 
        self.best_W_hidden_1 = np.random.rand(n_input,n_hidden_1)/100
        self.best_W_hidden_2 = np.random.rand(n_hidden_1,n_hidden_2)/100
        self.best_W_output = np.random.rand(n_hidden_2,n_output)/100
        self.best_b_hidden_1 = np.zeros((1, n_hidden_1))
        self.best_b_hidden_2 = np.zeros((1, n_hidden_2))
        self.best_b_output = np.zeros((1, n_output))

    def forward_pass(self, X):
        """
        Performs a forward pass through the network, calculating the output for each layer.

        Parameters:
        - X (np.array): Input data.

        Returns:
        - tuple: Activations from the first and second hidden layers, and the output layer.
        """

        # Calculate activations for the first hidden layer
        Z1 = np.dot(X, self.W_hidden_1) + self.b_hidden_1
        A1 = np.tanh(Z1)  # # Tanh activation function 

        # Calculate activations for the second hidden layer
        Z2 = np.dot(A1, self.W_hidden_2) + self.b_hidden_2
        A2 = np.tanh(Z2)  # Tanh activation function

        # Calculate the output layer's activations
        Z_output = np.dot(A2, self.W_output) + self.b_output

        # Use softmax for multi-class classification, linear for regression/single output
        if self.n_output == 1:
            A_output = Z_output
        else:
            A_output = self.softmax(Z_output) 

        return A1, A2, A_output
    
    def backpropagation(self, X, y_true, lr):
        """
        Performs the backpropagation algorithm for a single training iteration,
        updating the network's weights and biases based on the error gradient.

        Parameters:
        - X (np.array): The input data for the current batch.
        - y_true (np.array): The true labels for the current batch.
        - lr (float): The learning rate.

        Returns:
        - None
        """

        # Forward pass to compute activations
        A1, A2, A_output = self.forward_pass(X)

        # Compute error gradients for each layer starting from the output
        error_output = A_output - y_true 
        dW_output = np.dot(A2.T, error_output)
        db_output = np.sum(error_output, axis=0)
        
        # Compute error for second hidden layer
        error_hidden_2 = np.dot(error_output, self.W_output.T) * (1 - np.power(A2, 2))  # Derivative of tanh is (1 - tanh^2)
        dW_hidden_2 = np.dot(A1.T, error_hidden_2)
        db_hidden_2 = np.sum(error_hidden_2, axis=0)
        
        # Compute error for first hidden layer
        error_hidden_1 = np.dot(error_hidden_2, self.W_hidden_2.T) * (1 - np.power(A1, 2))
        dW_hidden_1 = np.dot(X.T, error_hidden_1)
        db_hidden_1 = np.sum(error_hidden_1, axis=0)
        
        # Update weights and biases for each layer
        self.W_output -= lr * dW_output
        self.b_output -= lr * db_output
        self.W_hidden_2 -= lr * dW_hidden_2
        self.b_hidden_2 -= lr * db_hidden_2
        self.W_hidden_1 -= lr * dW_hidden_1
        self.b_hidden_1 -= lr * db_hidden_1
    
    def train(self, X_train, y_train, X_val, y_val, epochs, lr, patience=np.inf):
        """
        Trains the network using the provided training data, evaluates it on the validation data,
        and implements early stopping based on validation performance.

        Parameters:
        - X_train (np.array): Training data features.
        - y_train (np.array): Training data labels.
        - X_val (np.array): Validation data features.
        - y_val (np.array): Validation data labels.
        - epochs (int): Number of epochs to train.
        - lr (float): Learning rate.
        - patience (int): Patience for early stopping.

        Returns:
        - Tuple of lists: Training and validation performance metrics, and the final metric after training.
        """

        metrics = []
        val_metrics = []
        best_val_metric = np.inf
        patience_counter = 0  

        for epoch in range(epochs):
            # Perform backpropagation to update weights
            self.backpropagation(X_train, y_train, lr)
        
            # Compute metrics for both training and validation sets
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

            # # # Print progress
            # if epoch % 100 == 0:
            #     if self.n_output == 1:
            #         print(f"Epoch {epoch}/{epochs}, Train MSE: {train_metric}, Val MSE: {val_metric}")
            #     else:
            #         print(f"Epoch {epoch}/{epochs}, Train Loss: {train_metric}, Val Loss: {val_metric}")

            # Early stopping check
            if val_metric < best_val_metric:
                best_val_metric = val_metric
                patience_counter = 0  # Reset counter if there's an improvement
                
                # Update best weights and biases
                self.best_W_hidden_1 = self.W_hidden_1.copy()
                self.best_W_hidden_2 = self.W_hidden_2.copy()
                self.best_W_output = self.W_output.copy()
                self.best_b_hidden_1 = self.b_hidden_1.copy()
                self.best_b_hidden_2 = self.b_hidden_2.copy()
                self.best_b_output = self.b_output.copy()

            else:
                patience_counter += 1  # Increment counter if no improvement

            # Stop training if patience is exceeded
            if patience_counter > patience:
                # print(f"No improvement in validation metric for {patience} epochs, stopping training.")
                break

        # Load best weights and biases if early stopping was triggered
        self.W_hidden_1 = self.best_W_hidden_1
        self.W_hidden_2 = self.best_W_hidden_2
        self.W_output = self.best_W_output
        self.b_hidden_1 = self.best_b_hidden_1
        self.b_hidden_2 = self.best_b_hidden_2
        self.b_output = self.best_b_output
         
        # Final metric evaluation
        if self.n_output == 1:
            final_metric = train_metric
        else:
            # After the final epoch, calculate the final accuracy for classification
            final_metric = self.calculate_accuracy(y_train, A_output)
            
        return metrics, val_metrics, final_metric

class AutoEncoder(BaseNetwork):
    """
    Implements an AutoEncoder neural network.

    Attributes:
    - config (dict): Configuration settings for the AutoEncoder.
    - n_encoder (int): Number of neurons in the encoder (and decoder) layer.
    - W_encoder (np.array): Weights of the encoder layer.
    - W_decoder (np.array): Weights of the decoder layer.
    - b_encoder (np.array): Biases of the encoder layer.
    - b_decoder (np.array): Biases of the decoder layer.
    """

    def __init__(self, config, n_input, n_encoder) -> None:
        """
        Initializes the AutoEncoder with specified configurations.

        Parameters:
        - config (dict): Configuration settings for the AutoEncoder.
        - n_input (int): Number of neurons in the input layer.
        - n_encoder (int): Number of neurons in the encoder (and decoder) layer.
        """
        
        super().__init__()
        self.config = config
        self.n_encoder = n_encoder

        # Initialize encoder and decoder weights with small random values
        self.W_encoder = np.random.rand(n_input,n_encoder)/100
        self.W_decoder = np.random.rand(n_encoder,n_input)/100

        # Initialize encoder and decoder biases to zeros
        self.b_encoder = np.zeros((1, n_encoder))
        self.b_decoder = np.zeros((1, n_input))
    
    def forward_pass(self, X):
        """
        Performs a forward pass through the AutoEncoder, returning the encoder's
        activations and the reconstructed output.

        Parameters:
        - X (np.array): Input data.

        Returns:
        - tuple: Encoder activations and the reconstructed output.
        """
        
        # Encoder layer
        Z1 = np.dot(X, self.W_encoder) + self.b_encoder
        A1 = self.sigmoid(Z1)  # Sigmoid activation function

        # Decoder layer
        A_output = np.dot(A1, self.W_decoder) + self.b_decoder
        
        return A1, A_output
    
    def backpropagation(self, X, lr):
        """
        Performs the backpropagation algorithm, updating the AutoEncoder's
        weights and biases based on the reconstruction error.

        Parameters:
        - X (np.array): Original input data.
        - lr (float): Learning rate.

        Returns:
        - None
        """

        # Forward pass
        A1, A_output = self.forward_pass(X)

        # Output layer error (delta)
        error_decoder = A_output - X

        # Compute gradients for the decoder
        dW_decoder = np.dot(A1.T, error_decoder)
        db_decoder = np.sum(error_decoder, axis=0)
        
        # Compute gradients for the encoder
        error_encoder = np.dot(error_decoder, self.W_decoder.T) * A1 * (1 - A1) 
        dW_encoder = np.dot(X.T, error_encoder)
        db_encoder = np.sum(error_encoder, axis=0)
        
        # Update encoder and decoder weights and biases
        self.W_decoder -= lr * dW_decoder
        self.b_decoder -= lr * db_decoder
        self.W_encoder -= lr * dW_encoder
        self.b_encoder -= lr * db_encoder

    def train(self, X_train, max_epochs=1000, lr=0.0001, patience=np.inf):
        """
        Trains the AutoEncoder on the provided training data using backpropagation,
        with early stopping based on the reconstruction loss.

        Parameters:
        - X_train (np.array): The input data for training.
        - max_epochs (int): The maximum number of epochs for training. Default is 1000.
        - lr (float): The learning rate for weight updates. Default is 0.0001.
        - patience (int): The number of epochs to wait for an improvement in loss before stopping early. Default is infinity.

        Returns:
        - list: A list of mean squared error losses for each epoch of training.
        """

        best_loss = np.inf 
        patience_counter = 0
        losses = []

        for epoch in range(max_epochs):

            # Update the model weights using backpropagation
            self.backpropagation(X_train, lr)

            # Forward pass 
            _, A_output = self.forward_pass(X_train)

            # Calculate error
            error = A_output - X_train
            loss = np.mean(error**2)
            losses.append(loss)
            
            # print(f"Epoch {epoch}/{max_epochs}, Loss: {loss}")

            # Check for improvement in loss
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter > patience:
                    break
        return losses

class CombinedModel(BaseNetwork):
    """
    A combined model that utilizes a pre-trained autoencoder for feature extraction and
    adds additional layers for a specific task, such as classification or regression.

    The first part of the model is the encoder part of an autoencoder, which is followed
    by additional trainable layers for the task.

    Attributes:
    - autoencoder (AutoEncoder): A pre-trained AutoEncoder object.
    - n_output (int): Number of neurons in the output layer, typically matching the number
      of classes for classification tasks.
    - W_hidden_1, W_hidden_2, W_output (np.array): Weight matrices for the first hidden (encoder),
      second hidden, and output layers, respectively.
    - b_hidden_1, b_hidden_2, b_output (np.array): Bias vectors for the first hidden (encoder),
      second hidden, and output layers, respectively.
    - best_W_hidden_1, best_W_hidden_2, best_W_output, best_b_hidden_1, best_b_hidden_2, best_b_output:
      Attributes to store the best weights and biases (for implementing early stopping, etc.).
    """

    def __init__(self, autoencoder, n_hidden_2, n_output) -> None:
        """
        Initializes the CombinedModel with the encoder from a pre-trained autoencoder and
        additional layers for further tasks.

        Parameters:
        - autoencoder (AutoEncoder): The pre-trained AutoEncoder whose encoder part is to be used.
        - n_hidden_2 (int): Number of neurons in the second hidden layer.
        - n_output (int): Number of neurons in the output layer.
        """

        super().__init__()
        self.autoencoder = autoencoder
        self.n_output = n_output

        # Copy encoder weights and biases from the pre-trained autoencoder
        self.W_hidden_1 = autoencoder.W_encoder.copy()
        self.b_hidden_1 = autoencoder.b_encoder.copy()

        # Initialize weights and biases for the new layers
        self.W_hidden_2 = np.random.rand(autoencoder.n_encoder, n_hidden_2) / 100
        self.b_hidden_2 = np.zeros((1, n_hidden_2))
        self.W_output = np.random.rand(n_hidden_2, n_output) / 100
        self.b_output = np.zeros((1, n_output))

        # Prepare attributes for storing the best model state
        self.best_W_hidden_1 = self.W_hidden_1.copy()
        self.best_W_hidden_2 = self.W_hidden_2.copy()
        self.best_W_output = self.W_output.copy()
        self.best_b_hidden_1 = self.b_hidden_1.copy()
        self.best_b_hidden_2 = self.b_hidden_2.copy()
        self.best_b_output = self.b_output.copy()
    
    def forward_pass(self, X):
        """
        Performs a forward pass through the combined model, using the encoder from
        the autoencoder and the additional layers.

        Parameters:
        - X (np.array): Input data.

        Returns:
        - tuple: Activations from the encoder, second hidden layer, and the output prediction.
        """

        # Encoder layer from the pre-trained autoencoder
        Z1 = np.dot(X, self.W_hidden_1) + self.b_hidden_1
        A1 = self.sigmoid(Z1)  # Activation function

        # Second hidden layer 
        Z2 = np.dot(A1, self.W_hidden_2) + self.b_hidden_2
        A2 = np.tanh(Z2)  # Tanh Activation function

        # Output layer
        Z_output = np.dot(A2, self.W_output) + self.b_output

        # Use softmax for multi-class classification or linear for regression/single output
        if self.n_output == 1:
            A_output = Z_output
        else:
            A_output = self.softmax(Z_output) 

        return A1, A2, A_output   
    
    def backpropagation(self, X, y_true, lr):
        """
        Executes the backpropagation algorithm for updating the model's weights and biases
        based on the prediction error.

        Parameters:
        - X (np.array): Input data.
        - y_true (np.array): True labels.
        - lr (float): Learning rate.

        The method calculates gradients for all layers (output, second hidden, and first hidden)
        and updates weights and biases accordingly.
        """

        # Perform forward pass to get activations
        A1, A2, A_output = self.forward_pass(X)

        # Compute gradients for the output layer
        error_output = A_output - y_true 
        dW_output = np.dot(A2.T, error_output)
        db_output = np.sum(error_output, axis=0)
        
        # Compute gradients for the second hidden layer
        error_hidden_2 = np.dot(error_output, self.W_output.T) * (1 - np.power(A2, 2))  # Derivative of tanh is (1 - tanh^2)
        dW_hidden_2 = np.dot(A1.T, error_hidden_2)
        db_hidden_2 = np.sum(error_hidden_2, axis=0)
        
        # Compute gradients for the first hidden layer (encoder layer)
        error_hidden_1 = np.dot(error_hidden_2, self.W_hidden_2.T) * A1 * (1 - A1) 
        dW_hidden_1 = np.dot(X.T, error_hidden_1)
        db_hidden_1 = np.sum(error_hidden_1, axis=0)
        
        # Update the model's weights and biases based on gradients
        self.W_output -= lr * dW_output
        self.b_output -= lr * db_output
        self.W_hidden_2 -= lr * dW_hidden_2
        self.b_hidden_2 -= lr * db_hidden_2
        self.W_hidden_1 -= lr * dW_hidden_1
        self.b_hidden_1 -= lr * db_hidden_1

    def train(self, X_train, y_train, X_val, y_val, epochs, lr, patience=np.inf):
        """
        Trains the model on the training dataset and evaluates it on the validation dataset.

        Parameters:
        - X_train (np.array): Training dataset features.
        - y_train (np.array): Training dataset labels.
        - X_val (np.array): Validation dataset features.
        - y_val (np.array): Validation dataset labels.
        - epochs (int): Number of training epochs.
        - lr (float): Learning rate.
        - patience (int): Patience parameter for early stopping.

        Returns:
        - Tuple[List[float], List[float], float]: Training and validation metrics over epochs, and the final evaluation metric.
        """

        metrics = []
        val_metrics = []
        best_val_metric = np.inf
        patience_counter = 0  

        for epoch in range(epochs):
            # Update model weights using backpropagation
            self.backpropagation(X_train, y_train, lr)
        
            # Evaluate model performance on both training and validation sets
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

            # # Print progress
            # if epoch % 100 == 0:
            #     if self.n_output == 1:
            #         print(f"Epoch {epoch}/{epochs}, Train MSE: {train_metric}, Val MSE: {val_metric}")
            #     else:
            #         print(f"Epoch {epoch}/{epochs}, Train Loss: {train_metric}, Val Loss: {val_metric}")

            # Implement early stopping based on validation performance
            if val_metric < best_val_metric:
                best_val_metric = val_metric
                patience_counter = 0  # Reset counter if there's an improvement

                # Save the current best model weights and biases
                self.best_W_hidden_1 = self.W_hidden_1.copy()
                self.best_W_hidden_2 = self.W_hidden_2.copy()
                self.best_W_output = self.W_output.copy()
                self.best_b_hidden_1 = self.b_hidden_1.copy()
                self.best_b_hidden_2 = self.b_hidden_2.copy()
                self.best_b_output = self.b_output.copy()

            else:
                patience_counter += 1  # Increment counter if no improvement

            # Stop training if patience is exceeded
            if patience_counter > patience:
                # print(f"No improvement in validation metric for {patience} epochs, stopping training.")
                break

        # Load best model state if early stopping was triggered
        self.W_hidden_1 = self.best_W_hidden_1
        self.W_hidden_2 = self.best_W_hidden_2
        self.W_output = self.best_W_output
        self.b_hidden_1 = self.best_b_hidden_1
        self.b_hidden_2 = self.best_b_hidden_2
        self.b_output = self.best_b_output
         
        # Final metric evaluation
        if self.n_output == 1:
            final_metric = train_metric
        else:
            # After the final epoch, calculate the final accuracy for classification
            final_metric = self.calculate_accuracy(y_train, A_output)
            # print(f"Final Accuracy: {final_metric}")
            
        return metrics, val_metrics, final_metric