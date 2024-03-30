import numpy as np

class LinearNetwork:

    def __init__(config) -> None:
        self.config = config
        
    def softmax(z):
        exp_scores = np.exp(z)
        probs = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
        return probs
    
    def cross_entropy_loss(y, probs):
        # Compute the cross-entropy loss
        N = y.shape[0]
        cross_entropy_loss = -np.sum(y * np.log(probs)) / N
        return cross_entropy_loss
    