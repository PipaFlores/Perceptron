# %%
import numpy as np

class Perceptron(object):

    def __init__(self, eta=0.01, n_iter=10):
        # Initialize learning rate (eta) and number of iterations
        self.eta = eta
        self.n_iter = n_iter

    def weighted_sum(self, X):
        # Calculate the weighted sum of inputs
        # X: input features
        # w_[1:]: weights (excluding bias)
        # w_[0]: bias term
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        # Make predictions based on the weighted sum
        # Returns 1 if weighted sum is >= 0, else -1
        return np.where(self.weighted_sum(X) >= 0.0, 1, -1)
    
    def fit(self, X, y):
        # Train the perceptron
        # X: input features
        # y: target values

        # Initialize weights (including bias) with zeros
        self.w_ = np.zeros(1 + X.shape[1])
        # List to store number of misclassifications in each epoch
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                # Calculate the update value
                update = self.eta * (target - self.predict(xi))
                # Update weights (excluding bias)
                self.w_[1:] += update * xi
                # Update bias
                self.w_[0] += update
                # Count number of misclassifications
                errors += int(update != 0.0)
            # Append number of misclassifications for this epoch
            self.errors_.append(errors)
        return self
