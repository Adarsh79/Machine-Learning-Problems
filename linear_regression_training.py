import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self, 
        X: NDArray[np.float64], 
        Y: NDArray[np.float64], 
        num_iterations: int, 
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        '''
            task: return the final weights after training in the form of a NumPy array with dimension 3.

            input:
            - `X`: the dataset for training the model. `X.length = n and X[i].length = 3 for 0 <= i < n`.
            - `Y`: the correct answers from the dataset. `Y.length = n`.
            - `num_iterations`: the number of iterations to run gradient descent for. `num_iterations > 0`.
            - `initial_weights`: the initial weights for the model (w1, w2, w3). `initial_weights.length = 3`.
        '''

        # you will need to call get_derivative() for each weight
        # and update each one separately based on the learning rate!
        # return np.round(your_answer, 5)
        N = len(X)
        for i in range(num_iterations):
            y_pred = self.get_model_prediction(X, initial_weights)

            d1 = self.get_derivative(y_pred, Y, N, X, 0)
            d2 = self.get_derivative(y_pred, Y, N, X, 1)
            d3 = self.get_derivative(y_pred, Y, N, X, 2)

            initial_weights[0] -= self.learning_rate * d1
            initial_weights[1] -= self.learning_rate * d2
            initial_weights[2] -= self.learning_rate * d3
        
        return np.round(initial_weights, 5)