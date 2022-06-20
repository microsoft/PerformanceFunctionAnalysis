import numpy as np
from scipy.optimize import least_squares


class AMUEPerfFunc:
    """
    Additive Model With Unequal Elasiticities Performance function.
    Has the following functional form:

    """

    def __init__(self):
        self.theta = np.array([20, 1, 1, 1, 1])

    def __call__(self, X, theta=None):
        """
        Gets the performance prediction for input X 
        which will contain amount of manual and translated data for different configurations
        
        Inputs:
            - X (np.ndarray) : Matrix of shape [N, 2] containing amounts of manual and translated data for N configurations
            - theta (np.ndarray): Parameters of AMUE Model, shape: [5,]. If None, the stored parameters will be used
        Returns a numpy array of shape [N,] containing performance prediction for each
        """

        if theta is None:
            theta = self.theta

        a0 = theta[0]
        a1 = theta[1]
        alpha1 = theta[2]
        a2 = theta[3]
        alpha2 = theta[4]
        return a0 + a1 * (X[:, 0] ** alpha1) + a2 * (X[:, 1] ** alpha2)

    def residual_function(self, theta, *args, **kwargs):

        """
        Computes the residual for AMUE or the error between the prediction and the actual values

        Inputs:
            - theta (np.ndarray): Parameters of AMUE Model, shape: [5,].

        Additionally X,y must be supplied as *args where
            - X (np.ndarray) : Matrix of shape [N, 2] containing amounts of manual and translated data for N configurations
            - y (np.ndarray) : Numpy array of shape [N,] containing performance of the corresponding configurations
        
        Returns the residual for each configuration i.e. a numpy array of shape [N,]
        
        """

        X = args[0]
        y = args[1]

        y_pred = self(
            X, theta
        )
        residual = y_pred - y
        return residual

    def fit(self, X, y):

        """
        Fits the AMUE performance function to estimate its parameters stored in `self.theta`

        Inputs:
            - X (np.ndarray) : Matrix of shape [N, 2] containing amounts of manual and translated data for N configurations
            - y (np.ndarray) : Numpy array of shape [N,] containing performance of the corresponding configurations
        """

        opt_results = least_squares(
            self.residual_function,
            x0=self.theta,
            bounds=([0, 0, 0, 0, 0], [np.inf, np.inf, 1, np.inf, 1]),
            args=(X, y),
        )
        self.theta = opt_results.x

