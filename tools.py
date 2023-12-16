#!/usr/bin/env python
# -*- coding=utf-8 -*-
#
#

"""Documention for Module/Script ( tools ).
----------------------------------------
 Description:
     This is a script that contains all of our tools
---------------
 Info:
     Author = Mahdi
     Date = 01/12/2023 11:04:51
---------------
 TODO :
     --> Add more awesome stuffs.
"""


### Import modules

import matplotlib.pyplot as plt
import numpy as np


### Functions
def PNorms(x, p, Ax):
    """
    p-Norm calculator

    This function calculates the pth norm of x using axis Ax

    Parameters
    ----------
    x : Array
        A vector or matrix
    p : Integer
        A integer as order of norm
    Ax : Integer
        A integer as axis of calculation

    Raises
    ------
    TypeError:
        If p isn't an integer, TypeError
    """
    if isinstance(p, int):
        raise TypeError("Please enter p as integer.")
    else:
        return np.sum(np.abs(x) ** p, axis=Ax) ** (1 / p)


def FrobNorm(x):
    """
    Frobenuis norm calculator

    This function calculates the Frobenuis norm of x

    Parameters
    ----------
    x : Array
        A matrix
    """
    n, d = x.shape
    res = 0.0
    for i in range(n):
        for j in range(d):
            res += x[i, j] ** 2

    return res**0.5


class GaussElimination:
    """Documention for Module/Script ( GaussElimination ).
    ----------------------------------------
     Description:
         This class solves a linear system like Ax=b by GaussElimination
         algorithm
    ---------------
     Attributes:
         Inputs --> A : Array, b : Array, size : Integer
         Outputs --> res : Array
    ---------------
    ---------------
     TODO:
         --> Advance this class
    """

    def __init__(self, A, b, size):
        self.A = A
        self.b = b
        self.size = size

    def Forward(self):
        A = self.A
        b = self.b

        for i in range(self.size):
            A[i, :] = A[i, :] / A[i, :] * A[i, i]
            b[i] = b[i] / A[i, i]

            for j in range(i + 1, self.size):
                A[j, :] = A[j, :] - A[i, :] * A[j, i]
                b[j] = b[j] - b[i] * A[j, i]
                A[j, i] = 0
        return A, b

    def Backward(self):
        res = np.zeros((self.size, 1))
        for i in range(self.size - 1, -1, -1):
            res[i] = self.b[i]
            for j in range(i + 1, self.size):
                res[i] = res[i] - self.A[i, j] * res[j]
        return res

    def Solve(self):
        A, b = self.Forward()
        res = self.Backward()
        return res


def LU_partial_decomposition(matrix):
    """
    LU factorizator

    This function calculates the LU decomposition of matrix

    Parameters
    ----------
    matrix : Array
        A non-conditional matrix
    """
    n, m = matrix.shape
    P = np.identity(n)
    L = np.identity(n)
    U = matrix.copy()
    PF = np.identity(n)
    LF = np.zeros((n, n))
    for k in range(0, n - 1):
        index = np.argmax(abs(U[k:, k]))
        index = index + k
        if index != k:
            P = np.identity(n)
            P[[index, k], k:n] = P[[k, index], k:n]
            U[[index, k], k:n] = U[[k, index], k:n]
            PF = np.dot(P, PF)
            LF = np.dot(P, LF)
        L = np.identity(n)
        for j in range(k + 1, n):
            L[j, k] = -(U[j, k] / U[k, k])
            LF[j, k] = U[j, k] / U[k, k]
        U = np.dot(L, U)
    np.fill_diagonal(LF, 1)
    return PF, LF, U


def conditionNumber(A, order):
    """
    Condition Number function (basic)

    This function basicly calculates the condition number of matrix A with order
    of norm order.

    Parameters
    ----------
    A : Array
        A square matrix
    order : Integer
        Order of norm
    """
    assert len(A.shape) == 2
    return np.linalg.norm(A, ord=order) * np.linalg.norm(np.linalg.inv(A), ord=order)


def pca(D):
    mu = np.mean(D, axis=1)
    X = D - np.ones(D.shape[0]) * mu
    C = np.cov(X.T)
    Eval, Evec = np.linalg.eig(C)
    return Eval, Evec


def Lines(m, b, color="b--", lw=0.1):
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    x = np.arange(-10, 10, 0.1)
    plt.plot(x, m * x + b, color, lw)


def GDAlgorithms(X, y, learning_rate, num_iterations, algorithm):
    # Initialize parameters
    m, n = X.shape
    theta = np.zeros(n)
    loss_history = []

    for iteration in range(num_iterations):
        # Calculate predictions and errors
        predictions = X.dot(theta)
        errors = predictions - y

        # Update parameters based on the selected algorithm
        if algorithm == "batch":
            gradient = (1 / m) * (X.T.dot(errors))
            theta = theta - learning_rate * gradient
        elif algorithm == "stochastic":
            for i in range(m):
                random_index = np.random.randint(m)
                xi = X[random_index : random_index + 1]
                yi = y[random_index : random_index + 1]
                prediction = xi.dot(theta)
                error = prediction - yi
                gradient = xi.T.dot(error)
                theta = theta - learning_rate * gradient
        elif algorithm == "mini-batch":
            batch_size = 50
            random_indices = np.random.choice(m, batch_size, replace=False)
            xi = X[random_indices]
            yi = y[random_indices]
            predictions = xi.dot(theta)
            errors = predictions - yi
            gradient = (1 / batch_size) * (xi.T.dot(errors))
            theta = theta - learning_rate * gradient
        else:
            raise ValueError(
                'Please choose either "batch", "stochastic", or "mini-batch".'
            )

        # Calculate and store the loss
        loss = np.mean(errors**2)
        loss_history.append(loss)

    return theta, loss_history


def optimization_algorithm(key):
    if key == "Adam":
        return AdamOptimizer()
    elif key == "RMSprop":
        return RMSpropOptimizer()
    elif key == "AdaGrad":
        return AdaGradOptimizer()
    else:
        raise ValueError("Invalid optimization algorithm key.")


class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-5):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, parameters, gradients):
        self.t += 1
        for param_key in parameters.keys():
            if param_key not in self.m:
                self.m[param_key] = np.zeros_like(parameters[param_key])
                self.v[param_key] = np.zeros_like(parameters[param_key])

            self.m[param_key] = (
                self.beta1 * self.m[param_key] + (1 - self.beta1) * gradients[param_key]
            )
            self.v[param_key] = self.beta2 * self.v[param_key] + (
                1 - self.beta2
            ) * np.square(gradients[param_key])
            m_hat = self.m[param_key] / (1 - np.power(self.beta1, self.t))
            v_hat = self.v[param_key] / (1 - np.power(self.beta2, self.t))
            parameters[param_key] -= (
                self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            )

        return parameters


class RMSpropOptimizer:
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-5):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.cache = {}

    def update(self, parameters, gradients):
        for param_key in parameters.keys():
            if param_key not in self.cache:
                self.cache[param_key] = np.zeros_like(parameters[param_key])

            self.cache[param_key] = self.beta * self.cache[param_key] + (
                1 - self.beta
            ) * np.square(gradients[param_key])
            parameters[param_key] -= (
                self.learning_rate
                * gradients[param_key]
                / (np.sqrt(self.cache[param_key]) + self.epsilon)
            )

        return parameters


class AdaGradOptimizer:
    def __init__(self, learning_rate=0.001, epsilon=1e-5):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.cache = {}

    def update(self, parameters, gradients):
        for param_key in parameters.keys():
            if param_key not in self.cache:
                self.cache[param_key] = np.zeros_like(parameters[param_key])

            self.cache[param_key] += np.square(gradients[param_key])
            parameters[param_key] -= (
                self.learning_rate
                * gradients[param_key]
                / (np.sqrt(self.cache[param_key]) + self.epsilon)
            )

        return parameters


class LinearRegression:
    def __init__(self, optimizer="Adam", learning_rate=0.01, num_iterations=100):
        self.optimizer = optimization_algorithm(optimizer)
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.parameters = None

    def fit(self, X, y):
        self.parameters = self.initialize_parameters(X.shape[1])
        for iteration in range(self.num_iterations):
            gradients = self.compute_gradients(X, y)
            self.parameters = self.optimizer.update(self.parameters, gradients)

    def predict(self, X):
        if self.parameters is None:
            raise ValueError("Model has not been trained yet.")
        return np.dot(X, self.parameters["weight"]) + self.parameters["bias"]

    def initialize_parameters(self, num_features):
        parameters = {"weight": np.zeros(num_features), "bias": 0}
        return parameters

    def compute_gradients(self, X, y):
        num_samples = X.shape[0]
        y_pred = self.predict(X)
        gradients = {
            "weight": (1 / num_samples) * np.dot(X.T, (y_pred - y)),
            "bias": (1 / num_samples) * np.sum(y_pred - y),
        }
        return gradients
