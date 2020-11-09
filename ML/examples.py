import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ML import Perceptron, LinearRegression, MultiVariateLinearRegression, Stump, LogisticRegression


def perceptron_example():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    x = df.iloc[0:100, [0, 2]].values
    xlabel_text = "Index 0"
    ylabel_text = "Index 2"
    zlabel_text = "Index 3"

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label='setosa')
    ax.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='x', label='versicolor')
    ax.set_xlabel(xlabel_text)
    ax.set_ylabel(ylabel_text)
    ax.legend()

    perceptron = Perceptron(.1, 100)
    perceptron.fit(x, y)
    # print(perceptron.get_weights())
    plt.figure()
    perceptron.plot_decision_regions(x, y, perceptron)
    plt.xlabel(xlabel_text)
    plt.ylabel(ylabel_text)
    plt.show()


def linear_regression_example():
    df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv')
    df = df[df['price'] > 1]
    x = df.iloc[1:, 6].values
    y = df.iloc[1:, 0].values
    print(y[0])


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x[:], y[:], color='blue', marker='o', label='diamonds')
    ax.legend()

    regession = LinearRegression()

    regession.fit(x, y)

    fit_x = np.linspace(np.min(x), np.max(x), 100)
    fit_y = regession.get_slope() * fit_x + regession.get_intercept()
    ax.plot(fit_x, fit_y)
    plt.show()


def multilinear_regression_example():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

    x = df.iloc[0:50, [0, 1]].values
    y = df.iloc[0:50, 2].values

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[:, 0], x[:, 1], y[:], color='blue', marker='o')
    #ax.legend()

    regession = MultiVariateLinearRegression(.001, 1000)

    regession.fit(x, y)

    weights = regession.get_weights()
    fit_x = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 100)
    fit_z = np.linspace(np.min(x[:, 1]), np.max(x[:, 1]), 100)
    fit_y = weights[2] * fit_z + weights[1] * fit_x + weights[0]

    ax.plot(fit_x, fit_z, fit_y)
    plt.show()

def stump_example():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    x = df.iloc[0:100, [0, 2]].values
    xlabel_text = "Index 0"
    ylabel_text = "Index 2"
    zlabel_text = "Index 3"

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label='setosa')
    ax.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='x', label='versicolor')
    ax.set_xlabel(xlabel_text)
    ax.set_ylabel(ylabel_text)
    ax.legend()

    stump = Stump()
    stump.fit(x, y)
    print(stump.get_threshold())
    fit_x = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 100)
    fit_y = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 100)
    fit_y[:] = stump.get_threshold()
    ax.plot(fit_x, fit_y)
    plt.show()

def logistic_regression_example():
    dataset = [[2.7810836, 2.550537003, 0],
               [1.465489372, 2.362125076, 0],
               [3.396561688, 4.400293529, 0],
               [1.38807019, 1.850220317, 0],
               [3.06407232, 3.005305973, 0],
               [7.627531214, 2.759262235, 1],
               [5.332441248, 2.088626775, 1],
               [6.922596716, 1.77106367, 1],
               [8.675418651, -0.242068655, 1],
               [7.673756466, 3.508563011, 1]]
    l_rate = 0.6
    n_epoch = 100
    classifier = LogisticRegression()
    coef = classifier.coefficients_sgd(dataset, l_rate, n_epoch)
    print(coef)
