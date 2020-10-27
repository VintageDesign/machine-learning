import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ML import Perceptron, LinearRegression, Stump


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
