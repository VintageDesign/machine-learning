# This is a sample Python script.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from matplotlib.colors import ListedColormap


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
    plt.show()


def main():
    # Use a breakpoint in the code line below to debug your script.
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    x = df.iloc[0:100, [0, 2]].values

    plt.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='x', label='versicolor')
    plt.xlabel("Petal Length")
    plt.ylabel("Sepal Length")
    plt.legend()
    #plt.show()

    perceptron = Perceptron(.1, 10)

    perceptron.fit(x, y)
    #print(perceptron.get_weights())

    plot_decision_regions(x, y, perceptron)
# Press the green button in the gutter to run the script.


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
