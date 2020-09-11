# This is a sample Python script.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


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
    print(perceptron.get_weights())
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
