import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
from math import exp
from ML import PCA, Perceptron, LinearRegression, MultiVariateLinearRegression, Stump, LogisticRegression, KNN, SVM
from scipy.io import wavfile




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
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

    y = df.iloc[0:100, 4].values
    classes = np.where(y == 'Iris-setosa', 0, 1)

    dataset = df.iloc[0:100, 2].values
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataset[:50], classes[:50], color='red', label='Not Setosa')
    ax.scatter(dataset[50:], classes[50:], color='blue', label='Setosa')
    ax.legend()
    l_rate = 0.5
    n_epoch = 100
    classifier = LogisticRegression(l_rate, n_epoch)
    classifier.fit(dataset, classes)
    coef = classifier.get_coeff()
    print(coef)

    fit_x = np.linspace(np.min(dataset[:]), np.max(dataset[:]), 100)
    fit_y = classifier.predict(fit_x)

    ax.plot(fit_x, fit_y)
    plt.show()

def knn_example():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

    dataset = df.iloc[0:150, :].values

    classifier = KNN(8, dataset[:149, :])

    new_class = classifier.predict(dataset[149, :-1])

    print(new_class)


def svm_example():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)


    x = df.iloc[0:100, [2, 3, 4]].values
    x[:, 2] = np.where(x[:, 2] == 'Iris-setosa', -1, 1)
    xlabel_text = "Index 0"
    ylabel_text = "Index 2"

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label='setosa')
    ax.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='x', label='versicolor')
    ax.set_xlabel(xlabel_text)
    ax.set_ylabel(ylabel_text)
    ax.legend()

    classifer = SVM(.001, 1000, .01)
    classifer.fit(x)

    fit_x = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 100)
    w = classifer.get_weights()
    fit_y = (w[0] + fit_x * w[1]) / w[2]
    plt.plot(fit_x, fit_y)
    #plt.fill_between(fit_x, fit_y - np.linalg.norm(w), fit_y + np.linalg.norm(w), edgecolor='none',
                     #color='#AAAAAA', alpha=0.4)


    plt.xlabel(xlabel_text)
    plt.ylabel(ylabel_text)
    plt.show()

    print(classifer.get_weights())


def read_data():
    data = []
    min_size = 3000000
    folder_name_list = []
    for foldername in sorted(os.listdir(os.path.join(os.getcwd(), 'dataset'))):
        for filename in sorted(os.listdir(os.path.join(os.path.join(os.getcwd(), 'dataset'),foldername))):
            sample, single_file = wavfile.read(os.path.join(os.getcwd() + '/dataset/' + foldername, filename))
            mfcc = librosa.feature.mfcc(single_file[:,0].astype(np.float64).T, sr=sample)
            min_size = min(mfcc.flatten().shape[0], min_size)
    for foldername in sorted(os.listdir(os.path.join(os.getcwd(), 'dataset'))):
        for filename in sorted(os.listdir(os.path.join(os.path.join(os.getcwd(), 'dataset'),foldername))):
            sample, single_file = wavfile.read(os.path.join(os.getcwd() + '/dataset/' + foldername, filename))
            folder_name_list.append(foldername)
            # time = np.linspace(0, single_file.shape[0]/ sample, single_file.shape[0])
            # plt.plot(time, single_file[:, 0])
            # plt.plot(time, single_file[:, 1])
            # plt.show()
            single_file = librosa.feature.mfcc(single_file[:, 0].astype(np.float64).T, sr=sample)
            single_file = single_file.flatten()
            data.append(single_file[:min_size])
    data = np.vstack(data)
    return min_size, data, folder_name_list


def read_tests(min_size):
    data = []
    classes = []
    for filename in sorted(os.listdir(os.path.join(os.getcwd(), 'tests'))):
        sample, single_file = wavfile.read(os.path.join(os.getcwd() + '/tests/', filename))
        classes.append(filename)
        single_file = librosa.feature.mfcc(single_file[:, 0].astype(np.float64).T, sr=sample)
        single_file = single_file.flatten()
        data.append(single_file[:min_size])
    data = np.vstack(data)
    return classes, data


def compute_energy_recovery(U, S, target_energy):
    k = 0
    actual_energy = 0
    norm2 = np.sum(S)

    while target_energy > actual_energy:
        k += 1
        sigma_val = np.sum(S[:k])
        actual_energy = sigma_val / norm2

    return k


def pca_example():
    min_size, data, classes = read_data()
    classes = np.array(classes, dtype=np.str)
    # Map class names
    list_of_classes = np.unique(classes).tolist()
    for name in list_of_classes:
        classes = np.where(classes == name, list_of_classes.index(name), classes)
    test_classes, test_cases = read_tests(min_size)

    pca = PCA(data, classes)

    for i in range(len(test_classes)):
        point = test_cases[i, :]
        index = pca.predict(point)
        print("Predicted:", list_of_classes[index])
        print("Actual:", test_classes[i])
