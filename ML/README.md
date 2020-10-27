# Machine Learning Package
###### Author: Riley Kopp
Enclosed is the detailed descriptions of each ML algorithm. See the repo's readme for details on installing the package.

## Table of Contents
- [Perceptron](#Perceptron)
- [Linear Regression](#Linear-Regression)
- [Decision Stump](#Decision-Stump)
## Perceptron
### How to Use:
To use the perceptron in your own code use the following line:
```
from ML import Perceptron
```

Then create the object of the perceptron by running:
```
classifier = Perceptron(<scale>, <epochs>)
```
Then fit the model by calling:
```
classifier.fit(X, y)
```
Where `X` is an `m x n` matrix where `m` is the number of training data points and `n` is the number of features within 
each datapoint and `y` is a vector of size `m` that contains the label of the corresponding datapoint in `X`.

Once the model is fitted, there are a few more functions that can be run. To view the current weights of the perceptron,
call `classifier.get_weights()`, this will return a vector of size `w` where `w = n + 1` and `w_0` is the bias of the
classifier.

To run the classifier on testing data, call `classifier.perdict(X)`

### What Does the Perceptron Do?
At a high level, the perceptron is a binary classifier. Meaning it can separate items into two classes based on their 
features. At a lower level, the perceptron takes a dot product of the feature set `X_i` and the weight vector `w` and 
adds the bias. From there the sum of the dot product and the bias, `u`, is run through the following step function:
```
S(u) = u > 1 ? 1 : -1
```

The weights are determined during `fit()` using the following equation:
```
w = w + (y_actual - y_predicted) * scale * X_i)
```

And the bias is determined by using:
```
bias = bias + (y_actual - y_predicted) * scale)
```

Both `w` and `bias` are updated after every prediction using the following algorithm:
```
For Each epoch:
    error = 0
    For i in X:
        y_predicted = classifer.predict(i)
        w = w + (y_actual - y_predicted) * scale * X_i)
        bias = bias + (y_actual - y_predicted) * scale)
        error += abs(y_actual - y_predicted)
    End
    If error = 0
        break
    End
End
```

Once the Perceptron is fit, the Perceptron can be run on test data.
[Code][Perceptron.py]

## Linear Regression
### How to Use:
To use the Linear Regression Class in your own code use the following line:
```
from ML import LinearRegression
```

Then create the object of the Linear Regression Class by running:
```
classifier = LinearRegression()
```
Then fit the model by calling:
```
classifier.fit(x, y)
```
Where `x` is a vector of size `n` and contains the independent data points and where `y` is a vector of size `n` and 
contains the dependent data points.
  
Once the model is fitted, the functions `get_slope()` and `get_intercept()` can be called to obtain the constants for
linear equation that was fit to the data points.

### What Does Linear Regression do?
No Idea
[Code](LinearRegression.py)
## Decision Stump


[Perceptron.py]: Perceptron.py
