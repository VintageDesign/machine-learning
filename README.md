# Machine Learning Library
To what ever poor soul who has found this repo, welcome! Enclosed is a collection of ML algorithms I wrote during CSC 692 at SDSM&T.

## Installing
Assuming that you have the repo cloned, we can start with installing the requirements. 

Do this by running the following command:
```
pip install -r requirements.txt
```

Once that finishes, you should be able to utilize this ML package.


## Common Usage
For every algorithm in the package, there are 3 functions that you will need to know.

### Initialization
On init of each of the algorithms there will usually be a couple of optional arguments that modify parameters of the model. Such as weight scales or epochs.
To pass these optional arugments in, simpily declare the object like this:

```
model = Algorithm(scale=.1, epochs=1000)
```

And just like that, you have  inistantiated the algorithm! For more detailed information on the initialization of each algorithm, please see the algorithms documentation or docstring.

### Fitting the Model
Fitting the model is the most important step of the usage. Without fitting the model you cannot use it to predict/classify/etc.

In order to fit the models, each model requires datapoints and the expected label/result for each data point. 

To fit the algorithm, use the following format:
```
model.fit(data_points, labels)
```

### Using the Model



