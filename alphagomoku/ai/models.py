import sklearn
import numpy as np
from sklearn.neural_network import MLPRegressor

class NeuralNetworks:
    def __init__(self, input_size, output_size, hidden_layers, **args):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers

        if "verbose" in args:
            self.verbose = args["verbose"]

        self.models = list()

    def init(self):
        self.models = list()
        for i in range(len(self.output_size)):
            hidden_layers = self.hidden_layers[i]
            model = MLPRegressor(hidden_layer_sizes=hidden_layers)

            X_init, y_init = np.random.random((1,self.input_size)), np.random.random((1,self.output_size[i]))
            model.fit(X_init,y_init)
            self.models.append(model)

    def predict(self, X):
        ys = list()
        for i in range(len(self.output_size)):
            y = self.models[i].predict(X)[0]
            ys.append(y)
        return ys

    def update(self, X_train, y_train):
        for i in range(len(self.output_size)):
            self.models[i].partial_fit(X_train,y_train[i])

    def evaluate(self, X, y):
        scores = list()
        for i in range(len(self.output_size)):
            score = self.models[i].score(X,y[i])
            scores.append(score)
        return scores

class RandomForest:
    def __init__(self):
        pass
