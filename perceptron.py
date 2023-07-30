import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = np.recfromcsv('Linearly_Sepa_Data.csv')
data = np.asarray(data)
data_len = len(data)

p_x_val = []
p_y_val = []
n_x_val = []
n_y_val = []


x = []
y = []
for i in range(data_len):
    x = np.append(x, data[i][0])
    y = np.append(y, data[i][2])

for i in range(data_len):
    if data[i][2] == 1:
        p_x_val = np.append(p_x_val, data[i][0])
        p_y_val = np.append(p_y_val, data[i][1])
    elif data[i][2] == 0:
        n_x_val = np.append(n_x_val, data[i][0])
        n_y_val = np.append(n_y_val, data[i][1])
    else:
        print("Error")


def step_fun(q):
    if q > 0:
        return 1.0
    else:
        return 0



class Perceptron:
    def __init__(self, lr, iters):
        self.lr = lr
        self.iters = iters
        self.act_fun = step_fun
        self.weights = None
        self.bias = None

    def train(self, x, y):
        samples, features = data.shape

        self.weights = np.zeros(features)
        self.bias = 0

        y_ = np.where(y >= 0, 1, 0)

        for u in range(self.iters):
            for index, x_i in enumerate(x):
                output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.act_fun(output)

                update = self.lr * (y_[index] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def test(self, x):
        output = np.dot(x, self.weights) + self.bias
        y_predicted = self.act_fun(output)
        return y_predicted



