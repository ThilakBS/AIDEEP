import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Sample_Data2.csv')

for column in data.columns:
    data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())


x = data.iloc[:, 0:2]
ones = np.ones((x.shape[0], 1))
x = np.concatenate((ones, x), axis=1)

y = data.iloc[:, 2:3].values

weights = np.zeros([1, 3])

lr = 0.01
iters = 1000



def findloss(x, y, weights):
    loss_num = np.power(((x @ weights.T) - y), 2)
    return np.sum(loss_num) / (2 * len(x))


def desent(x, y, weights, iters, lr):
    loss = np.zeros(iters)
    for i in range(iters):
        loss[i] = findloss(x, y, weights)
        weights = weights - (lr / (len(x))) * np.sum(x * (x @ weights.T - y), axis=0)

    return weights, loss


f_weights, loss = desent(x, y, weights, iters, lr)


f_loss = findloss(x, y, f_weights)
#print(f_loss)

k = f_weights.size

def create_x_list(k):
    for i in range(k):
        power = x ** i
    return power

