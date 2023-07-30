import numpy as np
import pandas as pd

from network import Network,FCLayer
from activation import ActivationLayer,tanh,tanh_prime,mse,mse_prime

df = pd.read_csv('Linearly_Sepa_Data.csv')

msk = np.random.rand(len(df)) < 0.8
df_train = df[msk]
df_test = df[~msk]

x_train = df_train[['X1', 'X2']].values
y_train = df_train['y']
x_train = np.array(x_train)
y_train = np.array(y_train)

x_test = df_test[['X1', 'X2']].values
y_test = df_test['y']
x_test = np.array(x_test)
y_test = np.array(y_test)


net = Network()
net.add(FCLayer(2, 3))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(3, 1))
net.add(ActivationLayer(tanh, tanh_prime))

net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)
out = net.predict(x_train)

x_cord = float(input("Enter your X-value: "))
y_cord = float(input("Enter your Y-value: "))
input_cord = (x_cord, y_cord)

predicted_class = Network.predict_class(net, input_cord)
print(f"{input_cord}: Estimated Class {predicted_class}")




