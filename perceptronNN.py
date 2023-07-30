


class Activation(Layer):
    def __init__(self,activation,derivitive_activation):
        self.activation = activation
        self.derivitive_activation = derivitive_activation

    def activation_sigmoid(x):
        return 1.0/(1+np.exp(-1*x))

    def derivitive_activation(self,x):
        return self.activation_sigmoid(x)*(1-self.activation_sigmoid(x))


'''
Big Picture
1) We feed input data into the NN
2) The data flows from layer to layer until we have an output
3) w/ the output we calc the error which is scalar
4) Adjust weights and biases by subtracting the error_prime wrt the parameter itself
5) repeat


Forward Propagation: Output of one layer is the input of the next layer

GOAL: Minimize the error by changing the weights

'''














