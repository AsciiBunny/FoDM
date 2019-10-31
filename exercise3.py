# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# %%
""" Necessary Function Definitions """


def reLu(X):
    return np.maximum(X, 0)


def delReLu(X):
    return 1 * (X > 0)


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def delSigmoid(X):
    return sigmoid(X) * (1 - sigmoid(X))


def crossEntropy(O, Y):
    Z = -1 * (Y * np.log(O + np.exp(-15)) + (1 - Y) * np.log(1 - O + np.exp(-15)))
    return Z


def delCrossEntropy(O, Y):
    Z = ((1 - Y) / (1 - O + np.exp(-15))) - (Y / (O + np.exp(-15)))
    return Z


class NeuralNetwork:
    """ Defining weights and bias """

    np.random.seed(1)

    Wih = np.random.rand(2, 10)
    bih = np.random.rand(1, 10)

    Whh = np.random.rand(10, 10)
    bhh = np.random.rand(1, 10)

    Who = np.random.rand(10, 1)
    bho = np.random.rand(1, 1)

    J = []  # Cost function
    accuracy = 0

    def FeedForward(self, X):
        self.Zih = np.dot(np.transpose(X), self.Wih) + self.bih
        self.Oih = np.transpose(reLu(self.Zih))

        self.Zhh = np.dot(np.transpose(self.Oih), self.Whh) + self.bhh
        self.Ohh = np.transpose(reLu(self.Zhh))

        self.Zho = np.dot(np.transpose(self.Ohh), self.Who) + self.bho
        self.O = sigmoid(self.Zho)
        return

    def BackPropogate(self, X, Y, alpha, n):
        """ alpha is the learning rate
            n is the batch size """

        err_oh = delCrossEntropy(self.O, Y) * delSigmoid(self.Zho) / n
        grad_ho = np.dot(self.Ohh, err_oh)
        grad_bho = np.sum(err_oh, axis=0)

        err_hh = np.dot(err_oh, np.transpose(self.Who)) * delReLu(self.Zhh)
        grad_hh = np.dot(self.Oih, err_hh)
        grad_bhh = np.sum(err_hh, axis=0)

        err_hi = np.dot(err_hh, np.transpose(self.Whh)) * delReLu(self.Zih)
        grad_ih = np.dot(X, err_hi)
        grad_bih = np.sum(err_hi, axis=0)

        """ Update Weights """
        self.Who -= alpha * grad_ho
        self.bho -= alpha * grad_bho
        self.Whh -= alpha * grad_hh
        self.bhh -= alpha * grad_bhh
        self.Wih -= alpha * grad_ih
        self.bih -= alpha * grad_bih
        return

    def set_to_Zero(self):
        self.Wih = np.zeros((2, 10))
        self.bih = np.zeros((1, 10))

        self.Whh = np.zeros((10, 10))
        self.bhh = np.zeros((1, 10))

        self.Who = np.zeros((10, 1))
        self.bho = np.zeros((1, 1))
        return

    def set_to_normal_distibution(self,sigma):
        self.Wih = np.random.normal(loc=0.0, scale=sigma, size=(2, 10))
        self.bih = np.random.normal(loc=0.0, scale=sigma, size=(1, 10))

        self.Whh = np.random.normal(loc=0.0, scale=sigma, size=(10, 10))
        self.bhh = np.random.normal(loc=0.0, scale=sigma, size=(1, 10))

        self.Who = np.random.normal(loc=0.0, scale=sigma, size=(10, 1))
        self.bho = np.random.normal(loc=0.0, scale=sigma, size=(1, 1))
        return

    def learning_phase(self, data, number_epochs=5000, learning_rate=0.05, to_print = True):
        sc = StandardScaler()
        data["X_0"] = sc.fit_transform(data["X_0"].values.reshape(-1, 1))
        data["X_1"] = sc.fit_transform(data["X_1"].values.reshape(-1, 1))

        TotalData = len(data)
        batches = 10
        n = TotalData // batches

        b1 = 0
        b2 = 1

        alpha = learning_rate
        count = []
        iterations = number_epochs
        for itr in range(iterations):

            for batch in range(batches):
                b1 = batch
                b2 = batch + 1
                X0 = np.reshape(np.array(data.X_0[b1 * n:b2 * n]), (1, n))
                X1 = np.reshape(np.array(data.X_1[b1 * n:b2 * n]), (1, n))
                Y = np.reshape(np.array(data.y[b1 * n:b2 * n]), (n, 1))
                X = np.append(X0, X1, axis=0)
                self.FeedForward(X)
                self.BackPropogate(X, Y, alpha, n)
            j = np.mean(crossEntropy(self.O, Y))
            self.J.append(j)

            if to_print == True:
                print("Cost Function after epoch  %d is %f" % (itr+1, j))

    def validation_phase(self, data_valid, to_print=True):
        sc = StandardScaler()
        data_valid["X_0"] = sc.fit_transform(data_valid["X_0"].values.reshape(-1, 1))
        data_valid["X_1"] = sc.fit_transform(data_valid["X_1"].values.reshape(-1, 1))
        X0_val = np.reshape(np.array(data_valid.X_0), (1, len(data_valid)))
        X1_val = np.reshape(np.array(data_valid.X_1), (1, len(data_valid)))
        Y_val = np.reshape(np.array(data_valid.y), (len(data_valid), 1))
        X_val = np.append(X0_val, X1_val, axis=0)

        self.FeedForward(X_val)

        fin_err = self.O - Y_val
        mat = np.zeros((len(Y_val), 1))
        cr = []  # Correct prediction
        wr = []  # wrong prediction
        unc = []  # uncertain

        for i in range(len(Y_val)):
            if abs(fin_err[i]) <= 0.45:
                cr.append(i)
            elif 0.45 < abs(fin_err[i]) < 0.55:
                unc.append(i)
            else:
                wr.append(i)
        self.accuracy = len(cr)/(len(cr)+len(wr)+len(unc))
        if to_print==True:
            print("Prediction Results")
            print("Number of correct predictions is %d" % len(cr))
            print("Number of wrong predictions is %d" % len(wr))
            print("Number of uncertain predictions is %d" % len(unc))

    def vizualize_layers(self, data):
        sc = StandardScaler()
        data["X_0"] = sc.fit_transform(data["X_0"].values.reshape(-1, 1))
        data["X_1"] = sc.fit_transform(data["X_1"].values.reshape(-1, 1))
        X0_val = np.reshape(np.array(data.X_0), (1, len(data)))
        X1_val = np.reshape(np.array(data.X_1), (1, len(data)))
        Y = np.reshape(np.array(data.y), (len(data), 1))
        X = np.append(X0_val, X1_val, axis=0)

        self.Zih = np.dot(np.transpose(X), self.Wih) + self.bih
        self.Oih = np.transpose(reLu(self.Zih))
        activations_layer1 = np.sum(self.Oih, axis=1).reshape((10,1))
        self.Zhh = np.dot(np.transpose(self.Oih), self.Whh) + self.bhh
        self.Ohh = np.transpose(reLu(self.Zhh))
        activations_layer2= np.sum(self.Ohh, axis=1).reshape((10,1))
        showdata = np.concatenate((activations_layer1, activations_layer2), axis=1)
        plt.imshow(showdata, cmap='hot')
        plt.show()
        return
