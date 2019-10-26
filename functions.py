# import required packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops


# define the sigmoid function


def sigmoid(z):
    g = 1.0 / (1.0 + np.exp(-z))
    return g


# define the tanh function


def tanh(z):
    g = (1-np.exp(-2*z))/(1+np.exp(-2*z))
    return g


# compute the cost using theta, x, y, lambda


def cost_function(theta, x, y, lambda_):

    # initialize the parameters
    m = y.shape[0]
    j = 0
    grad = np.zeros(theta.shape)
    # print(m, j, grad)

    # compute the cost
    h = sigmoid(np.dot(x, theta))
    j = (-np.dot(y.T, np.log(h)) - np.dot((1 - y).T, np.log(1 - h))) / m
    # print(h, j)

    theta_g = theta
    theta_g[0] = 0
    # print(theta, theta_g)

    j = j + np.dot(theta_g.T, theta_g) * lambda_ / (2*m)

    grad = np.dot(x.T, (h - y)) / m
    grad = grad + theta_g * lambda_ / m

    return j, grad


def cost_function2d(theta, x, y, lambda_):

    # initialize the parameters
    m = y.shape[0]
    j = 0
    grad = np.zeros(theta.shape)
    # print(m, j, grad)

    # compute the cost
    h = sigmoid(np.dot(x, theta))
    # j = (-np.dot(y.T, np.log(h)) - np.dot((1 - y).T, np.log(1 - h))) / m
    j = np.sum((-y * np.log(h) - (1 - y) * np.log(1 - h))) / m
    # print(h, j)

    theta_g = theta
    theta_g[0, :] = 0
    # print(theta, theta_g)

    # j = j + np.dot(theta_g.T, theta_g) * lambda_ / (2*m)
    j = j + np.sum(theta_g * theta_g) * lambda_ / (2*m)

    grad = np.dot(x.T, (h - y)) / m
    grad = grad + theta_g * lambda_ / m

    return j, grad


# define the first forward propagation layer.
def first_layer_to_hidden_layer(x, w1, b1):
    # INPUT:
    # x: shape (m, pixel number)
    # w1: shape (pixel number+1, hidden layer neuron number)
    # b1: shape (1, hidden layer neuron number)
    # HIDDEN PARAMETER:
    # a1: shape (m, pixel number+1)
    # OUTPUT:
    # z1: shape (m, hidden layer neuron number)
    a1 = np.hstack((np.ones((x.shape[0], 1)), x))
    # print("a1", a1)
    z1 = np.dot(a1, w1) + b1
    z1 = z1
    return z1, a1


# define the forward propagation of hidden layer to output layer.
def hidden_layer_to_output_layer(z1, w2, b2):
    # INPUT:
    # z1: shape (m, hidden layer neuron number)
    # w2: shape (hidden layer neuron number+1, output layer neuron number)
    # b2: shape (1, output layer neuron number)
    # HIDDEN PARAMETER:
    # a2: shape (m, hidden layer neuron number+1)
    # OUTPUT:
    # z2: shape (m, output layer neuron number)
    # print("z1:", z1)

    # activation function:
    # np.tanh(z1), np.maximum(0, z1)
    a2 = np.tanh(z1)
    # print("a2:", a2)
    a2 = np.hstack((np.ones((z1.shape[0], 1)), a2))
    # print("a2:", a2)
    z2 = np.dot(a2, w2) + b2
    # print("z2:", z2)
    return z2, a2


# output layer analysis compute cost and back propagate
def output_layer_analysis(z2, y, a1, a2, w2):
    # , lambda_w1, lambda_b1, lambda_w2, lambda_b2
    # INPUT:
    # z2: shape (m, output layer neuron number)
    # y: shape (m, number of labels i.e. output layer neuron numbers)
    # a1: shape (m, pixel number+1)
    # a2: shape (m, hidden layer neuron number+1)
    # w2: shape (hidden layer neuron number+1, output layer neuron number)
    # HIDDEN PARAMETER:
    # a3: the same shape of y.
    # OUTPUT:
    # j: scalar number.
    m = y.shape[0]
    a3 = sigmoid(z2)

    # cost function without regularization
    # print("a3:", a3)
    j = np.sum((-y * np.log(a3) - (1 - y) * np.log(1 - a3))) / m
    # 暂时不加正则化项，先测试系统。
    dz2 = a3 - y  # dz2: shape (m, number of labels)
    dw2 = np.dot(a2.T, dz2)/m  # dw2: shape (hidden layer neuron number+1, number of labels)
    db2 = np.sum(dz2, axis=0, keepdims=True)/m  # db2: shape (1, number of labels)

    # if the activation function is:
    # tanh: dz1 = np.dot(dz2, w2.T)*(1-np.power(a2, 2))
    # ReLU: dz1 = np.dot(dz2, w2.T)*((a2 > 0)+0)
    dz1 = np.dot(dz2, w2.T)*(1-np.power(a2, 2))  # dz1: shape (m, hidden layer neuron number+1)

    dw1 = np.dot(a1.T, dz1)/m  # dw1: shape (pixel number+1, hidden layer neuron number+1)
    dw1 = np.delete(dw1, 0, axis=1)  # dw1: shape (pixel number+1, hidden layer neuron number+1)
    db1 = np.delete(dz1, 0, axis=1)  # db1: shape (m, hidden layer neuron number)
    db1 = np.sum(db1, axis=0, keepdims=True)/m  # db1: shape (1, hidden layer neuron number)
    grad = {"dW1": dw1, "db1": db1, "dW2": dw2, "db2": db2}
    return j, grad


# define the whole model
def model(x_train, y_train, x_test, y_test, learning_rate=0.8, hidden_layer_size=32):
    m = x_train.shape[0]
    pixel_number = x_train.shape[1]
    output_layer_number = y_train.shape[1]
    y_train_orig = np.argmax(y_train, axis=1)
    # initialize parameters:
    w1 = np.random.randn(pixel_number+1, hidden_layer_size) * 0.01
    b1 = np.zeros((1, hidden_layer_size))
    w2 = np.random.randn(hidden_layer_size+1, output_layer_number) * 0.01
    b2 = np.zeros((1, output_layer_number))
    z1 = np.zeros((m, hidden_layer_size))
    a1 = np.zeros((m, pixel_number+1))
    z2 = np.zeros((m, output_layer_number))
    a2 = np.zeros((m, hidden_layer_size+1))
    j = 0
    loss = []
    epoch = []
    train_accuracy_list = []

    # w1 = np.array(pd.read_csv('dataset/W1.csv'))
    # w2 = np.array(pd.read_csv('dataset/W2.csv'))
    # b1 = np.array(pd.read_csv('dataset/b1.csv'))
    # b2 = np.array(pd.read_csv('dataset/b2.csv'))

    for i in range(500):
        epoch.append(i+1)
        z1, a1 = first_layer_to_hidden_layer(x_train, w1, b1)
        z2, a2 = hidden_layer_to_output_layer(z1, w2, b2)
        j, grad = output_layer_analysis(z2, y_train, a1, a2, w2)
        loss.append(j)
        dw1, db1, dw2, db2 = grad["dW1"], grad["db1"], grad["dW2"], grad["db2"]

        # gradient decent:
        w1 -= learning_rate * dw1
        b1 -= learning_rate * db1
        w2 -= learning_rate * dw2
        b2 -= learning_rate * db2

        # if i == 10000:
        #     learning_rate = learning_rate/10
        #
        # if j > j_prev:
        #     learning_rate = learning_rate/5
        print("Epoch ", (i+1), ": loss = ", j)
        # test the train set
        # z2: shape (m, output layer neuron number)
        h = np.argmax(sigmoid(z2), axis=1)
        train_accuracy = np.sum(((h - y_train_orig) == 0) + 0) / m
        train_accuracy_list.append(train_accuracy)
        print("The train set's accuracy is: ", train_accuracy)

        if ((i+1) % 500) == 0:
            # plot the data
            plt.plot(epoch, loss)
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.show()
            plt.plot(epoch, train_accuracy_list)
            plt.xlabel("epoch")
            plt.ylabel("train_accuracy")
            plt.show()
            # save theta
            df_w1 = pd.DataFrame(w1)
            df_w1.to_csv("dataset/W1.csv", index=False)
            df_b1 = pd.DataFrame(b1)
            df_b1.to_csv("dataset/b1.csv", index=False)
            df_w2 = pd.DataFrame(w2)
            df_w2.to_csv("dataset/W2.csv", index=False)
            df_b2 = pd.DataFrame(b2)
            df_b2.to_csv("dataset/b2.csv", index=False)

    # test the train set
    # z2: shape (m, output layer neuron number)
    h = np.argmax(sigmoid(z2), axis=1)
    train_accuracy = np.sum(((h - y_train_orig) == 0) + 0) / m
    print("The train set's accuracy is: ", train_accuracy)

    # # test the test set
    # h = np.argmax(np.dot(X_test_flatten, theta), axis=1)
    # accuracy_test = np.sum(((h-Y_test_orig) == 0)+0) / Y_test_orig.shape[0]
    # print("The test set's accuracy is: ", accuracy_test)

    # save the hypothesis of our prediction
    z1_test, a1_test = first_layer_to_hidden_layer(x_test, w1, b1)
    z2_test, a2_test = hidden_layer_to_output_layer(z1_test, w2, b2)
    h_test = np.argmax(sigmoid(z2_test), axis=1).reshape(x_test.shape[0], 1)
    n_test = np.array(range(1, x_test.shape[0] + 1)).reshape(x_test.shape[0], 1)
    h_test = np.hstack((n_test, h_test))
    df_h_test = pd.DataFrame(h_test, columns=["ImageId", "Label"])
    df_h_test.to_csv("dataset/predictions.csv", index=False)

