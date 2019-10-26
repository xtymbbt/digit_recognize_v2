from functions import *
from keras.utils import to_categorical

# matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# load_ext autoreload
# autoreload 2

np.random.seed(1)

# load the data
csv_train_data = pd.read_csv('dataset/train.csv')  # load train data
csv_test_data = pd.read_csv('dataset/test.csv')  # load test data
# print(csv_data.shape)  # (42000, 785)
csv_np = np.array(csv_train_data)
csv_np_test = np.array(csv_test_data)
# print(csv_np.shape)  # (42000,785)
Y_train_orig = csv_np[:, 0]
# Y_test_orig = csv_np[:, 0]
# print(train_Y.shape)  # (42000,1)
X_train_flatten = csv_np[:, 1:785]
X_test_flatten = csv_np_test[:, :]
# print(train_X_flatten.shape)  # (42000,784)
# Standardize data to have feature values between 0 and 1.

X_train = X_train_flatten/255.
X_test = X_test_flatten/255.

Y_train = to_categorical(Y_train_orig)
# Y_test = to_categorical(Y_test_orig)

X_train_orig = np.reshape(X_train_flatten, (42000, 28, 28))
X_test_orig = np.reshape(X_test_flatten, (28000, 28, 28))

# index = 10
# plt.imshow(X_train_orig[index])
# plt.show()


input_layer_size = 784  # 20x20 Input Images of Digits
num_labels = 10          # 10 labels, from 1 to 10
# (note that we have mapped "0" to label 10)


model(X_train, Y_train, X_test, Y_train)


# Test case for lrCostFunction
# print('\nTesting lrCostFunction() with regularization')
#
# theta_t = np.array([[-2], [-1], [1], [2]])
# ones = np.ones((5, 1))
# arange = np.arange(1, 16).reshape(5, 3) / 10
# X_t = np.hstack((ones, arange))
# y_t = (np.array([[1], [0], [1], [0], [1]]) >= 0.5) + 0
# lambda_t = 3
# J, grad = cost_function2d(theta_t, X_t, y_t, lambda_t)
#
# print('\nCost: %f\n', J)
# print('Expected cost: 2.534819\n')
# print('Gradients:\n')
# print(' %f \n', grad)
# print('Expected gradients:\n')
# print(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n')

# # initialize theta as a random matrix
# theta = np.zeros((input_layer_size, num_labels))
#
# # initialize lambda
# lambda_ = 0.1
# # compute cost and gradient using the 2D cost function
# J, grad = cost_function2d(theta, X_train, Y_train, lambda_)
# print("Epoch 0: J = ", J)
# print("grad = ", grad)
# # test the train set
# h = np.argmax(np.dot(X_train_flatten, theta), axis=1)
# accuracy_train = np.sum(((h-Y_train_orig) == 0)+0) / Y_train_orig.shape[0]
# print("The train set's accuracy is: ", accuracy_train)
#
# # initialize delta_J as a learning rate
# delta_J = 0.01
#
# # update theta automatically in a for loop
# for i in range(1000):
#     theta = theta - (delta_J * np.linalg.pinv(grad)).T
#     J_prev = J
#     J, grad = cost_function2d(theta, X_train, Y_train, lambda_)
#     # if J > J_prev:
#     #     delta_J = delta_J/10
#     print("Epoch ", (i+1), ":")
#     print("J = ", J)
#     print("grad = ", grad)
#     # test the train set
#     h = np.argmax(np.dot(X_train_flatten, theta), axis=1)
#     accuracy_train = np.sum(((h - Y_train_orig) == 0) + 0) / Y_train_orig.shape[0]
#     print("The train set's accuracy is: ", accuracy_train)
#
# # save theta
# df_theta = pd.DataFrame(theta)
# df_theta.to_csv("dataset/theta.csv")
#
# # test the train set
# h = np.argmax(np.dot(X_train_flatten, theta), axis=1)
# accuracy_train = np.sum(((h-Y_train_orig) == 0)+0) / Y_train_orig.shape[0]
# print("The train set's accuracy is: ", accuracy_train)
#
# # # test the test set
# # h = np.argmax(np.dot(X_test_flatten, theta), axis=1)
# # accuracy_test = np.sum(((h-Y_test_orig) == 0)+0) / Y_test_orig.shape[0]
# # print("The test set's accuracy is: ", accuracy_test)
#
# # save the hypothesis of our prediction
# h = np.argmax(np.dot(X_test_flatten, theta), axis=1).reshape(X_test_flatten.shape[0], 1)
# n = np.array(range(1, X_test_flatten.shape[0]+1)).reshape(X_test_flatten.shape[0], 1)
# h = np.hstack((n, h))
# df_h = pd.DataFrame(h, columns=["ImageId", "Label"])
# df_h.to_csv("dataset/predictions.csv", index=False)

