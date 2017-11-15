import argparse
from sklearn.model_selection import KFold
from sklearn.datasets.mldata import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--data", type = str  )
parser.add_argument("--weights_save_dir", type = str)
parser.add_argument("--hidden_layer", nargs="+", type = int )

# Activation functions and their derivatives
def softmax(x):
	y = np.exp(x)
	y_sum = np.sum(y)
	return y/y_sum

def cross_entropy_derivative(a, y):
	# return ((a - y)/(a*(1.0 - a)))
	return (a - y)

def sigmoid(x):
	return (1.0 / (1.0 + np.exp(-1.0 * x)))

def derivative_sigmoid(x):
	t = sigmoid(x)
	return t * (1.0 - t)

args = parser.parse_args()
filename = args.data
layers = args.hidden_layer
output_layer_neurons = 10
softmax_bool = False

# Reading the dataset
if(filename == "none"):
	mnist = fetch_mldata('MNIST original')
	x = mnist.data
	y = mnist.target
	softmax_bool = True

else:
	hf = h5py.File(filename, 'r')	# the dataset has only 2 classes - 7 and 9
	# for name in hf:
	# 	print name
	x = hf['X'][:]
	y = hf['Y'][:]
	x = x.reshape((14251, 28*28))
	output_layer_neurons = 2	
	y[y==7] = 0
	y[y==9] = 1
	# softmax_bool = True

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
features = x[0].shape[0]

# Initializing the neural network
learning_rate = 0.01 #0.01 for 10 epochs
epochs_list = [10, 10, 20, 10, 30, 20, 100, 100, 100, 100]
# [10, 20 , 40, 50, 80, 100, 200, 300, 400, 500]
epochs_cumulative = [10, 20 , 40, 50, 80, 100, 200, 300, 400, 500]
mini_batch = 50
layers = [features] + layers + [output_layer_neurons]

weights = np.array([np.random.normal(0,0.01,a) for a in zip(layers[1::1], layers[0::1])])
bias = np.array([np.random.normal(0,0.01,(i, 1)) for i in layers[1:]])
activation = []
z = []
for k in range(mini_batch):
	activation.append(np.array([np.zeros((i, 1)) for i in layers]))
	z.append(np.array([np.zeros((i, 1)) for i in layers[1:]]))
activation = np.array(activation)
z = np.array(z)

temp_data = zip(x_train, y_train)
dataset = [temp_data[i:i+mini_batch] for i in range(0, len(x_train), mini_batch)]

print "started"

accuracy = []
for epochs in epochs_list:
	for m in xrange(epochs):
		np.random.shuffle(temp_data)
		dataset = [temp_data[i:i+mini_batch] for i in 	range(0, len(x_train), mini_batch)]
		for batch in dataset:
			changed_weights = np.array([np.zeros(a) for a in zip(layers[1::1], layers[0::1])])
			changed_bias = np.array([np.zeros((i, 1)) for i in layers[1:]])
			for index,d in enumerate(batch):
				# forward pass
				w = d[0]
				w = np.reshape(w, (w.shape[0], 1))
				e = (np.zeros((output_layer_neurons, 1)))
				e[int(d[1])] = 1
				for i in range(len(layers)):
					if i == 0: # set first index of activations as the input feature vectors
						activation[index][i] = w				
					else:
						z[index][i - 1] = np.reshape(np.dot(weights[i - 1], activation[index][i - 1]), (np.dot(weights[i - 1], activation[index][i - 1]).shape[0], 1)) + bias[i - 1]
						if( not(softmax_bool) or i < len(layers) - 1):
							activation[index][i] = sigmoid(z[index][i - 1])
						else:
							activation[index][i] = softmax(z[index][i - 1])
				# backprop
				delta = cross_entropy_derivative(activation[index][-1], e)
				changed_bias[-1] += delta
				changed_weights[-1] += np.dot(delta, activation[index][-2].transpose())
				for l in xrange(2, len(layers)):
					delta = np.multiply(np.dot(weights[-l+1].transpose(), delta), derivative_sigmoid(z[index][-l]))
					changed_bias[-l] += delta
					changed_weights[-l] += np.dot(delta, activation[index][-l-1].transpose())
			# Gradient Descent
			weights = np.array([wt - (learning_rate/mini_batch)*dwt for wt, dwt in zip(weights, changed_weights)])
			bias = np.array([bi - (learning_rate/mini_batch)*dbi for bi, dbi in zip(bias, changed_bias)])
		print "epoch done " + str(m)
	print "training done"
	y_pred = []
	temp = np.array([np.zeros((i, 1)) for i in layers])
	for q in x_test:
		# forward pass
		for i in range(len(layers)):
			if i == 0: # set first index of activations as the input feature vectors
				temp[i] = np.reshape(q, (q.shape[0], 1))
			else:
				if( not(softmax_bool) or i < len(layers) - 1):
					temp[i] = sigmoid(np.dot(weights[i - 1], temp[i - 1]) + bias[i - 1])
				else:
					temp[i] = softmax(np.dot(weights[i - 1], temp[i - 1]) + bias[i - 1])
		y_pred.append(np.argmax(temp[-1]))
	aha = accuracy_score(y_test, y_pred)*100
	accuracy.append(aha)
	print aha
# file = open("weights_large_sigmoid_self.pkl", "wb")
# file1 = open("bias_large_sigmoid_self.pkl", "wb")
# pickle.dump(weights, file)
# pickle.dump(bias, file1)
# file.close()
# file1.close()
# print epochs_cumulative
print accuracy
# plt.plot(epochs_cumulative, accuracy)
# plt.plot(epochs_cumulative, accuracy, 'go')
# plt.show()

