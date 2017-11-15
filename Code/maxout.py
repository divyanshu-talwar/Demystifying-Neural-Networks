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

def maxout(x):
	t = np.amax(x, axis = 0)
	u = np.argmax(x, axis = 0)
	return t,u

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
learning_rate = 1e-04
linear_fx = 3
epochs_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
epochs_cumulative = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
mini_batch = 50
layers = [features] + layers + [output_layer_neurons]

weights = [np.random.normal(0,0.01,(linear_fx, ) + a) for a in zip(layers[1::1], layers[0::1])]
bias = [np.random.normal(0,0.01,(linear_fx, i, 1)) for i in layers[1:]]
activation = []
z = []
update_indices = []
for k in range(mini_batch):
	activation.append(np.array([np.zeros((i, 1)) for i in layers]))
	update_indices.append(np.array([np.zeros((i, 1), dtype = int) for i in layers[1:]]))
	z.append([np.zeros((linear_fx, i, 1)) for i in layers[1:]])
activation = np.array(activation)

temp_data = zip(x_train, y_train)
dataset = [temp_data[i:i+mini_batch] for i in range(0, len(x_train), mini_batch)]

print "started"

accuracy = []
for epochs in epochs_list:
	for m in xrange(epochs):
		np.random.shuffle(temp_data)
		dataset = [temp_data[i:i+mini_batch] for i in range(0, len(x_train), mini_batch)]
		for batch in dataset:
			changed_weights = [np.zeros((linear_fx, ) + a) for a in zip(layers[1::1], layers[0::1])]
			changed_bias = [np.zeros((linear_fx, i, 1)) for i in layers[1:]]
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
						for blah in range(linear_fx):
							z[index][i - 1][blah] = np.dot(weights[i - 1][blah], activation[index][i - 1]) + bias[i - 1][blah]
						if( not(softmax_bool) or i < len(layers) - 1):
							activation[index][i], update_indices[index][i - 1] = maxout(z[index][i - 1])
						else:
							activation[index][i] = softmax(maxout(z[index][i - 1])[0])
				# backprop
				delta = cross_entropy_derivative(activation[index][-1], e)
				changed_bias[-1] += delta
				changed_weights[-1] += np.dot(delta, activation[index][-2].transpose())
				for l in xrange(2, len(layers)):
					next_delta = []
					for neurons, max_index in enumerate(update_indices[index][-l].flatten()):
						delta_neuron = np.dot(weights[-l+1][max_index].transpose(), delta)
						next_delta.append(delta_neuron)
						changed_bias[-l][max_index] += delta_neuron
						changed_weights[-l][max_index] += np.dot(delta_neuron, activation[index][-l-1].transpose())
					delta = np.array(delta_neuron)
			# Gradient Descent
			for l in xrange(len(layers) - 1):
				for max_index in xrange(linear_fx):
					weights[l][max_index] = weights[l][max_index] - (learning_rate/mini_batch)*changed_weights[l][max_index]
					bias[l][max_index] = bias[l][max_index] - (learning_rate/mini_batch)*changed_bias[l][max_index]
		print "epoch done " + str(m)
	print "training done"
	y_pred = []
	temp = np.array([np.zeros((i, 1)) for i in layers])
	temp_z = [np.zeros((linear_fx, i, 1)) for i in layers[1:]]
	for q in x_test:
		# forward pass
		for i in range(len(layers)):
			if i == 0: # set first index of activations as the input feature vectors
				temp[i] = np.reshape(q, (q.shape[0], 1))
			else:
				for blah in range(linear_fx):
					temp_z[i - 1][blah] = np.dot(weights[i - 1][blah], temp[i - 1]) + bias[i - 1][blah]
				if( not(softmax_bool) or i < len(layers) - 1):
					temp[i], _ = maxout(temp_z[i - 1])
				else:
					temp[i] = softmax(maxout(temp_z[i - 1])[0])
		y_pred.append(np.argmax(temp[-1]))
	aha = accuracy_score(y_test, y_pred)*100
	accuracy.append(aha)
	print aha
	# file = open("weights_large_sigmoid_self" + str(m) +".pkl", "wb")
	# file1 = open("bias_large_sigmoid_self" + str(m) +".pkl", "wb")
	# pickle.dump(weights, file)
	# pickle.dump(bias, file1)
	# file.close()
	# file1.close()
print epochs_cumulative
print accuracy
# plt.plot(epochs_cumulative, accuracy)
# plt.plot(epochs_cumulative, accuracy, 'go')
# plt.show()