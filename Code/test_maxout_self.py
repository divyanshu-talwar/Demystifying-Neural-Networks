import argparse
from sklearn.model_selection import KFold
from sklearn.datasets.mldata import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import h5py
import numpy as np
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--data", type = str  )
parser.add_argument("--weights_save_dir", type = str)
parser.add_argument("--bias_save_dir", type = str)
parser.add_argument("--hidden_layer", nargs="+", type = int )
parser.add_argument("--softmax_bool", type = bool)

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
wfilename = args.weights_save_dir
bfilename = args.bias_save_dir
layers = args.hidden_layer
softmax_bool = args.softmax_bool

output_layer_neurons = 10
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
# do k-fold instead.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
features = x[0].shape[0]
layers = [features] + layers + [output_layer_neurons]
linear_fx = 3

weights_file = open(wfilename, "rb")
bias_file = open(bfilename, "rb")
weights = pickle.load(weights_file)
bias = pickle.load(bias_file)

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
accuracy = accuracy_score(y_test, y_pred)*100
print accuracy