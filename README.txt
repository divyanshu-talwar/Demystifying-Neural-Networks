Accuracies:
1. mnist subset
model										-accuracy(test)
_________________________________________________________________________
trained_sigmoid 							- 99.6070
trained_relu_softmax			 			- 99.8421
trained_maxout_softmax						- 97.5442
sklearn 									- 98.5966
2. entire mnist
model										-my_accuracy(test),sklearn_accuracy(test)
_________________________________________________________________________
trained_sigmoid_softmax						- 96.3771
trained_relu_softmax			 			- 99.5214
trained_maxout_softmax						- 94.9428
sklearn 									- 94.9942

Commands:
1. For training on subset
	python maxout.py --hidden_layer 100 50  --data ../dataset_partA.h5
2. For training on entire subset
	python maxout.py --hidden_layer 100 50  --data none

3. For testing pretrained model
	Sigmoid (MNIST subset)-
python test_sigmoid_self.py --data ../dataset_partA.h5 --weights_save_dir ./pickled-weights/weights_small_sigmoid_self.pkl --bias_save_dir ./pickled-weights/bias_small_sigmoid_self.pkl  --hidden_layer 100 50 --softmax_bool True

	Sigmoid (MNIST)-
python test_sigmoid_self.py --data none --weights_save_dir ./pickled-weights/weights_large_sigmoid_self.pkl --bias_save_dir ./pickled-weights/bias_large_sigmoid_self.pkl  --hidden_layer 100 50 --softmax_bool True

	ReLU (MNIST subset)-
python test_relu_self.py --data ../dataset_partA.h5 --weights_save_dir ./pickled-weights/weights_small_relu_self.pkl --bias_save_dir ./pickled-weights/bias_small_relu_self.pkl  --hidden_layer 100 50 --softmax_bool True

	ReLU (MNIST)-
python test_maxout_self.py --data none --weights_save_dir ./pickled-weights/weights_large_maxout_self.pkl --bias_save_dir ./pickled-weights/bias_large_maxout_self.pkl  --hidden_layer 100 50 --softmax_bool True

	MaxOut (MNIST subset) -
python test_maxout_self.py --data ../dataset_partA.h5 --weights_save_dir ./pickled-weights/weights_small_maxout_self.pkl --bias_save_dir ./pickled-weights/bias_small_maxout_self.pkl  --hidden_layer 100 50 --softmax_bool True

	MaxOut (MNIST) -
python test_maxout_self.py --data none --weights_save_dir ./pickled-weights/weights_large_maxout_self.pkl --bias_save_dir ./pickled-weights/bias_large_maxout_self.pkl  --hidden_layer 100 50 --softmax_bool True