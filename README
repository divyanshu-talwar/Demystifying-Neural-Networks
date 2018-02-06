# Understanding Neural Networks

## About
A self implementation of `Forward-Pass` and `Backpropagation` - used to train a neural network - to get a better understanding of the same.

## Approach
The self implemented models were tested by comparing their performance on - MNIST small (7 and 9) and MNIST complete dataset - with the SKlearn's MLP classifier models.

* The activation functions for which the self implementation is done include the following : 
	* Sigmoid (with and without the softmax as the last layer in the ANN)
	* ReLu (with and without the softmax as the last layer in the ANN)
	* MaxOut (with and without the softmax as the last layer in the ANN)

## Dependencies
* For Python code:
    * python 2.7
    * numpy
    * scikit-learn
    * matplotlib
    * h5py
    * pickle

## Contents
* `Report.pdf` - contains the results, observations and conclusions of the experiment.
* `plot/` - contains all the plots generated (Epochs vs Accuracy)
* `dataset_partA.h5` - contains the MNIST small dataset (7 and 9)
* `Code/` - contains the self implemented scripts, test scripts (comparision with SKlearn's MLP classifier) and the result plotting script.
* `Code/pickled-weights` - contains the saved models generated (for both self implemented and SKlearn MLP classifier models) for direct verification of results.
* `Results.txt` - Results

## Execution:
1. For training on the subset (MNIST small)
	* `python <name_self_implemented_activation_function.py> --hidden_layer 100 50  --data ../dataset_partA.h5`
2. For training on the subset (MNIST small)
	* `python <name_self_implemented_activation_function.py> --hidden_layer 100 50  --data none`

3. For testing pretrained model
	* Sigmoid (MNIST subset)-
`python test_sigmoid_self.py --data ../dataset_partA.h5 --weights_save_dir ./pickled-weights/weights_small_sigmoid_self.pkl --bias_save_dir ./pickled-weights/bias_small_sigmoid_self.pkl  --hidden_layer 100 50 --softmax_bool True`

	* Sigmoid (MNIST)-
`python test_sigmoid_self.py --data none --weights_save_dir ./pickled-weights/weights_large_sigmoid_self.pkl --bias_save_dir ./pickled-weights/bias_large_sigmoid_self.pkl  --hidden_layer 100 50 --softmax_bool True`

	* ReLU (MNIST subset)-
`python test_relu_self.py --data ../dataset_partA.h5 --weights_save_dir ./pickled-weights/weights_small_relu_self.pkl --bias_save_dir ./pickled-weights/bias_small_relu_self.pkl  --hidden_layer 100 50 --softmax_bool True`

	* ReLU (MNIST)-
`python test_maxout_self.py --data none --weights_save_dir ./pickled-weights/weights_large_maxout_self.pkl --bias_save_dir ./pickled-weights/bias_large_maxout_self.pkl  --hidden_layer 100 50 --softmax_bool True`

	* MaxOut (MNIST subset) -
`python test_maxout_self.py --data ../dataset_partA.h5 --weights_save_dir ./pickled-weights/weights_small_maxout_self.pkl --bias_save_dir ./pickled-weights/bias_small_maxout_self.pkl  --hidden_layer 100 50 --softmax_bool True`

	* MaxOut (MNIST) -
`python test_maxout_self.py --data none --weights_save_dir ./pickled-weights/weights_large_maxout_self.pkl --bias_save_dir ./pickled-weights/bias_large_maxout_self.pkl  --hidden_layer 100 50 --softmax_bool True`

## Author
* Divyanshu Talwar
