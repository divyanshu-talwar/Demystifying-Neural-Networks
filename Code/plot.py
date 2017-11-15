import matplotlib.pyplot as plt

accuracy = [88.382857142857148, 92.548571428571421, 93.879999999999995, 93.885714285714286, 94.034285714285716,  94.33142857142856, 94.537142857142854,  94.994285714285709]
epochs_cumulative = [1, 10, 20, 40, 50, 100, 200, 500]
name = 'Sigmoid (sklearn) Large Dataset'
plt.plot(epochs_cumulative, accuracy)
plt.plot(epochs_cumulative, accuracy, 'go')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title(name)
plt.savefig(name + ".png")
plt.show()