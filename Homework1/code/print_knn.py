import numpy as np

a = np.load("Results/knn_accuracies.npy")
print("accuracies", a)
r = np.load("Results/knn_runtimes.npy")
print("runtimes", r)

