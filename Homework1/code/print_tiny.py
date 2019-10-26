import numpy as np 

acc = np.load("Results/tiny_acc.npy")
runtimes = np.load("Results/tiny_time.npy")

print("accuracy", acc)
print("runtimes", runtimes)