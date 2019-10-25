import numpy as np


img_set = ["train", "test"]

for s in img_set:
    for i in range(12):
        a = np.load(f"Results/bow_{s}_{i}.npy")
        print(f"bow_{s}_{i}")
        print(a)