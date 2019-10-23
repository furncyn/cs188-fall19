import argparse
import numpy as np


feature_dict = {
    "si": "sift",
    "su": "surf",
    "o": "orb",
}

clustering_dict = {
    "k": "kmeans",
    "h": "hierarchical",
}

dict_sizes = [20, 50]

parser = argparse.ArgumentParser(description="Print numpy array.")
parser.add_argument("feature_type", type=str, default="si", help="Input si for sift, su for surf, or o for orb.")
parser.add_argument("clustering_type", type=str, default="k", help="Input k for kmeans or h for hierarchical")
parser.add_argument("dict_size", type=int, default=20, help="Input 20 or 50")
parser.add_argument("--all", type=bool, default=False, help="Set to true to print all combinations.")

args = parser.parse_args()

feature = feature_dict[args.feature_type]
cluster = clustering_dict[args.clustering_type]
dict_size = args.dict_size


def print_nparr(feature, cluster, dict_size):
    print(f"{feature}, {cluster}, {dict_size}")
    path = f"Results/voc_{feature}_{cluster}_{dict_size}.npy"
    nparr = np.load(path)
    print(nparr.size)
    # print(nparr)


if args.all:
    for feature in feature_dict.values():
        for cluster in clustering_dict.values():
            for dict_size in dict_sizes:
                print_nparr(feature, cluster, dict_size)
else:
    print_nparr(feature, cluster, dict_size)
