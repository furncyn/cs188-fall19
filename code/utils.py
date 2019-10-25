import cv2
import numpy as np
import timeit
from Typing import Tuple, Dict
from sklearn import neighbors, svm, cluster

def imresize(input_image, target_size):
    # resizes the input image to a new image of size [target_size, target_size]. 
    # normalizes the output image to be zero-mean, and in the [-1, 1] range.
    input_img = cv2.imread(input_image)
    resized_img = cv2.resize(input_img, tuple(target_size))
    output_image = cv2.normalize(resized_img, None, alpha=-1, beta=1)

    return output_image

def reportAccuracy(true_labels, predicted_labels, label_dict):
    # generates and returns the accuracy of a model
    # true_labels is a n x 1 cell array, where each entry is an integer
    # and n is the size of the testing set.
    # predicted_labels is a n x 1 cell array, where each entry is an 
    # integer, and n is the size of the testing set. these labels 
    # were produced by your system
    # label_dict is a 15x1 cell array where each entry is a string
    # containing the name of that category
    # accuracy is a scalar, defined in the spec (in %)
    
    # Assume that labels with different lenghts are allowed
    
    num_correct_predictions = 0
    num_predictions = len(true_labels)
    
    for i in range(len(predicted_labels)):
        try:
            if true_labels[i] == predicted_labels[i]:
                num_correct_predictions += 1  
        except:
            continue
    
    accuracy = num_correct_predictions/num_predictions * 100
    return accuracy

def buildDict(train_images, dict_size, feature_type, clustering_type):
    # this function will sample descriptors from the training images,
    # cluster them, and then return the cluster centers.

    # train_images is a n x 1 array of images
    # dict_size is the size of the vocabulary,
    # feature_type is a string specifying the type of feature that we are interested in.
    # Valid values are "sift", "surf" and "orb"
    # clustering_type is one of "kmeans" or "hierarchical"

    # the output 'vocabulary' should be dict_size x d, where d is the 
    # dimention of the feature. each row is a cluster centroid / visual word.

    #Write a function buildDict that samples the speciﬁed features from the training images 
    # (SIFT, SURF or ORB), and outputs a vocabulary of the speciﬁed size, by clustering them 
    # using either K-means or hierarchical agglomerative clustering. Cluster centroids will 
    # be the words in your vocabulary. Use an Euclidean metric to compute distances. 
    # (Hint: OpenCV has implementations for SIFT/SURF/ORB feature detection. 
    # Sklean has implementations for Kmeans and Hierarchical Agglomerative Clustering)
    if (clustering_type == "sift"):
        #Insert sift code here
    else if (clustering_type == "surf"):
        #Insert surf code here
    else if (clustering_type == "orb"):
        #insert orb code here
    else:
        #??
    # return a list
    return vocabulary

def computeBow(image, vocabulary, feature_type):
    # extracts features from the image, and returns a BOW representation using a vocabulary
    # image is 2D array
    # vocabulary is an array of size dict_size x d
    # feature type is a string (from "sift", "surf", "orb") specifying the feature
    # used to create the vocabulary

    # BOW is the new image representation, a normalized histogram
    return Bow

def tinyImages(train_features, test_features, train_labels, test_labels, label_dict):
    # train_features is a nx1 array of images
    # test_features is a nx1 array of images
    # train_labels is a nx1 array of integers, containing the label values
    # test_labels is a nx1 array of integers, containing the label values
    # label_dict is a 15x1 array of strings, containing the names of the labels
    # classResult is a 18x1 array, containing accuracies and runtimes
    return classResult
    