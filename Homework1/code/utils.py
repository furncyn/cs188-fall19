# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import timeit, time
from sklearn import neighbors, svm, cluster, preprocessing
from sklearn.cluster import AgglomerativeClustering, KMeans


def load_data():
    test_path = '../data/test/'
    train_path = '../data/train/'
    
    train_classes = sorted([dirname for dirname in os.listdir(train_path)], key=lambda s: s.upper())
    test_classes = sorted([dirname for dirname in os.listdir(test_path)], key=lambda s: s.upper())
    train_labels = []
    test_labels = []
    train_images = []
    test_images = []
    for i, label in enumerate(train_classes):
        # We do not want to read some hidden directories such as .DS_Store
        if label.startswith("."):
            continue
        for filename in os.listdir(train_path + label + '/'):
            image = cv2.imread(train_path + label + '/' + filename, cv2.IMREAD_GRAYSCALE)
            train_images.append(image)
            train_labels.append(i)
    for i, label in enumerate(test_classes):
        if label.startswith("."):
            continue
        for filename in os.listdir(test_path + label + '/'):
            image = cv2.imread(test_path + label + '/' + filename, cv2.IMREAD_GRAYSCALE)
            test_images.append(image)
            test_labels.append(i)
            
    return train_images, test_images, train_labels, test_labels


def KNN_classifier(train_features, train_labels, test_features, num_neighbors):
    # outputs labels for all testing images

    # train_features is an N x d matrix, where d is the dimensionality of the
    # feature representation and N is the number of training features.
    # train_labels is an N x 1 array, where each entry is an integer
    # indicating the ground truth category for each training image.
    # test_features is an M x d array, where d is the dimensionality of the
    # feature representation and M is the number of testing features.
    # num_neighbors is the number of neighbors for the KNN classifier

    # predicted_categories is an M x 1 array, where each entry is an integer
    # indicating the predicted category for each test image.
    neigh = neighbors.KNeighborsClassifier(n_neighbors=num_neighbors)
    neigh.fit(train_features, train_labels)
    predicted_categories = neigh.predict(test_features)
    return predicted_categories


def SVM_classifier(train_features, train_labels, test_features, is_linear, svm_lambda):
    # this function will train a linear svm for every category (i.e. one vs all)
    # and then use the learned linear classifiers to predict the category of
    # every test image. every test feature will be evaluated with all 15 svms
    # and the most confident svm will "win". confidence, or distance from the
    # margin, is w*x + b where '*' is the inner product or dot product and w and
    # b are the learned hyperplane parameters.

    # train_features is an N x d matrix, where d is the dimensionality of
    # the feature representation and N the number of training features.
    # train_labels is an N x 1 array, where each entry is an integer 
    # indicating the ground truth category for each training image.
    # test_features is an M x d matrix, where d is the dimensionality of the
    # feature representation and M is the number of testing features.
    # is_linear is a boolean. If true, you will train linear SVMs. Otherwise, you 
    # will use SVMs with a Radial Basis Function (RBF) Kernel.
    # svm_lambda is a scalar, the value of the regularizer for the SVMs

    # predicted_categories is an M x 1 array, where each entry is an integer
    # indicating the predicted category for each test feature.
    return predicted_categories


def imresize(input_image, target_size):
    # resizes the input image, represented as a 2D array, to a new image of size [target_size, target_size]. 
    # Normalizes the output image to be zero-mean, and in the [-1, 1] range.
    resized_img = cv2.resize(input_image, tuple(target_size))
    output_image = cv2.normalize(resized_img, None, alpha=-1, beta=1)
    return output_image


def reportAccuracy(true_labels, predicted_labels):
    # generates and returns the accuracy of a model

    # true_labels is a N x 1 list, where each entry is an integer
    # and N is the size of the testing set.
    # predicted_labels is a N x 1 list, where each entry is an 
    # integer, and N is the size of the testing set. These labels 
    # were produced by your system.

    # Assume that labels with different lenghts are allowed
    num_correct_predictions = 0
    num_predictions = len(true_labels)

    for i in range(len(predicted_labels)):
        try:
            if true_labels[i] == predicted_labels[i]:
                num_correct_predictions += 1  
        except:
            continue
    accuracy = (num_correct_predictions/num_predictions) * 100
    return accuracy


def buildDict(train_images, dict_size, feature_type, clustering_type):
    # this function will sample descriptors from the training images,
    # cluster them, and then return the cluster centers.

    # train_images is a list of N images, represented as 2D arrays
    # dict_size is the size of the vocabulary,
    # feature_type is a string specifying the type of feature that we are interested in.
    # Valid values are "sift", "surf" and "orb"
    # clustering_type is one of "kmeans" or "hierarchical"

    # the output 'vocabulary' should be a list of length dict_size, with elements of size d, where d is the 
    # dimention of the feature. each row is a cluster centroid / visual word.

    # NOTE: Should you run out of memory or have performance issues, feel free to limit the 
    # number of descriptors you store per image.

    #Write a function buildDict that samples the speciﬁed features from the training images 
    # (SIFT, SURF or ORB), and outputs a vocabulary of the speciﬁed size, by clustering them 
    # using either K-means or hierarchical agglomerative clustering. Cluster centroids will 
    # be the words in your vocabulary. Use an Euclidean metric to compute distances. 
    # (Hint: OpenCV has implementations for SIFT/SURF/ORB feature detection. 
    # Sklean has implementations for Kmeans and Hierarchical Agglomerative Clustering)
    #sample descriptors from ALL training images
    vocabulary = np.zeros(dict_size)
    all_descriptors = []
    if (feature_type == "sift"):    
        feature = cv2.xfeatures2d.SIFT_create()
    elif (feature_type == "surf"):
        feature = cv2.xfeatures2d.SURF_create()
    elif (feature_type == "orb"):
        feature = cv2.ORB_create()
    for img in train_images:    
        _,des = feature.detectAndCompute(img,None)
        if (des is not None):
            for descriptor in des:
                all_descriptors.append(descriptor)
    print("descriptors calculated")
    # Built a list of descriptors, convert to numpy array
    if (clustering_type == "kmeans"):
        clustering = KMeans(n_clusters=dict_size, n_jobs=-1).fit_predict(all_descriptors)
    elif (clustering_type == "hierarchical"):
        # nparray = np.asarray(all_descriptors)
        clustering = AgglomerativeClustering(n_clusters=dict_size).fit(all_descriptors)
    for c in clustering:
        vocabulary[c] += 1
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


def tinyImages(train_features, test_features, train_labels, test_labels):
    # Resizes training images and flattens them to train a KNN classifier using the training labels
    # Classifies the resized and flattened testing images using the trained classifier
    # Returns the accuracy of the system, and the overall runtime (including resizing and classification)
    # Does so for 8x8, 16x16, and 32x32 images, with 1, 3 and 6 neighbors

    # train_features is a list of N images, represented as 2D arrays
    # test_features is a list of M images, represented as 2D arrays
    # train_labels is a list of N integers, containing the label values for the train set
    # test_labels is a list of M integers, containing the label values for the test set

    # classResult is a 18x1 array, containing accuracies and runtimes, in the following order:
    # accuracies and runtimes for 8x8 scales, 16x16 scales, 32x32 scales
    # [8x8 scale 1 neighbor accuracy, 8x8 scale 1 neighbor runtime, 8x8 scale 3 neighbor accuracy, 
    # 8x8 scale 3 neighbor runtime, ...]
    # Accuracies are a percentage, runtimes are in seconds
    
    img_sizes = [8, 16, 32]
    num_neighbors = [1, 3, 6]
    classResult = []

    def resize_list_of_images(image_features, img_size):
        resized_features = []
        for img in image_features:
            resized_features.append(imresize(img, img_size).flatten())
        return resized_features

    for i in range(len(img_sizes)):
        for j in range(len(num_neighbors)):
            img_size = [img_sizes[i], img_sizes[i]]
            start_time = time.time()
            predicted_labels = KNN_classifier(
                train_features=resize_list_of_images(train_features, img_size),
                train_labels=train_labels,
                test_features=resize_list_of_images(test_features, img_size),
                num_neighbors=num_neighbors[j],
            )
            runtime = time.time() - start_time
            accuracy = reportAccuracy(test_labels, predicted_labels)
            
            classResult.append(accuracy)
            classResult.append(runtime)
    return classResult
    