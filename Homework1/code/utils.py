# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import timeit, time
import random
from scipy import spatial
from sklearn import neighbors, svm, cluster, preprocessing
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.svm import SVC, LinearSVC


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
        # We do not want to read some hidden directories such as .DS_Store. However, this causes our labels to be between 1 and 15
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
    neigh = neighbors.KNeighborsClassifier(n_neighbors=num_neighbors)
    neigh.fit(train_features, train_labels)
    predicted_categories = neigh.predict(test_features)
    return predicted_categories


def SVM_classifier(train_features, train_labels, test_features, is_linear, svm_lambda):
    classifiers = []
    predicted_categories = []
    for i in range(1, 16):
        if is_linear:
            clf = LinearSVC(C=svm_lambda)
        else:
            clf = SVC(C=svm_lambda, kernel="rbf", gamma="scale")
        new_labels = []
        for label in train_labels:
            if label == i:
                new_labels.append(1)
            else:
                new_labels.append(0)
        clf = clf.fit(train_features, new_labels)
        classifiers.append(clf)
    for test in test_features:
        # + 1 to account for the fact that our labels are between 1 and 15 whereas the index returned by argmax are between 0 and 14
        predicted_categories.append(np.asarray([c.decision_function([test])[0] for c in classifiers]).argmax() + 1)
    return predicted_categories


def imresize(input_image, target_size):
    resized_img = cv2.resize(input_image, tuple(target_size))
    output_image = cv2.normalize(
            src=resized_img, 
            dst=None, 
            alpha=-1, 
            beta=1, 
            norm_type=cv2.NORM_MINMAX, # specified to improve accuracy by ~2
            dtype=cv2.CV_32F, # specified such that output is float instead of int
    )
    return output_image


def reportAccuracy(true_labels, predicted_labels):
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
    # Limit the number of features to prevent memory error
    feature_size = 25
    all_descriptors = []

    for img in train_images:
        # Extract features of the images using specified feature type
        if (feature_type == "sift"):    
            feature = cv2.xfeatures2d.SIFT_create(nfeatures=feature_size)
        elif (feature_type == "surf"):
            feature = cv2.xfeatures2d.SURF_create()
        elif (feature_type == "orb"):
            feature = cv2.ORB_create(nfeatures=feature_size)
    
        _, des = feature.detectAndCompute(img,None)
        if (feature_type == "surf") and len(des) > feature_size:
            des = random.sample(list(des), feature_size)
        if (des is not None):
            for descriptor in des:
                all_descriptors.append(descriptor)

    vocabulary = [None for x in range(dict_size)]
    count = [0 for x in range(dict_size)] # array of size dict_size that counts the number of elements in each of vocabulary's bins

    # Build clusters from all the descriptors obtained and create a vocabulary from clustering
    if (clustering_type == "kmeans"):
        clustering = KMeans(n_clusters=dict_size, n_jobs=-1).fit(all_descriptors)
        vocabulary = clustering.cluster_centers_
    elif (clustering_type == "hierarchical"):
        # Default affinity for AgglomerativeClustering is euclidian.
        clustering = AgglomerativeClustering(n_clusters=dict_size).fit(all_descriptors)
        # Add all descriptors with the same label and store it in vocabulary
        labels = clustering.labels_
        for i in range(len(labels)):
            if vocabulary[labels[i]] is None:
                vocabulary[labels[i]] = all_descriptors[i].astype(float)
            else:
                vocabulary[labels[i]] = np.add(vocabulary[labels[i]], all_descriptors[i])
            count[labels[i]] += 1
        # Calculate the cluster centroids by dividing the number of descriptors per labels to its sum and normalize the result
        for i in range(dict_size):
            vocabulary[i] = vocabulary[i]/float(count[i])

    return vocabulary


def computeBow(image, vocabulary, feature_type):
    if feature_type == "sift":    
        feature = cv2.xfeatures2d.SIFT_create()
    elif feature_type == "surf":
        feature = cv2.xfeatures2d.SURF_create()
    elif feature_type == "orb":
        feature = cv2.ORB_create()
    
    _, descriptors = feature.detectAndCompute(image, None)

    Bow = [0] * len(vocabulary)
    try:
        for des in descriptors:
            Bow[np.array(np.linalg.norm(des-vocabulary, axis=1)).argmax()]+=1
    except TypeError as e:
        print(f"WARNING: {str(e)}. Ignoring this image and returning all-zeroes Bow.")
        return Bow

    # Normalize the Bow representation
    Bow = np.asarray(Bow)/float(len(descriptors))
    return Bow


def tinyImages(train_features, test_features, train_labels, test_labels):
    img_sizes = [8, 16, 32]
    num_neighbors = [1, 3, 6]
    classResult = []

    def resize_list_of_images(image_features, img_size):
    # A helper function that takes in a list of images and a target image size
    # and calls imresize function for all images in the list. 
    # It returns a list of resized images.
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
            
            classResult.append(round(accuracy, 2))
            classResult.append(round(runtime, 2))

    return classResult
    