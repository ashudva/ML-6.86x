from string import punctuation, digits
import numpy as np
import random
from math import sqrt


def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices


def hinge_loss_single(feature_vector, label, theta, theta_0):
    z = label*(np.dot(feature_vector,theta) + theta_0)
    return max(0, 1 - z)


def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    loss = 0
    for i in range(len(feature_matrix)):
        loss += hinge_loss_single(feature_matrix[i], labels[i], theta, theta_0)
    return loss / len(labels)


def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.

    We need to make sure that if the output of the perceptron is  0 , the weights are still updated, hence the check for large inequality.
    In fact, because of numerical instabilities, it is preferable to identify  0  with a small range  [−ε,ε] .

    """
    if label * (np.dot(current_theta, feature_vector) + current_theta_0) <= 1e-7:
        current_theta += label * feature_vector
        current_theta_0 += label
    return (current_theta, current_theta_0)


def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.
    """
    (nsamples, nfeatures) = feature_matrix.shape
    theta = np.zeros(nfeatures)
    theta_0 = 0.0
    for t in range(T):
        for i in get_order(nsamples):
            theta, theta_0 = perceptron_single_step_update(
                feature_matrix[i], labels[i], theta, theta_0)
    return (theta, theta_0)


def average_perceptron(feature_matrix, labels, T):

    new_theta = np.zeros((feature_matrix.shape[1],))
    new_theta_0 = 0
    sum_of_theta = new_theta
    sum_of_theta_0 = new_theta_0
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            updated_data = perceptron_single_step_update(feature_matrix[i],labels[i],new_theta,new_theta_0)
            new_theta = updated_data[0]
            new_theta_0 = updated_data[1]
            sum_of_theta += new_theta
            sum_of_theta_0 += new_theta_0
            pass
    return sum_of_theta/(feature_matrix.shape[0]*T), sum_of_theta_0/(feature_matrix.shape[0]*T)


def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        current_theta,
        current_theta_0):

    for i in range(3):
        new_theta = current_theta
        new_theta_0 = current_theta_0
        if label * (np.dot(current_theta,feature_vector) + current_theta_0) <= 1:
            new_theta = new_theta*(1 - eta*L) + eta*label*feature_vector
            new_theta_0 = new_theta_0 + eta*label
        else:
            new_theta = (1 - eta*L)*new_theta
            new_theta_0 = new_theta_0
    return new_theta,new_theta_0


def pegasos(feature_matrix, labels, T, L):

    new_theta = np.zeros((feature_matrix.shape[1],))
    new_theta_0 = 0
    update_count = 0
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            update_count += 1
            eta = 1/sqrt(update_count)
            updated_data = pegasos_single_step_update(feature_matrix[i],labels[i],L,eta,new_theta,new_theta_0)
            new_theta = updated_data[0]
            new_theta_0 = updated_data[1]
    return new_theta, new_theta_0


def classify(feature_matrix, theta, theta_0):

    labels = []
    for feature_vector in feature_matrix:
        # Data points exactly on the classifier are predicted as -1
        if np.dot(theta,feature_vector) + theta_0 <= 0:
            labels.append(-1)
        elif np.dot(theta,feature_vector) + theta_0 > 0:
            labels.append(1)
    return np.array(labels)


def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):
    theta, theta_0 = classifier(train_feature_matrix, train_labels, **kwargs)
    predicted_train_labels = classify(train_feature_matrix,theta,theta_0)
    predicted_val_labels = classify(val_feature_matrix,theta,theta_0)
    return accuracy(predicted_train_labels,train_labels), accuracy(predicted_val_labels,val_labels)


def extract_words(input_string):
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()


def bag_of_words(texts,file_name):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input
    """
    with open(file_name, 'r', encoding = 'utf-8') as file:
        stop_words = list(file.read().strip())
    dictionary = {} # maps word to unique index
    for text in texts:
        if text not in stop_words:
            word_list = extract_words(text)
            for word in word_list:
                if word not in dictionary:
                    dictionary[word] = len(dictionary)
    return dictionary


def extract_bow_feature_vectors(reviews, dictionary):
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.
    """
    word_count = 0
    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                word_count += 1
                feature_matrix[i, dictionary[word]] = word_count
            else:
                word_count = 0
    return feature_matrix


def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the percentage and number of correct predictions.
    """
    return (preds == targets).mean()
