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
    hinge_loss_full = 0
    shape = np.shape(feature_matrix)
    for i, feature_vector in enumerate(feature_matrix):
        z = labels[i]*(np.dot(feature_vector,theta) + theta_0)
        if z < 1 :
            hinge_loss_full += max(0, 1 - z)
    return hinge_loss_full / shape[0]


def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):

    for i in range(3):
        new_theta = current_theta
        new_theta_0 = current_theta_0
        if label * (np.dot(current_theta,feature_vector) + current_theta_0) <= 0:
            new_theta = new_theta + label*feature_vector
            new_theta_0 = new_theta_0 + label
    return new_theta,new_theta_0


def perceptron(feature_matrix, labels, T):

    new_theta = np.zeros((feature_matrix.shape[1],))
    new_theta_0 = 0
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            updated_data = perceptron_single_step_update(feature_matrix[i],labels[i],new_theta,new_theta_0)
            new_theta = updated_data[0]
            new_theta_0 = updated_data[1]
            pass
    return new_theta,new_theta_0


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
