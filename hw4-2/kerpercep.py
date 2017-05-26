#!/usr/bin/python2.7
# coding=utf-8
import sys, re
import numpy as np  # using numpy version 1.12.1
import math

'''
    Author: Angel Gabriel Ortega 🙃
    Project: Assignment #4 - Machine Learning ~ Prof. Turk
    Kernel Perceptron
'''


def interpret_output_filename(file_with_digits):
    try:
        file_no = int(re.match('.+([0-9]+)[^0-9]*$', file_with_digits).group(1))
        outputfile = 'output{0}.txt'.format(file_no)
    except (ValueError, AttributeError) as e:
        outputfile = 'output.txt'
    return outputfile

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def load_data(training_file):
    '''
    Transform a data file into a np ndarray with any N features.
    The first column is ignored because the record number does not add any values as a feature
    :param training_file:
    :return:
    '''
    with open(training_file) as file:
        num_points, point_dimensionality = map(int, re.split('\s+', file.readline().strip()))
    file.close()

    # Data is (n_samples, n_features)
    data = np.loadtxt(training_file, skiprows=1, usecols=range(0, point_dimensionality))
    return data, num_points, point_dimensionality


class PerceptronModel():

    def __init__(self, predictors, targets, epoch, kernelmodel=('rbf', 1.0)):
        self.predictors = predictors
        self.targets = targets
        self.sigma = kernelmodel[1]
        self.kernel=kernelmodel[0]
        self.num_iterations = epoch
        self.alphas = np.zeros( len(targets), dtype=int)

    def RBF_gaussian_kernal(self, x, y):
        deltaT = np.dot(x,y.T) / float(sigma * sigma)
        gram = np.exp(-deltaT)
        return gram

    def parameterize_RBF(self, x, y):
        self.gram_mat = np.dot(x, y.T)

        for i in range(self.gram_mat.shape[0]):
            for k in range(self.gram_mat.shape[1]):
                self.gram_mat[i,k] = self.RBF_gaussian_kernal(x[i], x[k])

        self.similiarities = np.array(self.gram_mat)

        return self.similiarities

    def converge_training_weights(self, alphas):
        self.alphas = alphas
        converged = False
        while not converged:
            converged = True
            for i in range(self.num_iterations):
                mysum = sum(self.targets*self.alphas*self.gram_mat[:,i])
                weights = self.targets[i] * mysum
                if weights <= 0:
                    self.alphas[i] += 1
                    converged ^= False


    def predict(self, testing_predictors, testing_targets):
        self.guess = 0 * testing_targets
        for i, test in enumerate(testing_predictors):
            k =self.parameterize_RBF(self.predictors, testing_predictors)
            # k = k / np.linalg.norm(k)
            K_norm = k[i,i]
            inner_prod1 = testing_targets[i]*self.alphas
            inner_prod2 = inner_prod1*K_norm
            inner_prod3 = inner_prod2
            mysum = sum(inner_prod3)
            if mysum <= 0:
                self.guess[i] = 0
            else:
                self.guess[i] = 1
        return self.guess

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print 'Error with program argument - usage: [sigma] [pos-training-file] [neg-training-file] [pos-testing-file]' \
              ' [neg-testing-file]'
        sys.exit(-1)

    sigma = float(sys.argv[1])
    pos_training_file = sys.argv[2]
    neg_training_file = sys.argv[3]
    pos_testing_file = sys.argv[4]
    neg_testing_file = sys.argv[5]
    output_filename = interpret_output_filename(pos_training_file)

    # Get predictors, also known as x_vector or feature_vector
    pos_training_predictors, pos_num_training_samples, pos_training_dimensionality = load_data(pos_training_file)
    neg_training_predictors, neg_num_training_samples, neg_training_dimensionality = load_data(neg_training_file)
    training_predictors = np.concatenate((pos_training_predictors, neg_training_predictors), axis=0)

    pos_testing_predictors, pos_num_testing_samples, pos_testing_dimensionality = load_data(pos_testing_file)
    neg_testing_predictors, neg_num_testing_samples, neg_testing_dimensionality = load_data(neg_testing_file)
    testing_predictors = np.concatenate((pos_testing_predictors, neg_testing_predictors), axis=0)

    # Get targets, also known as Y, actual, or labels
    pos_training_targets = np.ones(pos_num_training_samples, dtype=int)
    neg_training_targets = np.full(neg_num_training_samples, fill_value=-1, dtype=int)
    # y_training_targets = np.zeros(pos_num_training_samples + neg_num_training_samples, dtype=int)
    y_training_targets = np.concatenate((pos_training_targets, neg_training_targets))

    pos_testing_targets = np.ones(pos_num_testing_samples, dtype=int)
    neg_testing_targets = np.full(neg_num_testing_samples, fill_value=-1, dtype=int)
    y_testing_targets = np.concatenate((pos_testing_targets, neg_testing_targets))
    ground_truth_targets = map(lambda y: 0 if y == -1 else 1, y_testing_targets)

    num_training = len(y_training_targets)
    num_testing = len(y_testing_targets)

    clf_model = PerceptronModel(training_predictors, y_training_targets, num_training, kernelmodel=('rbf', sigma))
    similarities = clf_model.parameterize_RBF(training_predictors, training_predictors)
    clf_model.converge_training_weights(clf_model.alphas)
    predicted = clf_model.predict(testing_predictors, y_testing_targets)

    alphas_list = ' '.join('{0}'.format(v, i) for i, v in enumerate(clf_model.alphas))
    print 'Alphas:\t' + alphas_list



    pos_hits = np.logical_not(np.logical_xor(predicted, ground_truth_targets))
    false_positives = 0
    false_negatives = 0
    for i in range(len(predicted)):
        if predicted[i] == 1 and ground_truth_targets[i] == 0:
            false_positives += 1
        if predicted[i] == 0 and ground_truth_targets[i] == 1:
            false_negatives += 1
    print 'False positives:\t{0}'.format(false_positives)
    print 'False negatives:\t{0}'.format(false_negatives)

    error_rate = 1 - (sum(pos_hits) / float(len(y_testing_targets)))
    print 'Error rate:\t{0}%'.format(int(error_rate*100))


