# coding=utf-8
import sys, re
import numpy as np
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
    Transform a data file into a numpy ndarray with any N features.
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

    def __init__(self, predictors, targets, sigma):
        self.predictors = predictors
        self.targets = targets
        self.size_of_train = len(predictors)
        self.sigma = sigma
        false_positives = None
        false_negatives = None

    def converge_train_weights(self, num_iterations, sigma):
        delta_alph = np.zeros(self.size_of_train)
        for j in range(num_iterations):
            for i in range(self.size_of_train):
                K_mat_similarities = self.RBF_gaussian_kernal(self.predictors, self.targets, sigma)
                hypothesis = np.dot(np.linalg.multiply(delta_alph, self.targets), K_mat_similarities.T)
                if sigmoid(hypothesis) > 0.5:
                    predicted = 1
                else:
                    predicted = 0
                if predicted != self.predictors(i):
                    delta_alph = delta_alph + 1

        return delta_alph

    def RBF_gaussian_kernal(self, x, y, sigma):
        similarities = np.exp(- np.linalg.norm(x - y) / (2 * sigma ** 2))
        return similarities


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
    neg_training_targets = np.full(shape=(neg_num_training_samples,), fill_value= -1, dtype=int)
    y_training_targets = np.concatenate((pos_training_targets, neg_training_targets), axis=0)

    pos_testing_targets = np.ones(pos_num_testing_samples, dtype=int)
    neg_testing_targets = np.full(shape=(neg_num_testing_samples,1), fill_value=-1, dtype=int)
    y_testing_targets = np.concatenate((pos_testing_targets, neg_testing_targets), axis=0)

    # support_vectors =
    #  = converge_for_min(training_predictors, y_training_target)
    # kernel_mat = radial_basis_function(training_predictors, )

    clf_model = PerceptronModel(training_predictors, y_training_targets, sigma)
    alphas = clf_model.converge_train_weights(num_iterations=len(training_predictors)+len(testing_predictors))

    pos_hits =  np.logical_not(np.logical_xor(clf_model.RBF_gaussian_kernal(testing_predictors, y_testing_targets, sigma), y_testing_targets))

    error_rate = 1 - (pos_hits / float(len(y_testing_targets)))


    with open(output_filename, 'w') as file:
        line = ' '.join('{0}'.format(v, i) for i, v in enumerate(alphas))
        file.write('Alphas: ' + line + '\n')
        print 'Error'

    #
    # testing_data, test_num_points, testing_dimensionality = load_data(testing_file)
    #
    # knn(k, training_data, testing_data, labels, outputfilename=outputfile)

    # write_to_output(predicted, 'output1.txt')

    print 'Success'
