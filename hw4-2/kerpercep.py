# coding=utf-8
import sys, re
import numpy as np

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


# def kernelalize_matrix()


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
    neg_training_targets = np.zeros(neg_num_training_samples, dtype=int)
    y_training_target = np.concatenate((pos_training_targets, neg_training_targets), axis=0)

    pos_testing_targets = np.ones(pos_num_testing_samples, dtype=int)
    neg_testing_targets = np.zeros(neg_num_testing_samples, dtype=int)
    y_testing_target = np.concatenate((pos_testing_targets, neg_testing_targets), axis=0)


    # kernel_mat =

    #
    # testing_data, test_num_points, testing_dimensionality = load_data(testing_file)
    #
    # knn(k, training_data, testing_data, labels, outputfilename=outputfile)

    # write_to_output(predicted, 'output1.txt')

    print 'Success'
