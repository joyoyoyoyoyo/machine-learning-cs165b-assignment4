#!/usr/bin/python2.7
import re, sys
import numpy as np
from numpy import linalg # required for matrix multiplication or division
from collections import namedtuple

TestSample = namedtuple('TestSample', ['test', 'dimensionality'])
TrainSample = namedtuple('TrainSample', ['train', 'dimensionality'])

def load_data(training_or_testing_file):
    '''
    Dynamically transform a data file into a numpy ndarray with any N features.
    The first column is ignored because the record number does not add any values as a feature
    :param training_file: 
    :return: 
    '''
    with open(training_or_testing_file) as file:
        num_points, point_dimensionality = map(int, re.split('\s+', file.readline().strip()))
    file.close()

    # Data is (n_samples, n_features)
    data = np.loadtxt(training_or_testing_file, skiprows=1, usecols=range(0, point_dimensionality))

    # data = np.array(data)
    return data, num_points, point_dimensionality


def compute_centroid(data):
    '''
    
    :param data: 
    :return: mean numpy array of column means 
    '''
    mean = data.sum(axis=0) / float(len(data))
    return mean

def compute_covariance(data):
    '''
    
    :param data: 
    :return: 
    '''
    centroid = compute_centroid(data)
    variance = data - centroid
    varianceT = variance.T
    covariance_mat = variance.transpose().dot(variance)/float(len(data))

    return covariance_mat

def mahalanobis(covariance_mat, centroid, sample):
    deltaMean = sample.test - centroid
    covariance_mat_inverse = np.linalg.inv(covariance_mat)
    inner_product_1 = np.dot(deltaMean.T, covariance_mat_inverse)
    inner_product_2 = np.dot(inner_product_1.T, deltaMean)
    mahalanobis_distance = np.sqrt(inner_product_2)
    return mahalanobis_distance


if __name__ == "__main__":
    training_file = sys.argv[1]
    testing_file = sys.argv[2]
    train_mat, num_points, point_dimensionality = load_data(training_file)

    centroid = compute_centroid(train_mat)
    print 'Centroid:\n' + ', '.join('{0:.1f}'.format(v, i) for i, v in enumerate(centroid))


    covariance_mat = compute_covariance(np.array(train_mat))
    print 'Covariance matrix:'
    for i in range(len(covariance_mat)):
        cov_values = ' '.join('{0:.1f}'.format(v, i) for i, v in enumerate(covariance_mat[i]))
        print cov_values

    test_mat, test_num_points, test_point_dimensionality = load_data(testing_file)
    print 'Distances:'
    for index, test_sample in enumerate(test_mat):
        data_sample = TestSample(test_sample, point_dimensionality)
        distance = mahalanobis(covariance_mat, centroid, data_sample)
        data_values = ', '.join('{0:.1f}'.format(v, i) for i, v in enumerate(data_sample.test))
        print str(index+1) + '.\t' + data_values + '\t--\t' + '{0:.2f}'.format(distance)




