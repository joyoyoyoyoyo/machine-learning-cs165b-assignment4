import re, sys
import numpy as np
from numpy import linalg # required for matrix multiplication or division

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
    data = np.loadtxt(training_or_testing_file, skiprows=1, usecols=range(1, point_dimensionality-1))
    return data, num_points, point_dimensionality


def compute_centroid(data):
    '''
    
    :param data: 
    :return: mean numpy array of column means 
    '''
    mean = data.sum(axis=0) / float(len(data))  # element wise divide
    # centroid = np.mean(data, axis=0)
    return mean

def compute_covariance(data):
    '''
    
    :param data: 
    :return: 
    '''
    centroid = compute_centroid(data)
    variance = np.subtract(data, centroid)
    covariance_mat = variance.dot(variance.T) / float(len(data))
    # variance = data - centroid
    # varianceT = variance.transpose()
    # covariance_mat = variance.transpose().dot(variance)/float(len(data))

    return covariance_mat


if __name__ == "__main__":
    training_file = sys.argv[1]
    testing_file = sys.argv[2]
    data, num_points, point_dimensionality = load_data(training_file)

    centroid = compute_centroid(data)
    # print 'Centroid:\n' + ','.join(' {0:.2f}'.format(v, i) for i,v in enumerate(centroid))
    print 'Centroid:\n' + ', '.join('{0:.2f}'.format(v, i) for i, v in enumerate(centroid))

    covariance_mat = compute_covariance(data)
    print covariance_mat
    print '\n\n'
    print np.cov(data[:], bias=True)
    # print np.cov(data, ddof=0)





