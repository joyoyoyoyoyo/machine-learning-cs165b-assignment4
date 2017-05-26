import re, sys
import numpy as np
from numpy import linalg  # required for matrix multiplication or division


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
    return data, num_points, point_dimensionality


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print 'Error with program argument - usage: [k] [training-file] [testing-file]'
        sys.exit(-1)
    else:
        k = sys.argv[1]
        training_file = sys.argv[2]
        testing_file = sys.argv[3]

    data, num_points, point_dimensionality = load_data(training_file)

    print data.shape
    print 'Success'


    # print 'Centroid:\n' + ','.join(' {0:.2f}'.format(v, i) for i,v in enumerate(centroid))
    # print 'Centroid:\n' + ', '.join('{0:.2f}'.format(v, i) for i, v in enumerate(['blank1', 'blank2', 'blank3']))
