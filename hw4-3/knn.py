import re, sys
import numpy as np
from numpy import linalg  # required for matrix multiplication or division
from collections import OrderedDict

def load_data(training_file, is_supervised_data=False):
    '''
    Dynamically transform a data file into a numpy ndarray with any N features.
    The first column is ignored because the record number does not add any values as a feature
    :param training_file:
    :return:
    '''
    with open(training_file) as file:
        num_points, point_dimensionality = map(int, re.split('\s+', file.readline().strip()))
    file.close()

    # Data is (n_samples, n_features)

    if is_supervised_data:
        data = np.loadtxt(training_file, skiprows=1, usecols=range(0, point_dimensionality + 1))
        predictors = data[:,0:-1]
        labels = data[:,-1]
        return predictors, labels, num_points, point_dimensionality

    else:
        data = np.loadtxt(training_file, skiprows=1, usecols=range(0, point_dimensionality))
        return data, num_points, point_dimensionality


def euclidian(x, y):
    delta = np.abs(x - y)
    diff = delta * delta
    return sum(diff)

def knn(training_data, testing_data, labels, outputfilename):
    print 'knn starting...'

    output = {}
    with open(outputfilename, 'w') as file:
        for i, test_sample in enumerate(testing_data):
            distances = {}
            min_dist = None
            for train_sample, label in zip(training_data, labels):
                distance = euclidian(train_sample, test_sample)
                if distance not in distances:
                    distances[distance] = label
                else:
                    distance[distance] = min(distances[distance], label)

            min_dist = min(distances.keys())
            predicted = distances[min_dist]
            output[min_dist] = predicted
            # line = ' '.join('{0:.4f}'.format(v, i) for i, v in enumerate(test_sample))
            # label = predicted
            # file.write(str(i + 1) + '. ' + str(line) + ' -- ' + str(int(label)) + '\n')
            file.write('{0} . Minimum:\t{1:.4f} -- {2}\n'.format(i+1, min_dist, int(label))

    file.close()

    print 'knn finished...'
    return output




            # num_points, point_dimensionality = map(int, re.split('\s+', file.readline().strip()))
    # file.close()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print 'Error with program argument - usage: [k] [training-file] [testing-file]'
        sys.exit(-1)

    k = sys.argv[1]
    training_file = sys.argv[2]
    testing_file = sys.argv[3]

    training_data, labels, train_num_points, training_dimensionality = load_data(training_file, is_supervised_data=True)

    testing_data, test_num_points, testing_dimensionality = load_data(testing_file)

    predicted = knn(training_data, testing_data, labels, outputfilename='output1.txt')

    # write_to_output(predicted, 'output1.txt')

    print 'Success'


    # print 'Centroid:\n' + ','.join(' {0:.2f}'.format(v, i) for i,v in enumerate(centroid))
    # print 'Centroid:\n' + ', '.join('{0:.2f}'.format(v, i) for i, v in enumerate(['blank1', 'blank2', 'blank3']))
