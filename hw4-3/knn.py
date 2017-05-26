import re, sys
import numpy as np
from numpy import linalg  # required for matrix multiplication or division
from collections import OrderedDict, namedtuple, defaultdict

Sample = namedtuple('Sample', ['record_number', 'label', 'coordinates', 'euclidian_distance'])

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

def knn(k, training_data, testing_data, labels, outputfilename):
    print 'knn starting...'

    # output = {'sample_point':
    #               {'coordinates': [],
    #                'label': None,
    #                 'other_points': {'distance_sample_point': 'euclidian_distance': None}]}
    with open(outputfilename, 'w') as file:
        for i, test_sample in enumerate(testing_data):
            distances = {}
            distances_mutable = {}
            votes = {key: [] for key in labels}
            train_num = 1
            for train_sample, label in zip(training_data, labels):
                distance = euclidian(train_sample, test_sample)
                distances[train_num] = Sample(train_num, label, train_sample, distance)
                distances_mutable[train_num] = Sample(train_num, label, train_sample, distance)
                train_num += 1

            for num in range(k):
                min_dist = min(distances_mutable.values(), key=lambda key: key[3])
                predicted = min_dist.label
                votes[predicted].append(min_dist.euclidian_distance)
                del distances_mutable[min_dist.record_number]

            top = None
            for w in votes.values():
                top = max(len(w), top)

            nearest_neighbors = {key: min(v) for key, v in votes.items() if len(v) == top}
            lowest = {key: v for key,v in nearest_neighbors.items() if v == min(nearest_neighbors.values())}
            predicted_class = min(lowest.keys())
            # {k: v for k, v in votes.items() if v[lowest] == lowest}
            # x = min(nearest_neighbors, key=nearest_neighbors.get)

            # new_dict = defaultdict()
            # for k, v in votes.iteritems():
            #     new_dict[v].append(k)

            # max_overall = max(votes, key= lambda key: votes[key])
            # max_overall = [v for v in sorted(len(votes.items()), key=lambda kv: (-kv[1], kv[0]))]

            # min_dict = max_overall[0]
            # predicted = min_dict[0]
            # predicted = [v[0] for v in sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))

            line = ' '.join('{0:.4f}'.format(v, i) for i, v in enumerate(test_sample))
            predicted_class = str(predicted_class)
            # file.write(str(i + 1) + '. ' + str(line) + ' -- ' + str(int(label)) + '\n')
            file.write('{0} . {1} -- {2}\n'.format(i+1, line, predicted_class))
            # file.write('{0} . Minimum:\t{1:.4f} -- {2}\n'.format(i+1, min_dist, int(predicted)))

    file.close()

    print 'knn finished...'
    # return output




            # num_points, point_dimensionality = map(int, re.split('\s+', file.readline().strip()))
    # file.close()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print 'Error with program argument - usage: [k] [training-file] [testing-file]'
        sys.exit(-1)

    k = int(sys.argv[1])
    training_file = sys.argv[2]
    testing_file = sys.argv[3]

    try:
        file_no = int(re.match('.+([0-9]+)[^0-9]*$', training_file).group(1))
        outputfile = 'output{0}.txt'.format(file_no)
    except (ValueError, AttributeError) as e:
        outputfile = 'output.txt'

    training_data, labels, train_num_points, training_dimensionality = load_data(training_file, is_supervised_data=True)

    testing_data, test_num_points, testing_dimensionality = load_data(testing_file)

    knn(k, training_data, testing_data, labels, outputfilename=outputfile)

    # write_to_output(predicted, 'output1.txt')

    print 'Success'


    # print 'Centroid:\n' + ','.join(' {0:.2f}'.format(v, i) for i,v in enumerate(centroid))
    # print 'Centroid:\n' + ', '.join('{0:.2f}'.format(v, i) for i, v in enumerate(['blank1', 'blank2', 'blank3']))
