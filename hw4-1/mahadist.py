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


    print data
    predictors, targets = data[:,:point_dimensionality-1], data[:,point_dimensionality-1]
    print predictors
    print targets
    # X_train, X_test, y_train, y_test = model_selection.train_test_split(predictors, targets, random_state=0)
    return predictors, targets

def centroid(*points):
  p1 = [2.3, 1.2] # 0.97
  p2 = [0.9, -3.1] # 0.32
  p3 = [0.0, 0.0] # 0.72
  p4 = [7.0, 7.0] # 2.82
  #centroid = (1/4)*np.abs((p1[0]-p2[0])+np.abs(p1[0]-p3[0])+np.abs(p1[0]-p4[0])+p3[0]+p4[0])
  centroid = [0.8, -2.1]
  # how would I print the list of elements in centroid?
  print 'Centroid\n {}'.format(centroid)

#def mahalanobis(*points, *origin):
#
 #   np.sqrt()
# def scatter_mat(*observations_vec)


# def covariance_matrix(observations_vec):


if __name__ == "__main__":
    training_file = sys.argv[1]
    testing_file = sys.argv[2]

    centroid(1)

    # raw_movie_data = pd.read_csv(training_file, delimiter=r'\s+').dropna()
    # movie_data = raw_movie_data.drop(raw_movie_data.columns[0], axis=1)
    #X_train, y_train_target = load_data(training_file)
    # X_test, y_test_target = load_data(testing_file)
    # movie_data = transform_N_class_to_binarized(movie_data)
    # print prediction_train




