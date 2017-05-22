import re, sys
import numpy as np

def load_data(training_or_testing_file):
    '''
    Dynamically transform a data file into a numpy ndarray with any N features.
    The first column is ignored because the record number does not add any values as a feature
    :param training_file: 
    :return: 
    '''
    with open(training_or_testing_file) as file:
        num_points, point_dimensionality = re.split('\s+', file.readline().strip())
        ncols = len(line)
        num_points = line[0]
        point_dimensionality = line[1]

    file.close()
    # Data is (n_samples, n_features)
    data = np.loadtxt(training_or_testing_file, skiprows=1, usecols=range(1, ncols-1))
    print data
    predictors, targets = data[:,:ncols-2], data[:,ncols-2]
    print predictors
    print targets
    # X_train, X_test, y_train, y_test = model_selection.train_test_split(predictors, targets, random_state=0)
    return predictors, targets

# def covariance_matrix(observations_vec):


if __name__ == "__main__":
    training_file = sys.argv[1]
    testing_file = sys.argv[2]



    # raw_movie_data = pd.read_csv(training_file, delimiter=r'\s+').dropna()
    # movie_data = raw_movie_data.drop(raw_movie_data.columns[0], axis=1)
    X_train, y_train_target = load_data(training_file)
    # X_test, y_test_target = load_data(testing_file)
    # movie_data = transform_N_class_to_binarized(movie_data)
    # print prediction_train




