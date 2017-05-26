import sys, re


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print 'Error with program argument - usage: [k] [training-file] [testing-file]'
        sys.exit(-1)

    sigma = float(sys.argv[1])
    pos_training_file = sys.argv[2]
    neg_training_file = sys.argv[3]
    pos_testing_file = sys.argv[4]
    neg_testing_file = sys.argv[5]

    # try:
    #     file_no = int(re.match('.+([0-9]+)[^0-9]*$', training_file).group(1))
    #     outputfile = 'output{0}.txt'.format(file_no)
    # except (ValueError, AttributeError) as e:
    #     outputfile = 'output.txt'
    #
    # training_data, labels, train_num_points, training_dimensionality = load_data(training_file, is_supervised_data=True)
    #
    # testing_data, test_num_points, testing_dimensionality = load_data(testing_file)
    #
    # knn(k, training_data, testing_data, labels, outputfilename=outputfile)

    # write_to_output(predicted, 'output1.txt')

    print 'Success'