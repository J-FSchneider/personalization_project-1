def mean_squared_error(prediction, test_set):
    """
	Calculates the mean squared error between the test_set and the prediction
	:params prediction: dict | input of form: {(i,u): value}
	:params test_set: dict | input of form: {(i,u): value}
	"""
    error = 0
    for i in test_set.keys():
        i_error = (prediction[i] - test_set[i]) ** 2
        error += i_error
    error = error / len(test_set.keys())
    print("MSE: " + str(error))
    return error


def absolute_mean_error(prediction, test_set):
    """
	Calculates the absolute mean error between the test_set and the prediction
	:params prediction: dict | input of form: {(i,u): value}
	:params test_set: dict | input of form: {(i,u): value}
	"""
    error = 0
    for i in test_set.keys():
        i_error = abs(prediction[i] - test_set[i])
        error += i_error
    error = error / len(test_set.keys())
    print("AME: " + str(error))
    return error


def precision(prediction, test_set, threshhold=0.5):
    """
    Calculates the precision between the test_set and the prediction
    :param prediction: dict | input of form: {(i,u): value}
    :param test_set: input of form: {(i,u): value}
    :param threshhold: Probability threshhold for prediction
    :return: value of Precision
    """
    prediction1 = dict(prediction)
    for j in prediction1.keys():
        if prediction1[j] >= threshhold:
            prediction1[j] = 1
        else:
            prediction1[j] = 0
    tp = 0
    p = 0
    for i in test_set.keys():
        if prediction1[i] == 1:
            p = p + 1
        if prediction1[i] == 1 and test_set[i] == 1:
            tp = tp + 1
    prec = float(tp) / float(p)

    print("Precision: " + str(prec))
    return prec


def recall(prediction, test_set, threshhold=0.5):
    """
    Calculates the recall between the test_set and the prediction

    :param prediction: dict | input of form: {(i,u): value}
    :param test_set: input of form: {(i,u): value}
    :param threshhold: Probability threshhold for prediction
    :return: value of recall
    """
    prediction1 = dict(prediction)
    for j in prediction1.keys():
        if prediction1[j] >= threshhold:
            prediction1[j] = 1
        else:
            prediction1[j] = 0
    tp = 0.00
    fn = 0.00
    for i in test_set.keys():
        if prediction1[i] == 1 and test_set[i] == 1:
            tp = tp + 1

        if prediction1[i] == 0 and test_set[i] == 1:
            fn = fn + 1

    rec = float(tp) / (float(tp) + float(fn))

    print("recall: " + str(rec))
    return rec
