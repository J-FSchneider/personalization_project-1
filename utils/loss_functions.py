def mean_squared_error(prediction, test_set):
    """
    Calculates the mean squared error between the test_set and the prediction
    :params prediction: dict | input of form: {(i,u): value}
    :params test_set: dict | input of form: {(i,u): value}
    """
	error = 0
	for i in test_set.keys():
		i_error = (prediction[i] - test_set[i])**2
		error += i_error
	error = error/len(test_set.keys())
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
	error = error/len(test_set.keys())
	print("AME: " + str(error))
	return error
