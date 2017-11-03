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

def precision(prediction, test_set, threshhold = 0.5):
	"""
	Calculates the precision between the test_set and the prediction

	:param prediction: dict | input of form: {(i,u): value}
	:param test_set: input of form: {(i,u): value}
	:param threshhold: Probability threshhold for prediction
	:return: value of Precision
	"""
	for j in prediction.keys():
		if prediction[j] >= threshhold:
			prediction[j] = 1
		else:
			prediction[j] = 0
	prec = 0
	tp = 0
	p = 0
	for i in test_set.keys():
		if prediction[i] == 1:
			p = p + 1
		if prediction[i] == test_set[i]:
			tp = tp + 1
	prec = tp / p

	print("Precision: " + str(prec))
	return prec
