import numpy as np 
import pandas as pd

# TODO: should create a class for this model (Mohamed will take care of it)

def squared_error_gradient(M, U, V):
    # TODO: Add standard docstring
	# calculates the gradient and error of the simple MSE Loss function (unregularized)
    UV = np.dot(U, V)
    E = np.subtract(M, UV)
    J = 0.5 * np.nansum(np.square(E))
    
    return E, J


def simple_latent_factor_model(M,
                               latent_factors=10,
                               learning_rate=1e-6,
                               missing_data_weight=0,
                               missing_data_value=0):
    # TODO: two parameters not used
    # TODO: Add standard docstring
    # Calculates a latent factor model with give number of latent factors by gradient descent. 
    # Returns the ratings prediction matrix and the latent factor matrices U (users) and V (items)

    # Randomly initialize Matrix U and V
    m = M.shape[0]
    n = M.shape[1]
    U = pd.DataFrame(np.random.rand(m, latent_factors))
    V = pd.DataFrame(np.random.rand(latent_factors, n))

    # GD until convergence
    convergence = False
    i, J1 = 0, 1e6
    while not convergence:
        # compute Gradient and Loss
        E, J = squared_error_gradient(M, U, V)
        # fill Errors with 0 to ignore the values for prediction in the update
        E_upt = E.fillna(0)
        # update the matrices U and V by gradient descent
        U_update = np.dot(E_upt, np.transpose(V))
        V_update = np.transpose(np.dot(np.transpose(E_upt), U))
        U = U + learning_rate * U_update
        V = V + learning_rate * V_update

        print('Loss on iteration ' + str(i) + ': ' + str(J))

        # Convergence criteria
        if J / J1 > 0.9999:
            convergence = True
        else:
            J1 = J
        i += 1
    
    # TODO: to use in a separate function
    # calculate rating prediction

    R_hat = np.dot(U, V)
    # print R_hat.shape

    return R_hat, U, V


def test_accuracy(M, R_hat):
    # TODO: Add standard docstring
    # returns the accuracy of the model predictions

    # turn probabilities into predictions
    # TODO: fixe syntaxic errors here, cannot have a = b = c
    prediction = R_hat[R_hat > 0.5] = 1
    prediction = prediction[prediction <= 0.5] = 0

    # calculate difference between prediction and real value
    accuracy = np.subtract(M, prediction).abs()

    # return accuracy value
    # TODO: check unresolved reference here before
    total_acc = accuracy.sum().sum()/matrix.count().sum()

    return total_acc
