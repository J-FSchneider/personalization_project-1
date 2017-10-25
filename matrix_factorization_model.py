import numpy as np 
import pandas as pd 

def squared_error_gradient(M,U,V):
	# calculates the gradient and error of the simple MSE Loss function (unregularized)
    UV = np.dot(U,V)
    E = np.subtract(M,UV)
    J = 0.5 * np.nansum(np.square(E))
    
    return E, J


def simple_latent_factor_model(M,latent_factors=10, learning_rate=0.000001,missing_data_weight=0, missing_data_value=0):
    # Calculates a latent factor model with give number of latent factors by gradient descent. 
    # Returns the ratings prediction matrix and the latent factor matrices U (users) and V (items)

    # Randomly initialize Matrix U and V
    m = M.shape[0]
    n = M.shape[1]
    U = pd.DataFrame(np.random.rand(m,latent_factors))
    V = pd.DataFrame(np.random.rand(latent_factors,n))
    print V.shape


    # GD until convergence
    convergence = False
    J1 = 1000000
    i = 0
    while not(convergence): 
    	# compute Gradient and Loss
        E, J = squared_error_gradient(M,U,V)
        # fill Errors with 0 to ignore the values for prediction in the update
        E_upt = E.fillna(0)
        # update the matrixes U and V by gradient descent
        U_update = np.dot(E_upt,np.transpose(V))
        V_update = np.transpose(np.dot(np.transpose(E_upt),U))
        U_new = U + learning_rate*U_update
        V_new = V + learning_rate*V_update
        U = U_new
        V = V_new

        print 'Loss on iteration ' + str(i) + ': ' + str(J)
        
        # Convergence criteria
        if J/J1 > 0.9999:
            convergence = True
        else:
            J1 = J
        i += 1
    
    # calculate rating prediciton

    R_hat = np.dot(U,V)
    print R_hat.shape
    return R_hat, U, V


def test_accuracy(M,R_hat):
	# returns the accuracy of the model predictions

	# turn probabilities into predictions
	prediction = R_hat[R_hat > 0.5] = 1
	prediction = prediction[prediction <= 0.5] = 0

	# calculate difference between prediction and real value
	accuracy = np.subtract(M,prediction).abs()

	# return accuracy value
	totalacc = Accuracy.sum().sum()/matrix.count().sum()

	return totalacc





    