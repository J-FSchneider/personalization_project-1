# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Import Packages
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import pandas as pd

# =========================================================================

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Define Functions
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def parameter_test(k_val,
                   model,
                   model_tester,
                   loss_function,
                   data,
                   cv_times=3,
                   verbose=True):
    """
    This functions iteratively evaluates the performance of a given algor
    against different values of latent factor. For evaluation purposes
    it leaves a hold-out set to understand the variance of the estimation
    within the training set.
    :param k_val: list | a list of integers where to take the different
                        values of the latent factors
    :param model: class obj | an instance of a class that defines de algo
                            to be used
    :param model_tester: class obj | an instance of a class that defines
                                    the evaluation and split of the data
    :param loss_function: func | a function that computes the error
    :param data: pd.DataFrame | a DataFrame that contains the information
    :param cv_times: int | the number of times that the metrics are going
                        to be computed in the training set
    :param verbose: boolean | to show the prints or the errors
    :return df_train: dict | a dictionary that contains as keys the user
                            item combinations and the probabilities
                            that the user will like the item. This for th
                            test set
    :return d_train: dict | a dictionary with the same characteristics as
                            above but for the training set
    """
    # Initialize empty dictionaries
    d_test = {}
    d_train = {}
    # Save an original copy of the data
    data_original = data.copy()

    for k in k_val:
        # Generate model for the given k
        model_tester.fit_transform(data)
        model_k = model(k=k)
        model_k.fit(model_tester.data)
        if verbose:
            print("----------- k = %i -----------\n" % k)

        # Get the predictions for the test set, as the test set does not
        # change for the CV
        pred_test = {(u, i): model_k.predict(u, i)
                     for (u, i) in model_tester.test_set}
        if verbose:
            print("Test set")
        val_test = \
            model_tester.evaluate_test(pred_test,
                                       loss_function,
                                       verbose=verbose)

        # Fill-in the dictionaries
        if k not in d_test:
            d_test[k] = [val_test]
        else:
            d_test[k].append(val_test)

        for j in range(cv_times):
            if verbose:
                print(">>> Fold %i:\n" % j)
            # Obtain the predictions on train set for the model
            pred_train = {(u, i): model_k.predict(u, i)
                          for (u, i) in model_tester.train_set}

            # Get performance values by the given loss function
            if verbose:
                print("Train set")
            val_train = \
                model_tester.evaluate_train(pred_train,
                                            loss_function,
                                            verbose=verbose)

            # Fill-in the dictionaries
            if k not in d_train:
                d_train[k] = [val_train]
            else:
                d_train[k].append(val_train)

            # Shuffle CV and Train set
            model_tester.shuffle_cv()

        # Reset in-place modifications
        data = data_original.copy()

    return d_test, d_train


def ready_to_plot(dictionary):
    """
    This function takes as an input a dictionary that results from
    testing different latent factors to the model. Then it outputs
    a pd.DataFrame that summarizes the results
    :param dictionary: dict | dictionary that contains per value of k
                        what was the value of the loss function
    :return df: pd.DataFrame | a summary data frame where per
                                each latent factor we get the
                                mean and std of the loss function
    """
    df = pd.DataFrame(dictionary)
    mean = df.mean(axis=0)
    std = df.std(axis=0)
    df_mean = pd.Series.to_frame(mean)
    df_std = pd.Series.to_frame(std)
    df_mean.columns = ["mean"]
    df_std.columns = ["std"]
    df = pd.merge(df_mean, df_std,
                  how="inner",
                  left_index=True,
                  right_index=True)
    return df

# =========================================================================
