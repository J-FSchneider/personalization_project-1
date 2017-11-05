# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Import Packages
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import pandas as pd

# =========================================================================

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Define Function
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def parameter_test(k_val,
                   cv_times,
                   model,
                   model_tester,
                   loss_function,
                   data):

    # Initialize empty dictionaries
    d_test = {}
    d_train = {}
    model_tester.fit_transform(data)

    for k in k_val:
        # Generate model for the given k
        model_k = model(k=k)
        model_k.fit(model_tester.data)

        for j in range(cv_times):
            # Obtain the predictions on test and train for the model
            pred_test = {(u, i): model_k.predict(u, i)
                         for (u, i) in model_tester.test_set}
            pred_train = {(u, i): model_k.predict(u, i)
                          for (u, i) in model_tester.train_set}

            # Get performance values by the given loss function
            val_test = \
                model_tester.evaluate_test(pred_test, loss_function)
            val_train = \
                model_tester.evaluate_train(pred_train, loss_function)

            # Fill-in the dictionaries
            if k not in d_test:
                d_test[k] = [val_test]
            else:
                d_test[k].append(val_test)

            if k not in d_train:
                d_train[k] = [val_train]
            else:
                d_train[k].append(val_train)

            # Shuffle CV and Train set
            model_tester.shuffle_cv()

    return d_test, d_train

# =========================================================================

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plotting Function
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def ready_to_plot(dictionary):
    """
    This function takes as an input a dictionary that results from
    testing different latent factos to the model.
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
