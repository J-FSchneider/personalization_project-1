# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Define Function
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Note: I'm using the current rations schema for my dummy tests to work


def latent_factor_test(k_val,
                       cv_times,
                       model,
                       model_tester,
                       loss_function,
                       data,
                       ratios=(0.5, 0.2, 0.3)):

    # Initialize empty dictionaries
    d_test = {}
    d_train = {}

    for k in k_val:
        # Generate model for the given k
        model_k = model(k=k)
        tester = model_tester(ratios=ratios)
        tester.fit_transform(data)
        model_k.fit(tester.data)

        for j in range(cv_times):
            # Obtain the predictions on test and train for the model
            pred_test = {(u, i): model_k.predict(u, i)
                         for (u, i) in tester.test_set}
            pred_train = {(u, i): model_k.predict(u, i)
                          for (u, i) in tester.train_set}

            # Get performance values by the given loss function
            val_test = \
                tester.evaluate_test(pred_test, loss_function)
            val_train = \
                tester.evaluate_train(pred_train, loss_function)

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
            tester.shuffle_cv()

    return d_test, d_train

# =========================================================================
