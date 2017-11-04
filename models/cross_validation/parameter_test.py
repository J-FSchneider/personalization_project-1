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

    for k in k_val:
        # Generate model for the given k
        model_k = model(k=k)
        model_tester.fit_transform(data)
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
