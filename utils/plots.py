

def hyperparameterplots(x1, y1, x2, y2):
    plt.subplot(121)
    for i in range(len(y1)):
        plt.plot(latent_factors, x1[:, i], label=y1[i])
    plt.legend(loc='best', ncol=2, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.xlabel("Number of Latent factors")
    plt.ylabel("MSE")

    plt.subplot(122)
    for i in range(len(y2)):
        plt.plot(latent_factors, x2[:, i], label=x2[i])
    plt.legend(loc='best', ncol=2, shadow=True, fancybox=False)
    leg.get_frame().set_alpha(0.5)
    plt.xlabel("Number of Latent factors")
    plt.ylabel("AME")

    plt.subplots_adjust(wspace=0.8, top=0.8)

def scaling_plot(data):
