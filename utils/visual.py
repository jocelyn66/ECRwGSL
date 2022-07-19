import matplotlib.pyplot as plt


def plot(path, train_loss, dev_loss, test_loss):

    fig, ax = plt.subplots(2,1)
    epoch = list(range(len(train_loss)))
    ax[0].plot(epoch, train_loss, '-D', label='Train', color='#4472C4')
    ax[0].plot(epoch, dev_loss, '-o', label='Dev', color='#C00000')
    ax[0].plot(epoch, test_loss, 'g-s', label='Test')
    ax[0].legend(loc='best')

    ax[1].plot(epoch, train_loss, '-D', color='#4472C4')
    ax[1].set_xlabel(r'Train', fontsize=15)
    # ax.set_title(name)
    # plt.xscale('log')
    # plt.grid(b=True, axis='y')

    # ax.grid(True)
    fig.tight_layout()

    # plt.imshow()
    plt.savefig('path', bbox_inches='tight')

    plt.pause(5)
    plt.close(fig)
    