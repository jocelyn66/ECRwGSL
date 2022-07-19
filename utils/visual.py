import matplotlib.pyplot as plt
import os

from numpy import size


def plot(path, train_loss, dev_loss, test_loss, num):

    
    dir = os.path.join(path, 'converg') + str(num)

    fig, ax = plt.subplots(4,1)
    epoch = list(range(len(train_loss)))
    ax[0].plot(epoch, train_loss, label='Train', color='#4472C4')
    ax[0].plot(epoch, dev_loss, label='Dev', color='#C00000')
    ax[0].plot(epoch, test_loss, label='Test')
    ax[0].legend(loc='best')

    ax[1].plot(epoch, train_loss, color='#4472C4')
    ax[1].set_xlabel(r'Train', fontsize=10)
    ax[2].plot(epoch, dev_loss, color='#C00000')
    ax[2].set_xlabel(r'Dev', fontsize=10)
    ax[3].plot(epoch, test_loss)
    ax[3].set_xlabel(r'Test', fontsize=10)
    # ax.set_title(name)
    # plt.xscale('log')
    # plt.grid(b=True, axis='y')

    # ax.grid(True)
    fig.tight_layout()

    # plt.imshow()
    plt.savefig(dir, bbox_inches='tight')

    plt.pause(5)
    plt.close(fig)
    