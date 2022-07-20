from turtle import color
import matplotlib.pyplot as plt
import os

from numpy import size


def plot(path, num, train_loss, dev_loss, test_loss):
    
    dir = os.path.join(path, 'converg') + str(num)

    fig, ax = plt.subplots(4,1,dpi=100, figsize=(18,12))
    epoch = list(range(len(train_loss)))
    ax[0].plot(epoch, train_loss, label='Train', color='#4472C4')
    ax[0].plot(epoch, dev_loss, label='Dev', color='#C00000')
    ax[0].plot(epoch, test_loss, label='Test', color='green')
    ax[0].legend(loc='best')

    ax[1].plot(epoch, train_loss, color='#4472C4')
    ax[1].set_title('Train')
    ax[2].plot(epoch, dev_loss, color='#C00000')
    ax[2].set_title('Dev')
    ax[3].plot(epoch, test_loss, color='green')
    ax[3].set_title('Test')
    # ax.set_title(name)
    # plt.grid(b=True, axis='y')

    # ax.grid(True)
    fig.tight_layout()

    # plt.imshow()
    plt.savefig(dir, bbox_inches='tight')

    plt.pause(10)
    plt.close(fig)
    