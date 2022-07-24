from hashlib import new
from turtle import color
import matplotlib.pyplot as plt
import os

import numpy as np
from utils.evaluate import sigmoid
import networkx as nx


def plot(path, num, train_loss, dev_loss, test_loss):
    
    dir = os.path.join(path, 'converg') + str(num)

    fig, ax = plt.subplots(4,1,dpi=100, figsize=(12,18))
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
    

def plot_adj(path, descrip, adj, num=-1, weighted=False):

    # networks

    G = nx.Graph()
    if not weighted:
        edges = zip(np.where(adj>0))
        G.add_edges_from(edges)

    # else (x, y, w)
    plt.figure(figsize=(40,40), dpi=100)
    dir = os.path.join(path, descrip) + str(num) + '.png'
    pos = nx.spring_layout(G)

    if not weighted:
        nx.draw(G, pos, node_color='b', edgelist=edges, width=1.0, edge_cmap=plt.cm.Blues, node_size=1)
        nx.draw(G, pos, node_color='b', width=1.0, edge_cmap=plt.cm.Blues, node_size=1)
    else:
        pass
        # nx.draw(G, pos, node_color='b', edgelist=edges, edge_color=weights, width=1.0, edge_cmap=plt.cm.Blues, node_size=1)
    # plt.savefig('./edges1.png')
    plt.savefig(dir)
    plt.close()
