from ast import arg
import numpy as np

np.random.seed(22)

search_size = 14
rs_tunes = 'learning_rate,rand_node_rate'

# hps_dropout = [0] * 14
hps_lr = [0.00001] * 7 + [0.00003] * 7
# hps_lr = np.random.rand(search_size) * 0.004 + 0.001    # [0.001, 0.005]
# hps_lr = np.random.rand(search_size)*4-6
# hps_lr = np.power(10, hps_lr)   # [0.00001, 0.1]
hps_rand_node_rate = [1., 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]*2
# hps_encoder = ['gae', 'gvae'] * 10


rs_hp_range = {
    # "dropout": hps_dropout,
    "learning_rate": hps_lr,
    "rand_node_rate": hps_rand_node_rate,
    # "encoder": hps_encoder
}


def rs_set_hp_func(args, hp_values):
    hyperparams = rs_tunes.split(',')
    for hp in hyperparams:
        if hp == 'dropout':
            args.dropout = hp_values[hp]
        if hp == 'learning_rate':
            args.learning_rate = hp_values[hp]
        if hp == 'rand_node_rate':
            args.rand_node_rate = hp_values[hp]
        if hp == 'encoder':
            args.encoder = hp_values[hp]
