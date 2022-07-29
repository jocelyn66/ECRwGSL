from ast import arg
import numpy as np

np.random.seed(22)

search_size = 20
rs_tunes = 'learning_rate,rand_node_rate,beta'

# hps_dropout = [0] * 14
# hps_lr = [0.00001] * 6
# hps_lr = np.random.rand(search_size) * 0.004 + 0.001    # [0.001, 0.005]
hps_lr = np.random.rand(search_size)*3-5
hps_lr = np.power(10, hps_lr)   # [0.00001, 0.1]
hps_rand_node_rate = [0.2] * 20
# hps_encoder = ['gae', 'gvae'] * 10
hps_beta = np.random.rand(search_size)*6-5
# hps_beta = [0.1,0.001,0.0001,0.00001,0.000001,0.0000001]
hps_beta = np.power(10, hps_beta)


rs_hp_range = {
    # "dropout": hps_dropout,
    "learning_rate": hps_lr,
    "rand_node_rate": hps_rand_node_rate,
    "beta": hps_beta,
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
        if hp == 'beta':
            args.beta = hp_values[hp]
