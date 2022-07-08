import numpy as np

search_size = 20
rs_tunes = 'dropout,learning_rate'

hps_dropout = [0.3, 0, 0.3, 0]
hps_lr = np.random.rand(search_size) * 0.004 + 0.001    # [0.001, 0.005]
# hps_lr = np.power(10, hps_lr)   # [0.001, 0.01]

rs_hp_range = {
    "dropout": hps_dropout,
    "learning_rate": hps_lr,
}


def rs_set_hp_func(args, hp_values):
    hyperparams = rs_tunes.split(',')
    for hp in hyperparams:
        if hp == 'dropout':
            args.dropout = hp_values[hp]
        if hp == 'learning_rate':
            args.learning_rate = hp_values[hp]
