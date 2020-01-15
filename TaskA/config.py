config = dict()
config['n_layers'] = 5
config['out_channel'] = 16
config['kernel_size'] = 5
config['pool_size'] = 2
config['dropout'] = 0
config['dense_units'] = 128
config['activation'] = 'relu'
config['padding'] = 'SAME'
config['lr'] = 0.4
config['batch_size'] = 32
config['epochs'] = 20
config['l2'] = 0.01
def get_config(*args):
    if len(args) != 0:
        ret_config = dict()
        for arg in args:
            ret_config[arg] = config[arg]
        return ret_config
    else:
        return config