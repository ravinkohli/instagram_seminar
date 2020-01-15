from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop, Adam, Adadelta
from keras import backend as K
from keras import regularizers
from config import get_config
def create_model(image_shape):
    config = get_config()
    kernel_size = config['kernel_size']
    out_channel = config['out_channel']
    n_layers = config['n_layers']
    padding = config['padding']
    dense_units = config['dense_units']
    dropout = config["dropout"]
    pool_size = config['pool_size']
    activation = config['activation']
    lr = config['lr']
    l2 = config['l2']
    model = Sequential()
    
    # First layer of Batch norm

    # model.add(BatchNormalization(input_shape=image_shape))
    model.add(Conv2D(out_channel, kernel_size=(kernel_size, kernel_size),
                            padding=padding,
                            input_shape=image_shape, activation=activation))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size), padding=padding))

    for i in range(n_layers):
        model.add(Conv2D((out_channel), kernel_size=(kernel_size, kernel_size), activation=activation, padding=padding, kernel_regularizer=regularizers.l2(l2)))
        model.add(Dropout(dropout))
        model.add(MaxPooling2D(pool_size=(pool_size, pool_size), padding=padding))
    
    model.add(Conv2D((out_channel), kernel_size=(kernel_size, kernel_size), activation=activation, padding=padding))
    model.add(Dropout(dropout))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size), padding=padding))

    model.add(Flatten())
    model.add(Dense(dense_units, activation=activation))

    model.add(Dense(dense_units, activation=activation))

    model.add(Dense(4, activation='linear'))
    model.compile(loss='mean_absolute_error', optimizer=Adadelta(lr=lr), metrics=['mse', 'mae'])
    
    return model