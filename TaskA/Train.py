import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

from datetime import datetime
from model import create_model
from config import get_config
from plot import plot_history
from keras.utils import plot_model

def main():
    """
    code snippet to load the data and train a model
    :return:
    """
    data_directory = ""
    x_train = np.load(os.path.join(data_directory, "x_train.npy"))
    y_train = np.load(os.path.join(data_directory, "y_train.npy"))

    x_valid = np.load(os.path.join(data_directory, "x_valid.npy"))
    y_valid = np.load(os.path.join(data_directory, "y_valid.npy"))
    ####################################################################################################################
    # do some pre processing
    ####################################################################################################################
    # convert to grayscale if wanted (last dimension is the color channel)

    ####################################################################################################################
    # build your model
    ####################################################################################################################
    # in this example we use a simple linear model
    model = create_model(x_train.shape[1:])
    ret_values = get_config('batch_size', 'epochs')
    
    history = model.fit(x_train, y_train, batch_size=ret_values['batch_size'],
          epochs=ret_values['epochs'],
          validation_data=(x_valid, y_valid), 
          verbose=1)

    # save the model
    now = datetime.now()
    name_string = f"model_{now.timestamp()}"
    model.save_weights(f"./models/{name_string}")
    print("created and saved the model")

    plot_history(history, f"{name_string}_curve")
    plot_model(model, to_file=os.path.join('./plots', f'{name_string}_model.png'))

if __name__ == '__main__':
    main()
