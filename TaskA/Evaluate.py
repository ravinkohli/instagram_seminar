"""
Simple script to demonstrate how the data set can be loaded and a prediction can be made.
You can, but you don't have to use this example.
Adapt this script as you want to build a more complex model, do pre processing, design features, ...

Author: Tano Müller

"""

import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import argparse
from model import create_model
from keras.utils import plot_model

# create parser
parser = argparse.ArgumentParser()
 
# add arguments to the parser
parser.add_argument('--path', type=str, help='location of the model to be evaluated')
 
# parse the arguments
args = parser.parse_args()

def main():
    """
    code snippet to load and evaluate the model
    :return:
    """
    data_directory = ""
    x_valid = np.load(os.path.join(data_directory, "x_valid.npy"))
    y_valid = np.load(os.path.join(data_directory, "y_valid.npy"))

    ####################################################################################################################
    # do some pre processing
    ####################################################################################################################
    # convert to grayscale if wanted (last dimension is the color channel)
    # x_valid = np.mean(x_valid, axis=3)
    # x_valid -= np.mean(x_train)
    
    # # convert to a 2D matrix since simple sklearn example model needs 2D inputs
    # x_valid = np.reshape(x_valid, (x_valid.shape[0], -1))
    # x_valid = np.expand_dims(x_valid, axis=3)
    ####################################################################################################################
    # make predictions
    ####################################################################################################################
    # load the model
    if not args.path: 
        dirs = os.listdir('./models')
        timestamps = [dir.split('_')[1].split('.')[0] for dir in dirs]
        index = np.argmax(timestamps)
        path = dirs[index]
    else:
        path = args.path
    name = path
    path = os.path.join('./models', path)
    model = create_model(x_valid.shape[1:])
    model.load_weights(path)
    
    plot_model(model, to_file=os.path.join('./plots', f'{name}.png'))

    # create the predictions
    scores = model.evaluate(x_valid, y_valid, verbose=2, batch_size=32)

    # get some info
    print("mae:", scores[1])
    print("mse:", scores[0])


if __name__ == '__main__':
    main()
