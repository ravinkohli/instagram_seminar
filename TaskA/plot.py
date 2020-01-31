import matplotlib.pyplot as plt 
import os
import numpy as np
def plot_history(history, name):
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(0, len(history.history['loss']) +1, 1.0))
    plt.legend(['Train', 'Valid'], loc='upper left')  
    plt.savefig(f"{os.path.join('./plots', name)}.png")

def plot_final_curve(history, name, test_loss):
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.axhline(y=test_loss, color='r', linestyle='--')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')  
    plt.savefig(f"{os.path.join('./plots', name)}.png")