import matplotlib.pyplot as plt 
import os
def plot_history(history, name):
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')  
    plt.savefig(f"{os.path.join('./plots', name)}.png")
