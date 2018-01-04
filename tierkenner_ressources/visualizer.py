import cPickle
import matplotlib
matplotlib.use('Agg') # This is needed before importing pyplot to render plots
                      # without a display turned on.
import matplotlib.pyplot as plt

def save_training_visualization(history, dst_accuracy, dst_loss, description=''):
    '''
    Saves training results as a plot image.

    - Parameter history: Training history
    - Parameter version: Version number

    Generates two plots for accuracy and loss. x axis for epochs and y axis for accuracy/loss.
    '''

    plt.clf()

    # Summarize history for accuracy.
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('accuracy: ' + description)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(dst_accuracy, dpi=300)

    # Resets plot for new plot.
    plt.clf()

    # Summarize history for loss.
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('loss: ' + description)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(dst_loss, dpi=300)

def visualize(data_path):
    '''
    Opens history file to generate two plots.

    - Parameter src_path: Path to the history file.
    '''

    history_file = open(data_path, 'rb')
    history = cPickle.load(history_file)
    save_training_visualization(history, 'test_accuracy', 'test_loss')
    history_file.close()
