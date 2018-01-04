'''
Training.py
Bachelorthesis 'Eine App zur Objekterkennung in Bildern mittels neuronaler Netze,
trainiert mit dem ILSVRC-Datensatz'

Hochschule Niederrhein

Copyright (c) 2017 Martin Pietrowski

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the 'Software'), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import keras
import tierkenner_ressources.Callbacks.HistoryVisualizerCallback as historyVisualizerCallback
from tierkenner_ressources import Config
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # Is needed to handle images with truncated images.

class Training(object):
    '''
    Handles training of a neural network.
    '''
    def __init__(self):
        self.tmp_config = Config.Config()

    def define_training(self, model, optimizer, loss):
        '''
        Defining Training

        - Parameter model: Model to define its training.

        Configures an Adam optimizer with a learning rate and a decay. Also compiles a given
        model with a lossfunction, optimizer and metric to be ready to train.
        '''
        # Configuring the optimizer. Adam is used as the optimizer with a learning rate of
        # 0.001. The learning rate decays during the training with a rate of 1e-6 (0.000001)
        # opt = keras.optimizers.Adam(lr=0.001, decay=1e-6)

        # Makes the model ready for training. Categorical_crossentropy is used as lossfunction,
        # the defined optimizer above is used and accuracy as metric for the console to be printed
        # out.
        model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=['accuracy']
            )

    def get_callbacks(self):
        '''
        Creates callbacks for training.
        '''
        checkpoint_filepath = self.tmp_config.checkpoint_filepath()

        checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_filepath) #Currently not in use
        visualizer = historyVisualizerCallback.HistoryVisualizer()
        callback_list = [visualizer, checkpoint]

        return callback_list

    def start_training(self, model, train_data):
        '''
        Starts training.

        - Parameter model: Model to train with.
        - Parameter train_data: Dictionary which contains data for training. Keys:
            train_generator: Training generator
            validation_generator: Validation generator
            num_train_samples: Number of training input files
            num_validation_samples: Number of validation input files
            epochs: Number of epochs
            batch_size: Batch size for training
            version: Version number of model

        Using train_generator as dataset, calculating steps per epoch: 30000 // 64 = 468
        for comparison: 30000 / 64 = 468,75. An integer is needed, that is why // is being
        used. Using a given number of epochs. Defining
        validation data and steps with validation generator and validation samples with the
        same calculation as for training steps but now with validation samples as first
        operand. The method fit_generator returns the training history.

        - Returns: Training history
        '''

        history = model.fit_generator(
            train_data['train_generator'],
            steps_per_epoch=train_data['num_train_samples'] // train_data['batch_size'],
            epochs=train_data['epochs'],
            callbacks=self.get_callbacks(),
            validation_data=train_data['validation_generator'],
            validation_steps=train_data['num_validation_samples'] // train_data['batch_size']
        )

        return history
    