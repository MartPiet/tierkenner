'''
tierkenner.py

Version 3.0

Bachelorthesis "Eine App zur Objekterkennung in Bildern mittels neuronaler Netze,
trainiert mit dem ILSVRC-Datensatz"

Hochschule Niederrhein

Copyright (c) 2017 Martin Pietrowski

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import argparse
from tierkenner_ressources import ModelExporter, Model, Training, Config, ImagePreprocessor
import keras

def main(new, epochs):
    '''
    Defines model and training. Trains model with given data. Saves results.

    Change the parameters to set the training for your needs. To change model,
    go to Model.py. You can define multiple models to train them sequentially.
    '''
    exporter = ModelExporter.ModelExporter()
    model_architectures = [Model.Model()]
    preprocessor = ImagePreprocessor.ImagePreprocessor()
    training = Training.Training()
    tmp_config = Config.Config()


    # Directories from training and validation images.
    train_data_dir = tmp_config.train_data_dir()
    validation_data_dir = tmp_config.validation_data_dir()

    # Number of images for training and validation.
    num_train_samples = 150000
    num_validation_samples = 50000

    # Dimensions of the input images.
    img_width, img_height = 200, 150

    # Number of classes for objectclassification
    num_classes = 14

    # Number of training epochs.
    #epochs = 100

    # Batch size.
    batch_size = 64

    # Shape of the input: width, height, RGB.
    input_shape = (img_height, img_width, 3)

    # Train each model in model_architectures.
    for model_architecture in model_architectures:
        print tmp_config.get_current_version()

        # Set datagenerators for training and validation.
        train_generator, validation_generator = preprocessor.set_data_set_generators(
            img_width,
            img_height,
            batch_size,
            train_data_dir,
            validation_data_dir
            )

        optimizer = keras.optimizers.Adam(lr=0.0001, decay=1e-6)
        loss = 'mean_squared_error'

        # Creating dictionary for training input.
        train_data = {
            'train_generator': train_generator,
            'validation_generator': validation_generator,
            'num_train_samples': num_train_samples,
            'num_validation_samples': num_validation_samples,
            'epochs': epochs,
            'batch_size': batch_size,
            'version': tmp_config.get_current_version()
        }

        # Defining model and training.
        model = model_architecture.define_model(input_shape, num_classes)

        if not new:
            print 'Loading weights'
            current_version = tmp_config.get_current_version()
            model.load_weights('Results/Checkpoints/' + current_version + '_weighted.h5')
        
        training.define_training(model, optimizer, loss)

        # Starting training, then save results.
        try:
            history = training.start_training(model, train_data)
            exporter.save_results(model, history)
        except:
            print 'Error while starting training.'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load",
        help="Loads unfinished project.",
        action="store_true"
        )
    parser.add_argument(
        "--epochs",
        default=100,
        help="Set a specific number of epochs."
        )

    args = parser.parse_args()
    if args.load:
        main(False, args.epochs)
    else:
        main(True, args.epochs)
