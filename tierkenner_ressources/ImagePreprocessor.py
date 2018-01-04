'''
ImagePreprocessor.py
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
from keras.preprocessing.image import ImageDataGenerator

class ImagePreprocessor(object):
    '''
    Handles training and validation data for preprocessing.
    '''
    def set_data_set_generators(
            self,
            img_width,
            img_height,
            batch_size,
            train_data_dir,
            validation_data_dir
        ):
        '''
        Generates data set for training with a given path to images.

        - Parameter img_width: Width of images.
        - Parameter img_height: Height of images.
        - Parameter batch_size: Size of batches during training.
        - Parameter train_data_dir: Directory of training images.
        - Parameter validation_data_dir: Directory of validation images.

        - Returns: tuple of train generator and validation generator.
        '''
        # Defining the ImageDataGenerator for training. Enables random changes on pictures like
        # zoom, flipping etc.
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
            )

        # Defining the ImageDataGenerator for validation. Rescaling only.
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        # Using the above defined ImageDataGenerator to load images.
        # Classmode is set to categorical.
        # because there are 10 classes (or categories) as labels for the images.
        train_generator = train_datagen.flow_from_directory(
            str(train_data_dir),
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical'
            )

        # Using the above defined ImageDataGenerator to load images.
        # Classmode is set to categorical.
        # because there are 10 classes (or categories) as labels for the images.
        validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical'
            )

        return train_generator, validation_generator
