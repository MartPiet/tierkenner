'''
Model.py
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

from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D
from keras.layers import Activation, Dropout, AlphaDropout, Flatten, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras import regularizers

class Model(object):
    '''
    Defines a model architecture.

    - Returns: Defined model.
    '''
    def define_model(self, input_shape, num_classes):
        '''
        Defines neural net model.

        - Parameter input_shape: Shape of the input, in this case shape of the image input.
        - Parameter num_classes: Number of classes to classify the input.

        - Returns: Defined model.
        '''
        # This model wil be a feed forward net. The layers will be organized sequentially.
        model = Sequential()

        activation = LeakyReLU()

        # Model overview:
        # conv3-64
        # maxpool
        # conv3-64
        # maxpool
        # conv3-64
        # conv3-64
        # maxpool
        # conv3-64
        # maxpool
        # conv3-64
        # conv3-64
        # FC-512
        # FC-512
        # FC-512
        # FC-num_classes

        # Convolution layer with 64 Features in size of 3x3. First layer needs
        # to now the input shape explicitly. Padding with 'same' as value does not fill
        # the image on the borders with zeros so the image does shrink in size. After
        # convolution, the LeakyReLU activation function is used following another
        # Convolution layer. This time the input shape will be set implicitly. Behind another
        # Leaky ReLU layer a pooling layer of size 2x2 is following. Before every
        # activation layer, batch normalization is being used.
        model.add(Conv2D(64, (3, 3), kernel_initializer='lecun_normal', padding='same', input_shape=input_shape))
        #model.add(BatchNormalization())
        model.add(Activation('selu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), kernel_initializer='lecun_normal', padding='same'))
        #model.add(BatchNormalization())
        model.add(Activation('selu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), kernel_initializer='lecun_normal', padding='same'))
        #model.add(BatchNormalization())
        model.add(Activation('selu'))
        model.add(Conv2D(64, (3, 3), kernel_initializer='lecun_normal', padding='same'))
        #model.add(BatchNormalization())
        model.add(Activation('selu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), kernel_initializer='lecun_normal', padding='same'))
        #model.add(BatchNormalization())
        model.add(Activation('selu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), kernel_initializer='lecun_normal', padding='same'))
        #model.add(BatchNormalization())
        model.add(Activation('selu'))
        model.add(Conv2D(64, (3, 3), kernel_initializer='lecun_normal', padding='same'))
        #model.add(BatchNormalization())
        model.add(Activation('selu'))

        # After flattening the input from 3D (RGB) to 1D a fully connected
        # layer with 512 neurons is being used. After ReLU and pooling, a dropout of 0.4
        # is used to counteract overfitting. The Dropout here is higher, because
        # there are much more neurons than in the previous convolution layers.

        model.add(Flatten())
        model.add(Dense(512, kernel_initializer='lecun_normal'))
        #model.add(BatchNormalization())
        model.add(Activation('selu'))
        model.add(AlphaDropout(0.4))
    
        model.add(Dense(512, kernel_initializer='lecun_normal'))
        #model.add(BatchNormalization())
        model.add(Activation('selu'))
        model.add(AlphaDropout(0.4))

        model.add(Dense(512, kernel_initializer='lecun_normal'))
        #model.add(BatchNormalization())
        model.add(Activation('selu'))
        model.add(AlphaDropout(0.4))

        # Last layer (output layer). This layer has to have as much neurons as
        # there are labels. Softmax is the normalization function.
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        # Prints architecture of the defined model.
        model.summary()

        return model
