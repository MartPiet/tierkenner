README.MD

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

Note: The code in this folder is in a development stage. It is only showing the
Code that was running 100 Epochs during training the first model. The final
results is available under "Modell Version 2"

# Overall
## Requirements:
You need to have a appriopriate dataset to train this model. Please use the
utility programs for fetching needed data. Images have to be stored in this
folder under "images", consisting two folders: "train" and "validation". In
this two folder you have to store one folder for each classification with 
right pictures for each classification. The folders in "train" must be also
in "validation" but, of course, consisting with different pictures.

You may have to install some dependencies. First of all you need python.
You also need to have numpy install as well as keras, tensorflow, pillow,
matplotlib, coremltools, h5py and cPickle.

To save the generated results you need to have in the same folder where
tierkenner.py is located a folder named "Results". In That folder the
following folders are needed:

- CoreMLModels
- Plots
- Checkpoints
- ProgramHistory
- TrainHistories
- WeightedModels

## Usage
To run the program, simply use "python tierkenner.py" in your console.

You find most parameters for training in tierkenner.py. Otherwise feel
free to edit the code in Training.py and Model.py to fit your needs.
All parameters to manage e. g. filepaths are set in config.json. In 
this file you also can edit your metadata for the coremlmodel. I
recommend only editing the paramters under "coreml_metadata" and
"project".

## Output
During and after the program was running, a number of files were
created. All files the program created are stored under Results/.
Every file which is generated after a training gets a unique version 
number.
- mlmodel files will be saved in CoreMLModels.
- During the training two plots are generated after every epoch. This
plot is named partly current. One generated plot shows the accuracy
and one the loss. After each training there will be saved two additional
plot but now with the version number.
- In ProgramHistory the files tierkenner.py, Model.py and Training.py
will be saved.
- In TrainHistories the progression of the training will be saved.
- In WeightedModels all you weights will be saved.

## Note
To load a trained model, load get the model from your Model.py by using:

    model = define_model(input_shape, num_classes)

After you have your model, you need the weights that fits to your model:

    model.load_weights('{version_number}_weighted.h5')
