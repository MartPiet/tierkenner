'''
ModelExporter.py
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

import shutil
import pickle
import coremltools
import tierkenner_ressources.visualizer as visualizer
from tierkenner_ressources import Config

class ModelExporter(object):
    '''
    Saves results from trained model in form of plots, model
    and history. Converts trained model into .mlmodel.
    '''

    def __init__(self):
        self.tmp_config = Config.Config()

    '''
    Saving
    '''

    def save_results(self, model, history, description=''):
        '''
        Saves results after training.

        - Parameter model: Model that was trained.
        - Parameter histroy: Training history.
        '''
        self.save_program()
        self.save_history(history)
        self.save_weights(model)
        self.convert_to_coreml(model)
        self.save_training_visualization(history, description)
        self.tmp_config.increment_version_number()

    def save_cancelled_results(self, model):
        '''
        Svaes results after cancelled training.

        - Parameter model: Model that was trained.
        '''
        print '\nSaving program and model weights before cancelling training.\n'
        self.save_weights(model, cancelled=True)


    def save_program(self):
        '''
        Saves this program.
        '''
        shutil.copy(
            __file__,
            str(self.tmp_config.program_history_filepath())
        )
        shutil.copy(
            str(self.tmp_config.model_code_filepath()),
            str(self.tmp_config.model_code_history_filepath())
        )
        shutil.copy(
            str(self.tmp_config.training_code_filepath()),
            str(self.tmp_config.training_code_history_filepath())
        )

    def save_weights(self, model, cancelled=False):
        '''
        Saving trained weights after training.
        '''
        if cancelled:
            model.save_weights(self.tmp_config.cancelled_weight_models_history_filepath())
        else:
            model.save_weights(self.tmp_config.weight_models_history_filepath())

    def save_model(self, model):
        '''
        Saving model architecture.
        '''
        model.save(self.tmp_config.model_architecture_history_filepath())

    def save_history(self, history):
        '''
        Saving training history.

        - Parameter history: Training history.
        '''
        with open(self.tmp_config.train_history_filepath(), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

    def convert_to_coreml(self, model):
        '''
        Converts trained model to CoreML model.

        - Parameter model: Trained model to convert.

        Before converting meta data like author, class labels are set.
        '''

        coreml_model = coremltools.converters.keras.convert(
            model,
            input_names=self.tmp_config.coreml_input_names(),
            output_names=self.tmp_config.coreml_output_names(),
            image_input_names=self.tmp_config.coreml_image_input_names(),
            class_labels=self.tmp_config.coreml_class_labels()
        )

        coreml_model.author = self.tmp_config.coreml_author()
        coreml_model.license = self.tmp_config.coreml_license()
        coreml_model.short_description = self.tmp_config.coreml_description()

        coreml_model.input_description[
            self.tmp_config.coreml_input_names()[0]
            ] = self.tmp_config.coreml_input_description()

        coreml_model.output_description[
            self.tmp_config.coreml_output_names()[0]
            ] = self.tmp_config.coreml_output_description()

        print'CoreMLModel:'
        print coreml_model

        coreml_model.save(self.tmp_config.coreml_history_filepath())


    def save_training_visualization(self, history, description):
        '''
        Saves training results as a plot image.

        - Parameter history: Training history

        Generates two plots for accuracy and loss. x axis for epochs and y axis for accuracy/loss.
        Attention: not in use.
        '''

        visualizer.save_training_visualization(
            history=history.history,
            dst_accuracy=self.tmp_config.plot_accuracy_history_filepath(),
            dst_loss=self.tmp_config.plot_loss_history_filepath(),
            description=description
            )
