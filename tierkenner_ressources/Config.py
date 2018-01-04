'''
Config.py
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

import json

class Config(object):
    '''
    To reduce hardcoded strings in other .py files.
    '''
    def __init__(self):
        self.config_file_path = 'tierkenner_ressources/config.json'

        self.config_data = json.load(open(self.config_file_path, 'r'))

        self.proj_key = 'project'
        self.dir_key = 'directories'
        self.files_key = 'project_files'
        self.result_key = 'result_filenames'
        self.coreml_key = 'coreml_metadata'

        self.version = self.config_data['project']['version']
        self.results_dir = self.config_data['directories']['results_dir']
        self.proj_name = self.get_value(self.proj_key, 'project_name')

    def get_value(self, main_key, sub_key):
        '''
        - Parameter main_key: Keytype of sub_key.
        - Parameter sub_key: Key for a value in config.json

        - Returns: specific value from config.json.
        '''
        value = self.config_data[main_key][sub_key]

        return value

    def train_data_dir(self):
        '''
        - Returns: Path to train images directory.
        '''
        return self.get_value(self.dir_key, 'train_data_dir')

    def validation_data_dir(self):
        '''
        - Returns: Path to validation images directory.
        '''
        return self.get_value(self.dir_key, 'validation_data_dir')

    def construct_result_filepath(self, path_key, filename_key, cancelled=False):
        '''
        Constructs a filepath for Results ressources.

        - Parameter path_key: Key for a path to a ressource directory.
        - Parameter filename_key: Key for a file prefix.
        - Parameter cancelled: Is result file from a cancelled training?

        - Returns: Constructed filepath.
        '''
        path = self.get_value(self.dir_key, path_key)
        filename = self.get_value(self.result_key, filename_key)

        if cancelled:
            version = 'cancelled'
        else:
            version = self.version

        filepath = (
            self.results_dir +
            path +
            version +
            '_' +
            filename
        )

        return filepath 

    def checkpoint_filepath(self):
        '''
        - Returns: Filepath to directory for saving training during training.
        '''
        return self.construct_result_filepath('checkpoints_dir', 'checkpoint')

    def program_history_filepath(self):
        '''
        - Returns: Filepath to directory for programfiles used for training.
        '''
        return self.construct_result_filepath('program_history_dir', 'main_program')


    '''
    Version
    '''

    def get_current_version(self):
        '''
        Returns current version number.
        '''
        return self.version

    def increment_version_number(self):
        '''
        Increments version number suffix in config.json.
        '''
        version_prefix, version_suffix = self.version.split('_', 1)
        version_suffix = str(int(version_suffix) + 1)

        self.version = version_prefix + '_' + version_suffix
        
        self.config_data['project']['version'] = self.version

        with open(self.config_file_path, 'w') as config_file:
            json.dump(self.config_data, config_file, indent=4)
            
    '''
    Model
    '''

    def model_code_filepath(self):
        '''
        Returns filepath of Model.py
        '''
        path = self.get_value(self.dir_key, 'ressources_dir')
        filename = self.get_value(self.files_key, 'model_program')

        return path + filename

    def model_code_history_filepath(self):
        '''
        Generates an unique filepath to save code of Model.py after training.
        '''
        return self.construct_result_filepath('program_history_dir', 'model_program')

    def model_architecture_history_filepath(self):
        '''
        Generates an unique filepath to save model architecture after training.
        '''
        return self.construct_result_filepath('model_architectures_dir', 'model_architecture')

    '''
    Training
    '''

    def training_code_filepath(self):
        '''
        Returns filepath of Training.py
        '''
        path = self.get_value(self.dir_key, 'ressources_dir')
        filename = self.get_value(self.files_key, 'train_program')

        return path + filename

    def training_code_history_filepath(self):
        '''
        Generates an unique filepath to save code of Training.py after training.
        '''
        return self.construct_result_filepath('program_history_dir', 'train_program')

    def train_history_filepath(self):
        '''
        Generates an unique filepath to save train history after training.
        '''
        return self.construct_result_filepath('train_histories_dir', 'train_history')

    '''
    Weights
    '''

    def weight_models_history_filepath(self):
        '''
        Generates an unique filepath to save weighted model after training.
        '''
        return self.construct_result_filepath('weighted_models_dir', 'weight')

    def cancelled_weight_models_history_filepath(self):
        '''
        Generates an unique filepath to save weighted model after cancelled training.
        '''
        return self.construct_result_filepath('weighted_models_dir', 'weight', cancelled=True)

    '''
    Plot
    '''

    def plot_accuracy_history_filepath(self):
        '''
        Generates an unique filepath to save plot for accuracy training.
        '''
        return self.construct_result_filepath('plots_dir', 'plot_accuracy')

    def plot_loss_history_filepath(self):
        '''
        Generates an unique filepath to save plot for loss training.
        '''
        return self.construct_result_filepath('plots_dir', 'plot_loss')

    def construct_current_plot_filepath(self, filename_key):
        '''
        Constructs filepath for plot of currently running trainings.

        - Parameter filename_key: Suffix for filename (plot_loss or plot_accuracy)

        - Returns: Filepath to save plot of currently running training.
        '''
        
        path = self.get_value(self.dir_key, 'plots_dir')
        filename = self.get_value(self.result_key, filename_key)

        filepath = (
            self.results_dir +
            path +
            '_current_' +
            filename
        )

        return filepath 

    def plot_accuracy_current_filepath(self):
        '''
        Generates a filepath to save plot for accuracy from current training.
        '''
        return self.construct_current_plot_filepath('plot_accuracy')

    def plot_loss_current_filepath(self):
        '''
        Generates a filepath to save plot for loss from current training.
        '''
        return self.construct_current_plot_filepath('plot_loss')

    '''
    CoreML
    '''

    def coreml_history_filepath(self):
        '''
        Generates an unique filepath to save weighted model after training.
        '''
        return self.construct_result_filepath('coreml_models_dir', 'coreml')

    def coreml_input_names(self):
        '''
        Returns input names for coreml model.
        '''
        return self.get_value(self.coreml_key, 'input_names')

    def coreml_output_names(self):
        '''
        Returns output names for coreml model.
        '''
        return self.get_value(self.coreml_key, 'output_names')

    def coreml_image_input_names(self):
        '''
        Returns image input names for coreml model.
        '''
        return self.get_value(self.coreml_key, 'image_input_names')

    def coreml_class_labels(self):
        '''
        Returns class labels for coreml model.
        '''
        class_labels_unicode = self.get_value(self.coreml_key, 'class_labels')
        class_labels_ascii = []

        for class_label in class_labels_unicode:
            class_labels_ascii.append(class_label.encode('ascii', 'ignore'))

        return class_labels_ascii

    def coreml_author(self):
        '''
        Returns author of coreml model.
        '''
        return self.get_value(self.coreml_key, 'author')

    def coreml_license(self):
        '''
        Returns license for coreml model.
        '''
        return self.get_value(self.coreml_key, 'license')

    def coreml_description(self):
        '''
        Returns description of coreml model.
        '''
        return self.get_value(self.coreml_key, 'description')

    def coreml_input_description(self):
        '''
        Returns descriptions for each input for coreml model.
        '''
        return self.get_value(self.coreml_key, 'input_description')

    def coreml_output_description(self):
        '''
        Returns descriptions for each output for coreml model.
        '''
        return self.get_value(self.coreml_key, 'output_description')
