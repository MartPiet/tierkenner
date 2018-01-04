'''
imagesSorter.py
version 1.0

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

import shutil
from os import walk
from os import listdir

PATH = 'images/unsorted/'

def main():
    '''
    Sorts images into train and validation images.
    '''
    class_dirnames = listdir(PATH)
    class_dirnames.remove('train')
    class_dirnames.remove('validation')

    # Iterate through all available class directories.
    for class_dirname in class_dirnames:
        # Counter index for each class directory. Needed to set unique name for each file.
        train_count = 0
        validation_count = 0
        # Iterate through all available wnid directories.
        for wnid_dirname in listdir(PATH + '/' + class_dirname):
            # Iterate through all files in wnid directory.
            for (_, _, filenames) in walk(PATH + class_dirname + '/' + str(wnid_dirname)):
                train_images_count, validation_images_count = split_dataset(filenames)

                # Data for train and validation images.
                sort_metadata = {
                    'source_path': PATH,
                    'class_dirname': class_dirname,
                    'wnid_dirname': wnid_dirname,
                    'filenames': filenames
                }

                # Sort images for training.
                train_count = sort_images(
                    0,
                    train_images_count - 1,
                    str(train_count),
                    PATH + 'train/',
                    sort_metadata
                    )

                # Sort images for validation.
                validation_count = sort_images(
                    train_images_count,
                    train_images_count + validation_images_count - 1,
                    str(validation_count),
                    PATH + 'validation/',
                    sort_metadata
                    )

def sort_images(start_index, end_index, total_index, destination_path, sort_metadata):
    '''
    Sort images into two datasets.

    - Parameter start_index: Index where the iteration through an array of filenames starts
                             from.
    - Parameter end_index: Index where the iteration through an array of filenames ends.
    - Parameter total_index: Index of all images of an class. Used to name sorted files
                             uniquely. Is needed, because this function is called for each
                             wnid directory within a class directory.
    - Parameter destination_path: Path where the sorted images should be saved.
    - Parameter sort_metadata: Dictionary of parameters which are usually the same for training
                               and validation images:
    - Key source_path: Path to unsorted images.
    - Key class_dirname: Name of an image class directory.
    - Key wnid_dirname: Name of a wnid directory.
    - Key filenames: Filenames of images.

    - Returns: Incremented total_index.
    '''
    source_path = sort_metadata['source_path']
    class_dirname = sort_metadata['class_dirname']
    wnid_dirname = sort_metadata['wnid_dirname']
    filenames = sort_metadata['filenames']

    for i in range(start_index, end_index):
        shutil.move(source_path +
                    class_dirname + '/' +
                    wnid_dirname + '/' +
                    filenames[i],

                    destination_path + 'train/' +
                    class_dirname + '/' +
                    class_dirname + '.' +
                    total_index + '.jpg'
                   )

        total_index = total_index + 1

    return total_index

def split_dataset(files):
    '''
    Counts given files and calculates numbers of a 1/4 split.

    - Parameter files: Files to count and split.

    - Returns: Numbers of files in part a (1/4) and part b (3/4).
    '''
    total_files_count = len(files)

    part_a_count = int(total_files_count * 0.75)
    part_b_count = int(total_files_count * 0.25)

    return part_a_count, part_b_count

if __name__ == '__main__':
    main()
