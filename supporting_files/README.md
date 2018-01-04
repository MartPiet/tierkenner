README.md

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


# ImageDownloader.py #
## Requirements:
You need to have in the same folder as ImageDownloader.py is located the following
file structure:
* - images
* -- wnids.json
* -- unsorted
* --- bear
* --- ... (for each classification one folder)


The wnid.json file is a dictionary consisting of classification names as key and
an array of a number of wnid as value. E. g.:
{
    "bear": [
        "n02131653",
        ...
    ],
    ...
}
The keys must have a folder in unsorted/ with the same name as the key.


You also need to have access parameters you only get after a permitted request from
image-net.org. Fill your access parameters in USERNAME and ACCESSKEY.

## Note:
Feel free to change the global other variables WNIDS_JSON_PATH and
DOWNLOAD_SAVEPATH as you need them.

## Common issues:
After starting program, nothing happens:
This problem can occur because sometimes the image-net.org servers are really slow.

# SeparateImages.py
## Requirements
You need to have in the same folder as SeparateImages.py is located the following
file structure:
* - images
* -- unsorted
* --- bear
* ---- {wnid folder}
* ---- ...
* --- ... (for each classification one folder)
* -- train
* --- bear
* --- ... (for each classification one folder)
* -- validation
* --- bear
* --- ... (for each classification one folder)