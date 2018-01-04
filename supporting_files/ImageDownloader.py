'''
imageDownloader.py
version 1.2
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

import concurrent.futures
import urllib
import json

'''
Valid username and accesskey is needed to download ImageNet dataset.
See also: http://image-net.org/download-images
'''
USERNAME = 'unknown'
ACCESSKEY = 'unknown'

# Path of json file with wnids.
WNIDS_JSON_PATH = 'images/wnids.json'

# Savepath for downloaded data.
DOWNLOAD_SAVEPATH = 'images/unsorted/'

def main():
    '''
    Downloads specific image datasets from ImageNet Dataset. Fetches wnids from
    json file to find needed images.
    '''

    # Opens json file with wnids to download specific datasets.
    wnids = json.load(open(WNIDS_JSON_PATH, 'r'))

    print('Starting Download...')
    # imageNet allows only 2 concurrent threads for downloading.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as e:
        for key in wnids.keys():
            for wnid in wnids[key]:
                e.submit(
                    download,
                    construct_url(wnid),
                    DOWNLOAD_SAVEPATH + key + '/' + wnid + '.tar'
                    )

def download(url, savepath):
    '''
    Opens URL and downloads files into given savepath.

    - Parameter url: URl to download data from.
    - Parameter savepath: Path to save downloaded data.
    '''
    
    response = urllib.URLopener()
    response.retrieve(url, savepath)
    
    print('Downloaded: ', url)

def construct_url(wnid):
    '''
    Builds URL to download specific ImageNet dataset.

    - Parameter wnid: wnid for specific images.

    - Returns: Constructed URL to a specific ImageNet dataset.
    '''

    url = (
        'http://www.image-net.org/download/synset?' +
        'wnid=' + wnid +
        '&username=' + USERNAME +
        '&accesskey=' + ACCESSKEY +
        '&release=latest&src=stanford'
        )

    return url

if __name__ == '__main__':
    main()
