# Team lovelace-p3

## Getting Started

Follow the below steps for installation and to run the training and testing sets.

## Prerequisites

- [Python 3.6](https://www.python.org/downloads/release/python-360/)
- [Anaconda](https://www.anaconda.com/) - Python Environment virtualization.
- [Keras](https://keras.io/#installation) - Open-source neural network library
- [Tensorflow](https://www.tensorflow.org/) - API used as Backend for Keras
- [OpenCV](https://opencv.org/) - Open-source library aimed for real-time Computer Vision
- [Thunder NMF Extraction](https://github.com/dsp-uga/Canady) - NMF Feature Extraction 

## Problem Statement 

## Neuron Finding 

Goal is to develop an image segmentation pipeline that identifies as many of the neurons present as possible, as accurately as possible.

## Installation

## Anaconda 

Anaconda is a free and open-source distribution of the Python and R programming languages for scientific computing, that aims to simplify package management and deployment.

Download and install Anaconda from (https://www.anaconda.com/distribution/#download-section). 

### Running Environment

•	Once Anaconda is installed, open anaconda prompt using windows command Line.\
•	Run ```conda env create -f environment.yml``` will install all packages required for all programs in this repository.

## Keras 

Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. You can install keras using pip on command line ```sudo pip install keras```.

## Tensorflow 

You can install Tensorflow using pip on command line, for CPU ```sudo pip install tensorflow``` and for GPU ```sudo pip install tensorflow-gpu```

##  Setting up Thunder-Extraction

You can install Thunder-Extraction  using pip on command line\
```pip install thunder-extraction```

To import your NMF library from thunder by using following commands:

```import thunder as td```\
```from extraction import NMF```

# Usuage
## Installing lovelace-p3 using pip
```python -m pip install --index-url https://test.pypi.org/simple/ --no-deps lovelace-p3-jayant12345```

   In this case you can import the package and call different methods as follows: \
```import lovelace-p3``` \
  `import downloader as dld`\
  `import zip_extractor as zip`\
  `import NMF_experiment as nmf`\
  `import unetpipeline as un`
  
  Put all parameters in variables with k_value, max_size_value form. Or to run with intilialized parameters, keep the method empty.
  
  `nmf.NMF_experiments(k=k_value,max_size=max_size_value, min_size=min_size_value,percentile=percentile_value, max_iter=max_iter_value,     overlap=overlap_value)`\

## Downloading the package
Alternatively, you can download the source code and simply run the following command:

`cd lovelace-p3`

`python main.py`

List of command line arguments to pass to the program are as follows:

  `--technique: technique to use to segment neurons out of NMF and Unets.`
  `--k: number of blocks to estimate per block. `
  `--max_size: max_size maximum size of each region`
  `--mix_size: min_size minimum size for each region`
  `--max_iter: max_iter maximum number of algorithm iterations`
  `--percentile: percentile value for thresholding (higher means more thresholding)`
  `--overlap:  overlap value for determining whether to merge (higher means fewer merges) `


## Data 

Each file is a TIFF image, separated into folders, where each folder is a single sample. There are 19 training samples, and 9 testing samples. Each folder contains a variable number of images; sample 00.00 contains 3,024 images, while sample 00.01 contains 3,048. The image files themselves are numbered, e.g. image00000.tiff, but all the images in a single folder represent the same sample, just
taken at different times with different calcium levels. The training labels exist at the sample level, so we used all the images in a single folder to learn the locations of the neurons. Each folder will have a unique sample with unique numbers and positions of
neurons. 

## Results 

## Data Science Ethics Policy Checklist 

We have impletmented Data Science Ethics Policy Checklist which can be found [here](https://github.com/dsp-uga/Team-lovelace-p3/blob/master/ETHICS.md)\
Details of every policy applied on this project/repo can be found [here](https://github.com/dsp-uga/Team-lovelace-p3/wiki/Data-Science-Ethics-Policy)


## Authors
(Ordered alphabetically)

- **Denish Khetan**
- **Jayant Parashar**
- **Vishakha Atole** 

See the [CONTRIBUTORS.md](https://github.com/dsp-uga/Team-thweatt-p2/blob/master/CONTRIBUTORS.md) file for details.

### References 

Project 3 Guidebook
