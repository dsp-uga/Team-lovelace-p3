# Team lovelace-p3

# Team Members 

Denish Khetan\
Jayant Parashar\
Vishakha Atole

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

## Data 

Each file is a TIFF image, separated into folders, where each folder is a single sample. There are 19 training samples, and 9 testing samples. Each folder contains a variable number of images; sample 00.00 contains 3,024 images, while sample 00.01 contains 3,048. The image files themselves are numbered, e.g. image00000.tiff, but all the images in a single folder represent the same sample, just
taken at different times with different calcium levels. The training labels exist at the sample level, so you’ll use all the images in a single folder to learn the locations of the neurons. Each folder will have a unique sample with unique numbers and positions of
neurons. However, while time is a dimension to the data, you may not need to explicitly model it; you’re just interested in finding the active neurons in space.

## Results 

## Data Science Ethics Policy Checklist 

We have impletmented Data Science Ethics Policy Checklist. 
We have selected B.3 in data storage consisting data retention plan as we do not plan to alter or delete any data in future. 


### References 

- OpenCV Fourier Transform: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html
