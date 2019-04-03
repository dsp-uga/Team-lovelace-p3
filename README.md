# Team lovelace-p3

# Team Members 

Denish Khetan\
Jayant Parashar\
Vishakha Atole

See the [CONTRIBUTORS.md](https://github.com/dsp-uga/Team-lovelace-p3/blob/master/CONTRIBUTORS.md) file for details.

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

   In this case you can import the package and call different methods as follows:\
```import lovelace-p3``` \
  `import downloader as dld`\
  `import zip_extractor as zip`\
  `import NMF_experiment as nmf`\
  `import unetpipeline as un`
  
  Put all parameters in variables with k_value, max_size_value form. Or to run with intilialized parameters, keep the method empty.\
  
  `nmf.NMF_experiments(k=k_value,max_size=max_size_value, min_size=min_size_value,percentile=percentile_value, max_iter=max_iter_value,     overlap=overlap_value)`

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

The final parameters with which we got maximum accuracy are listed below for each test file. 

|     Dataset      |   Chunk_size  | k | max_iteration | Percentile | Accuracy |
|------------------|---------------|---|---------------|------------|----------|
| neurofinder00.00 |	50*50      | 10|      50       |     95     |   3      |
| neurofinder00.01 |	60*60      |  3|      30       |     95     |   3.2    |
| neurofinder01.00 |	50*50      | 10|      20       |     95     |   3.14115|
| neurofinder01.01 |	50*50      |  5|      50       |     95     |   3.29171|
| neurofinder02.00 |   100*100     |  5|      50       |     99     |   3.44819|
| neurofinder02.01 |   100*100     |  5|      50       |     99     |   3.3    |
| neurofinder03.00 |   100*100     | 10|      50       |     95     |   2.91006|
| neurofinder04.00 |   100*100     |  5|      50       |     95     |   2.96528|
| neurofinder04.01 |   100*100     |  5|      30       |     95     |   3.331  |

Overall Accuracy 👍 3.16947  


# Data Science Ethics Policy Checklist 

## A. Data Collection

 - [x] **A.3 Limit PII exposure**: Have we considered ways to minimize exposure of personally identifiable information (PII) for example through anonymization or not collecting information that isn't relevant for analysis?\
 This dataset does not use PII information. We remove subject names and their identity in the dataset collection, just maintaining one or two personal traits which can never be used to identity an individual solely such as ethinicity and gender. 

## B. Data Storage

 - [x] **B.1 Data security**: Do we have a plan to protect and secure data (e.g., encryption at rest and in transit, access controls on internal users and third parties, access logs, and up-to-date software)?\

## C. Analysis
- [x] **C.5 Auditability**: Is the process of generating the analysis well documented and reproducible if we discover issues in the future?\
 Yes, we have created a thorough documentation to reproduce our results. 
 
 ## D. Modeling
 - [x] **D.1 Proxy discrimination**: Have we ensured that the model does not rely on variables or proxies for variables that are unfairly discriminatory?\
     No variables are discriminatory. The firing of neurons in brain is a general phenomenon among humans.
 ## E. Deployment
 - [x] **E.1 Redress**: Have we discussed with our organization a plan for response if users are harmed by the results (e.g., how does the data science team evaluate these cases and update analysis and models to prevent future harm)?\
 The only way users can be harmed by this result is if they have a medical condition that we found and is leaked by us. But since we do not hold data with personal information. Users can not be harmed by results. Moreover, this study only calculate calcium ions based firing of neurons. It can not detect any diseases or ailments with certainity according to current medical knowledge. 
     
     
 We will protect this data against any misuse by giving access to only machine learning researcher of this field. 
 
 - [x] **B.2 Right to be forgotten**: Do we have a mechanism through which an individual can request their personal information be removed?\
 The data does not contain PII exposure. However, we reserve the right for every individual. They can share their index number with us and we can remove their data from our dataset. 

 Yes, we considered racial and gender bias. And we collected data from diverge backgrounds. 
We have impletmented Data Science Ethics Policy Checklist which can be found [here](https://github.com/dsp-uga/Team-lovelace-p3/blob/master/ETHICS.md)\
Details of every policy applied on this project/repo can be found [here](https://github.com/dsp-uga/Team-lovelace-p3/wiki/Data-Science-Ethics-Policy)

### References 
[Thunder-extraction](https://github.com/thunder-project/thunder-extraction)\
[Project 3 Guidebook](https://github.com/dsp-uga/sp19/blob/master/projects/p3/project3.pdf)
