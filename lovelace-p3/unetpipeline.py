#!/usr/bin/env python
# coding: utf-8


import json
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from PIL import Image
import matplotlib.image as mpimg
import os
import cv2
from unet_model import unet

from collections import namedtuple
from functools import reduce


def get_train_test_region_paths():

	"""
		reads data and creates path names list for training data , testing data and regions data
	
	
		Returns 
		train image, test image , train region urls: a tuple of lists , of String paths 
	
	"""
        
	filenames=os.listdir("../data")

	train_names=[files if files[-4:]!='test' else None for files in filenames]
	train_names=[train_name for train_name in train_names if train_name]
    #print(train_names)

	test_names=[files if files[-4:]=="test" else None for files in filenames]
	test_names=[test_name for test_name in test_names if test_name]
	#print(test_names)

	train_image_path=[]
	test_image_path=[]
	train_region_path=[]
	for name in train_names: 
		train_image_path.append('../data/'+name+'/images/*.tiff')

	for name in test_names: 
		test_image_path.append('../data/'+name+'/images/*.tiff')
    
	for name in train_names:
		train_region_path.append('../data/'+name+'/regions/regions.json')
	return  train_image_path,test_image_path,train_region_path



def create_nparray(train_images,name):
	"""
	creates an np array out of list of images
	Arguments
		---------
		train_images: list of nparray images 
	Returns 
		train_nparray: np array of images ,ready for training 
	"""
	train_nparray=np.ndarray(shape=(len(train_images), 512, 512, 1),dtype=np.float32)
	for i in range(0,len(train_images)):
		train_nparray[i]=train_images[i]
	np.save(name,train_nparray)
	return train_nparray
    


def get_image_list(train_image_path):
	"""
	creates a list of images 
	
	Arguments
		---------
		train_image_path: list of String 
	Returns 
		train_images: list of nparray images
	"""
    
	train_images=[]   
	for path in train_image_path:
		file=sorted(glob(path))
		#resize image to 512,512 while creating np array for each train folder
		image=np.array([cv2.resize(plt.imread(f),(512,512), interpolation = cv2.INTER_CUBIC) for f in file])
		train_images.append(image)
	return train_images

#dims = imgs.shape[1:]


def tomask(coords):
	"""
	creates a single mask out of coordinates 
	Arguments
		---------
		coords : list of coordinates, where coordinates are also 2 element list
		
	Returns 
	---------
	mask : returns an np array which has 0 and 1 unique values. 1 is for places where coordinates are.
		
	"""
	dims=(512,512)
	mask = np.zeros(dims)
	for indices in coords:
		mask[indices[0]][indices[1]] = 1
	return mask


def region_to_mask(train_region_path):
	"""
	creates regions to masks 
	
	Arguments
		---------
		train_region_path : list of regions.json bundeled with training files 
		
	Returns 
	---------
	mask_list : list of masks for each training folder 
		
	"""
	mask_list=[]
	for name in train_region_path:
		regions=[]
    # load the regions (training data only)
		with open(name) as f:
			regions.extend(json.load(f))
		masks = np.array([tomask(s['coordinates']) for s in regions])
		mask_region=masks.sum(axis=0)
		mask_region[mask_region>1]=1
		mask_list.append(mask_region)
	return mask_list
    

"""
reference= https://dsp.stackexchange.com/questions/2516/counting-the-number-of-groups-of-1s-in-a-boolean-map-of-numpy-array
@Paul Mcguire
"""
def points_adjoin(p1, p2):

    # to accept diagonal adjacency, use this form
    #return -1 <= p1.x-p2.x <= 1 and -1 <= p1.y-p2.y <= 1
	return (-1 <= p1.x-p2.x <= 1 and p1.y == p2.y or p1.x == p2.x and -1 <= p1.y-p2.y <= 1)

def adjoins(pts, pt):
	return any(points_adjoin(p,pt) for p in pts)

def locate_regions(datastring):
    
	Point = namedtuple('Point', 'x y')
	data = map(list, datastring.splitlines())
	regions = []
	datapts = [(Point(x,y) )
                for y,row in enumerate(data) 
                    for x,value in enumerate(row) if value=='1']
	for dp in datapts:
		# find all adjoining regions
		adjregs = [r for r in regions if adjoins(r,dp)]
		if adjregs:
			adjregs[0].add(dp)
			if len(adjregs) > 1:
				# joining more than one reg, merge
				regions[:] = [r for r in regions if r not in adjregs]
				regions.append(reduce(set.union, adjregs))
		else:
            # not adjoining any, start a new region
			regions.append(set([dp]))
	return regions

def region_index(regs, p):
	return next((i for i,reg in enumerate(regs) if p in reg), -1)



def masks_to_regions(mask,datasetname,dict_datasets=[]): 
    """
    This converts masks to regions. 
	 
	 The format of regions is :- 
    [
    {"dataset": "00.01.test",
    "regions":
    [
    {"coordinates": [ [0, 0], [0, 1] ]},
    {"coordinates": [ [10, 12], [14, 17] ]}
    ]
    }
    ]
	Arguments
	----------------
	mask : Mask is a numpy array with 512,512,1 dimensions. 
	datasetname : name of the dataset used
	dict_dataset: a dictionary containing previous datasets' regions on which this dataset is added
	
	Returns 
	----------------
	dict_dataset : returns the same list from argument after adding current mask converted regions of a dataset into the list. 
    
    """
    #mask=masks.sum(axis=0)
    m,n=mask.shape
    if(m!=512 and n!=512):
        raise Exception("Mask dimensions are wrong")
    data=""
    for i in range(0,m):
        for j in range(0,n):
            data=data+str(int(mask_region[i][j]))
            data=data+"\n"    
    regs = locate_regions(data)
    regions_dict={}
    regions_dict["dataset"]=datasetname
    regions_list=[]
    dict_reg={}
    for region in regs:
        reg_list=[]
        for values in region: 
            reg_list.append([values[0],values[1]])
        dict_reg["coordinates"]=reg_list
        regions_list.append(dict_reg)

    regions_dict["regions"]=regions_list  
    dict_datasets.append(regions_dict)
                
            

def masks_to_json():
    """
    converts masks list into json format for submission. 
    
    """
    dict_dataset=[]
    mask_list=[]
    for testname,mask in zip(test_names,mask_list):
        masks_to_regions(mask,testname,dict_dataset)

    with open('submission.json', 'w') as f:
        f.write(json.dumps(dict_reg))
    




def prepare_masks(result): 
    """converts an np array of (len,512,512,1) into proper masks...
    steps done =
    1. convert np array into list of len np arrays
    2. resize to original size (maybe a different into the sequence.)
    2. Drop the 1 in the end
    3. Convert float to int type and convert values into 0,1 and 2s
    4. convert whatever is left to greyscale (it might be something different liek black and white??)
    
    """
    length,x,y,z=result.shape
    masks={}
    if(len(test_names)!=length):
        raise Exception("Length of result is not equal to test_names length")
    for i in range(len(test_names)): 
        img =result[i][:,:,0]
        #calling rounding function..and convert to uint8 there
        int_img=round_float_image(img)
        x,y=dict_videos_test[test_names[i]][str(0)].shape
        resized_img= reshape_image(int_img,y,x) # dont understand why need to put inversed x,y , but thats how it comes right.
        masks[test_names[i]]=resized_img
        #grescale part is left, check if it is really so!! by generating masks
    
    return masks



def train_model(): 
	""" 
    trains the unet model and saves it 
	"""
	model = unet()
    #Fitting and saving model
	train_nparray=np.load("train.npy")
	masks=np.load("masks.npy")
	model.fit(train_nparray, masks, batch_size=1, epochs=20, verbose=1, shuffle=True)
	print("saving the model")
	model.save("model.h5")
	return None

def add_axis(img):
	"""use it for adding one axis ie the 1 dimensio in the end.."""
	return img[...,np.newaxis]



def predict():
	"""predicts values and save them in a single numpy array."""
    
	#loading model and predicting mask
	model=unet()
	model.load_weights("model.h5")
	test_nparray=np.load('test.npy')
	prediction = model.predict(test_nparray, batch_size=4,verbose=1)
	np.save('prediction.npy', prediction)
	return prediction
	
def remove_npy():
	"""
	removes npy files from current directory after creating submission.json
	
	"""
	os.remove("train.npy")
	os.remove("test.npy")
	os.remove("model.h5")
	os.remove("masks.npy")

train_image_path,test_image_path,train_region_path=get_train_test_region_paths()
train_images_list=get_image_list(train_image_path)
create_nparray(train_images,"train.npy")
test_images_list=get_image_list(test_image_path)
create_nparray(test_images_list,"test.npy")
mask_list=region_to_mask(train_region_path)
train_model()
result=predict()
masks=prepare_masks(result)
masks_to_json(masks)
remove_npy()
