#!/usr/bin/env python
# coding: utf-8

# In[23]:


import json
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from PIL import Image
import matplotlib.image as mpimg
import os
import cv2


# In[10]:


filenames=os.listdir("../data")

train_names=[files if files[-4:]!='test' else None for files in filenames]
train_names=[train_name for train_name in train_names if train_name]
print(train_names)

test_names=[files if files[-4:]=="test" else None for files in filenames]
test_names=[test_name for test_name in test_names if test_name]
print(test_names)


# In[35]:


#example.py bundled with data. 
#C:\Users\Jayant\Documents\sem2\dsp\project3\Team-lovelace-p3\data\neurofinder.00.02
# load the images

train_image_path=[]
test_image_path=[]
train_region_path=[]

for name in train_names: 
    train_image_path.append('../data/'+name+'/images/*.tiff')

for name in test_names: 
    test_image_path.append('../data/'+name+'/images/*.tiff')
    
for name in train_names:
    train_region_path.append('../data/'+name+'/regions/regions.json')
    
print(train_region_path)


# In[41]:


def get_train_image_list(train_image_path):
    
    train_images=[]   
    for path in train_image_path:
        file=sorted(glob(path))
    
        image=np.array([cv2.resize(plt.imread(f),(512,512), interpolation = cv2.INTER_CUBIC) for f in file])
        train_images.append(image)
    return train_images

#dims = imgs.shape[1:]


def tomask(coords):
    dims=(512,512)
    mask = np.zeros(dims)
    for indices in coords:
        mask[indices[0]][indices[1]] = 1
    #mask[zip(*coords)] = 1 # this is not working, anyways, it cant put 1 in every coordinate between a region. 
    return mask


def region_to_mask(train_region_path):
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
    
train_images_list=get_train_image_list(train_image_path)
mask_list=region_to_mask(train_region_path)
print(train_images_list[0].shape)
print(len(mask_list))


# In[ ]:


def create_train_nparray(train_images):
    train_nparray=np.ndarray(shape=(len(train_images), 512, 512, 1),
                     dtype=np.float32)
    for i in range(0,len(train_images)):
        train_nparray[i]=train_images[i]
    #np.save("train.npy",train_nparray)
    return train_nparray
    


# In[ ]:


#this converts regions to masks..how to do other way around?

def tomask(coords):
    mask = np.zeros(dims)
    for indices in coords:
        mask[indices[0]][indices[1]] = 1
    #mask[zip(*coords)] = 1 # this is not working, anyways, it cant put 1 in every coordinate between a region. 
    return mask

masks = np.array([tomask(s['coordinates']) for s in regions])
mask_region=masks.sum(axis=0)
mask_region[mask_region>1]=1

# show the outputs
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(imgs[1000,:,:])
plt.subplot(1, 2, 2)
plt.imshow(masks.sum(axis=0))
plt.show()
np.unique(mask_region)


# In[ ]:


from collections import namedtuple
from functools import reduce
Point = namedtuple('Point', 'x y')

def points_adjoin(p1, p2):
    # to accept diagonal adjacency, use this form
    #return -1 <= p1.x-p2.x <= 1 and -1 <= p1.y-p2.y <= 1
    return (-1 <= p1.x-p2.x <= 1 and p1.y == p2.y or
             p1.x == p2.x and -1 <= p1.y-p2.y <= 1)

def adjoins(pts, pt):
    return any(points_adjoin(p,pt) for p in pts)

def locate_regions(datastring):
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


# In[42]:


def masks_to_regions(masks,datasetname,dict_datasets=[]): 
    """
    Mask is a numpy array with 512*512 dimensions. One mask is created by summing all predictions from 3080 images' mask
    that are output of a unet model.
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
    Now THE PROBLEM is how to create regions?? Two steps :- first find if masks predictions are adjacent or not.. if not c
    create new region and enter its coordinates..
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
                
            


# In[ ]:


def masks_to_json():
    """
    
    """
    dict_dataset=[]
    mask_list=[]
    for testname,mask in zip(test_names,mask_list):
        masks_to_regions(mask,testname,dict_dataset)

    with open('submission.json', 'w') as f:
        f.write(json.dumps(dict_reg))
    


# In[ ]:


m,n=mask_region.shape
data=""
for i in range(0,m):
    for j in range(0,n):
        data=data+str(int(mask_region[i][j]))
    data=data+"\n"    
regs = locate_regions(data)


# In[ ]:


masks.shape


# In[ ]:


regions_dict={}
regions_dict["dataset"]="neurofinder.00.02"
regions_list=[]
dict_reg={}
for region in regs:
    reg_list=[]
    for values in region: 
        reg_list.append([values[0],values[1]])
    dict_reg["coordinates"]=reg_list
    regions_list.append(dict_reg)

regions_dict["regions"]=regions_list    


# In[ ]:


def json_foralltests():
    
with open('submission.json', 'w') as f:
    f.write(json.dumps(dict_reg))


# In[ ]:


#print(regions)


# In[ ]:


#print(regions_dict)


# In[ ]:


##this is a very noisy method...seems to give false regions and also does not detect many regions also...


# In[ ]:


from unet_model import unet


# In[ ]:


imgs.shape


# In[ ]:


mask_region.shape
new_mask=add_axis(mask_region)
new_mask=new_mask[np.newaxis,...]
new_mask.shape
m,n,o,p=image_array.shape


# In[ ]:


maskss= np.ndarray(shape=(m,n,o,p))
for i in range(m):
    maskss[i]=new_mask
                 
maskss.shape                 


# In[ ]:


np.save("train.npy",image_array)
np.save("masks.npy",maskss)


# In[ ]:


def train_model(train_nparray,masks): 
    """ 
    
    """
    model = unet()
    #Fitting and saving model
    model.fit(train_nparray, masks, batch_size=1, epochs=20, verbose=1, shuffle=True)
    model.save("model.h5")
    return None
def add_axis(img):
    """use it for adding one axis ie the 1 dimensio in the end.."""
    return img[...,np.newaxis]
#image_array=add_axis(imgs)


# In[ ]:



image_array=np.load("train.npy")
maskss=np.load("masks.npy")

train_model(image_array,maskss)


# In[ ]:




