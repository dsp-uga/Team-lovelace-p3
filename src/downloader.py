
"""
requirements :- 
gsutil
python
os
"""

import os
#if there is no dataset folder, creating one.  
if not os.path.exists('../data'):
    os.makedirs('../data')
	 

def data_download():
	"""
	Downloads data folder from google cloud , which contains zip folders  
	from gs://uga-dsp/project3
	"""
	#if data folder does not exist, it is created
	if not os.path.exists('../dataset/data'):
		os.makedirs('../dataset/data')

	command = 'gsutil cp gs://uga-dsp/project3/*'+' ../data/'
	os.system(command)


data_download()