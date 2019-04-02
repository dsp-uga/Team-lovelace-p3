import os
import zipfile

def extract_zips():
	
	"""creates new folder data within data and extracts all tar files into it
	"""
	#loops over all directories within data folder
	for filename in os.listdir('../data'):
		if filename[-3:]!="zip":
			continue
		#prints filename for testing
		#print(filename)
		#opens tarfile into tar object for conversion. 
		zip_ref = zipfile.ZipFile('../data/'+filename, 'r')
		zip_ref.extractall('../data')
		zip_ref.close()
		
		#deletes zip file as we have extracted images from it
		os.remove('../data/'+filename)


#extract_zips()