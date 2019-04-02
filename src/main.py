import os
import zipfile
import numpy as np
import argparse 
import downloader as dld
import zip_extractor as zip
import NMF_experiment as nmf
import unetpipeline as un

def main():

	# reading the command line arguments
	parser = argparse.ArgumentParser(description='Read in file paths and other parameters.')
	parser.add_argument('--technique', choices=['nmf','unet'], help='technique to use to segment neurons', default='nmf', type=str)
	parser.add_argument('--k', help='number of blocks to estimate per block', default="full", type=str)
	parser.add_argument('--max_size', help='max_size maximum size of each region', default="full", type=str)
	parser.add_argument('--min_size', help='min_size minimum size for each region', default=20, type=int)
	parser.add_argument('--max_iter', help='max_iter maximum number of algorithm iterations', default=20, type=int)
	parser.add_argument('--percentile', help='percentile value for thresholding (higher means more thresholding)',  default=95, type=int)
	parser.add_argument('--overlap', help='overlap value for determining whether to merge (higher means fewer merges)', default=0.1, type=float)
	
	args = parser.parse_args()

    #downloading the data as zip files
	#dld.download_data()

    #calling extractor to extract the downloaded files
	#zip.extract_zips()
	
	technique=args.technique
	k_value=args.k
	max_size_value=args.max_size
	min_size_value=args.min_size
	max_iter_value=args.max_iter
	percentile_value=args.percentile
	overlap_value=args.overlap
	
	
	
	if technique =='nmf':
		nmf.NMF_experiments(k=k_value,max_size=max_size_value, min_size=min_size_value,percentile=percentile_value, max_iter=max_iter_value, overlap=overlap_value)
	elif technique=="unet":
		print('Warning: This code is under progress')
		train_image_path,test_image_path,train_region_path=un.get_train_test_region_paths()
		train_images_list=un.get_image_list(train_image_path)
		un.create_nparray(train_images,"train.npy")
		test_images_list=un.get_image_list(test_image_path)
		un.create_nparray(test_images_list,"test.npy")
		mask_list=un.region_to_mask(train_region_path)
		un.train_model()
		result=un.predict()
		masks=un.prepare_masks(result)
		un.masks_to_json(masks)
		un.remove_npy()
if __name__ == '__main__':
	sys.exit(main())
