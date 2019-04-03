#!/usr/bin/env python
# coding: utf-8


"""source and reference = https://github.com/thunder-project/thunder-extraction
""" 
import json
import thunder as td
from extraction import NMF


def NMF_experiments(k_value=5,max_size_value='full', min_size_value=20,percentile_value=99,max_iterations=50,overlap_value=0.1):
	
	"""
	The algorithm takes the following parameters.
	k number of components to estimate per block
	max_size maximum size of each region
	min_size minimum size for each region
	max_iter maximum number of algorithm iterations
	percentile value for thresholding (higher means more thresholding)
	overlap value for determining whether to merge (higher means fewer merges)
	"""
	datasets = [
	'00.00.test','00.01.test','01.00.test',
	'01.01.test','02.00.test','02.01.test',
	'03.00.test','04.00.test','04.01.test'
	]

	submission = []

	for dataset in datasets:
		print('processing dataset: %s' % dataset)
		print('loading')
		path = '../data/neurofinder.' + dataset
		data = td.images.fromtif(path + '/images', ext='tiff')
		print('analyzing')
		algorithm = NMF(k=k_value,max_size=max_size_value, min_size=min_size_value,percentile=percentile_value, max_iter=max_iterations, overlap=overlap_value)
		model = algorithm.fit(data, chunk_size=(50,50), padding=(25,25))
		merged = model.merge(0.1)
		print('found %g regions' % merged.regions.count)
		regions = [{'coordinates': region.coordinates.tolist()} for region in merged.regions]
		result = {'dataset': dataset, 'regions': regions}
		submission.append(result)
	print('writing results')
	with open('submission.json', 'w') as f:
		f.write(json.dumps(submission))

NMF_experiments()
