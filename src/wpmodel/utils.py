import os
import re
import gzip
import io

import dill
 
from . import helpers  

def get_latest(
	model_name : str,
	model_df : str = 'out',
	from_cloud: bool = True
):
	"""Get the latest version of the requested model_name

	Parameters
	----------
	model_name: model component name
	model_df: folder name if use a disc file system
	from_cloud: override model_df, and retrieves model from Azure blob storage

	"""
	if from_cloud:
		compressed = True
		conn = helpers.container_conn()
		blob_list = [i['name'] for i in conn.list_blobs()]
		
		 
		files = [i for i in blob_list if re.search(f'{model_name}_\d+\\.gz', i )]
		if not files:
			files = [i for i in blob_list if re.search(f'{model_name}_\d+\\.pkl', i)]
			compressed = False
		if not files:
			files = [f'{model_name}.gz']
			compressed = True
		blob_name = max(files)
		
		blob_bytes = conn.get_blob_client(blob=blob_name).download_blob().readall()
		if compressed:
			with gzip.open(io.BytesIO(blob_bytes),'rb') as f:
			    model = dill.load(f)
		else:
			with helpers.FakeFile(blob_bytes) as ff:
				model = dill.load(ff.f)
		return model 	
			
	latest = max([
		f
		for f in os.listdir(model_df)
		if re.search(f'{model_name}_\d+.(pkl|gz)', f)
	])

	with open(os.path.join(model_df, latest), 'rb') as file:
		model = dill.load(file)

	return model

class Placeholder:
	def __init__(self):
		pass

def get_all_models(
	model_df : str = 'out',
	from_cloud: bool = True,
	detailed : bool = False,
	pattern : str = '.*'
):
	if from_cloud:
		conn = helpers.container_conn()
		blob_list = conn.list_blobs()
		 
		models = set([
			re.split('_\d+.pkl',i['name'])[0]  for i in blob_list 
			if re.search('_\d+.pkl', i['name'])
			])
	else:
		models = set([re.split('_\d+.pkl',x)[0] for x in os.listdir(model_df)\
		 if re.search('_\d+.pkl',x)])
	
	if not detailed:
		return set([m for m in models if re.search(pattern, m) is not None])

	latest_models = []
	for m in models:
		if re.search(pattern, m) is not None:
			try:
				latest_models.append((m, 'Loadable', None, get_latest(m)))
			except Exception as err:
				latest_models.append((m, 'Load Error', str(err), Placeholder))

	return helpers.compile_model_info(latest_models)


if __name__ == '__main__':
	get_latest('ED Baseline Forecast')
