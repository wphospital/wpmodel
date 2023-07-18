import os
import re
import dill
import helpers  

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
		conn = helpers.container_conn()
		blob_list = conn.list_blobs()
		 
		blob_name = max([
			i['name'] for i in blob_list 
			if re.search(f'{model_name}_\d+.pkl', i['name'])
			])
			
		blob_bytes = conn.get_blob_client(blob=blob_name).download_blob().readall()
		with helpers.FakeFile(blob_bytes) as ff:
			return dill.load(ff.f)
			
			
	latest = max([
		f
		for f in os.listdir(model_df)
		if re.search(f'{model_name}_\d+.pkl', f)
	])

	with open(os.path.join(model_df, latest), 'rb') as file:
		model = dill.load(file)

	return model

if __name__ == '__main__':
	get_latest('ED Baseline Forecast')
