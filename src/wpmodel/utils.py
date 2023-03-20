import os
import re
import dill

def get_latest(
	model_name : str,
	model_df : str = 'out'
):
	"""Get the latest version of the requested model_name
	"""

	 latest = max([
	 	f
	 	for f in os.listdir(model_df)
	 	if re.search(f'{model_name}_\d+.pkl', f)
	 ])

	 with open(os.path.join(model_df, latest), 'rb') as file:
        model = dill.load(file)
    
    return model