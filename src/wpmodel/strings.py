class errors:
	NO_DATA_DICT = '''No queries defined, can not populate the data dict!
	Either pass a query list here or set the class query list first!
	'''
	NO_DATA_DEFINED = '''This key has not been registered in the
	data dictionary
	'''
	DATA_NOT_CACHED = '''Some query list entries are not in the precache list.
	Data retrieval may be slow
	'''

DEFAULT_MODEL_NAME = 'WPH Model'

MAGIC_DATA = dict(
	WEATHER_QUERY_KEY='weather'
)
