import tensorflow as tf

########CONFIG########


YEAST_CONFIG = {
	'name': "yeast",
	'epochs': 100,
	'datasrc': "yeast.txt",
	'data_sep': ",",
	'lrate': 0.1,
	'showint':100,
	'mbs': 50,
	'vfrac': 0.1,
	'tfrac': 0.1,
	'sm': False,
	'bestk': 1,
	'hidac': (lambda x, y: tf.nn.relu(x,name=y)),
	'outac': (lambda x, y: tf.nn.softmax(x,name=y)),
	'layerlist': [8,9,10],
	'cfunc': "xent",
	'wgt_range': (-3,3),
	
}

WINE_CONFIG = {
	'name': "wine",
	'epochs': 100,
	'datasrc': "winequality_red.txt",
	'data_sep': ";",
	'lrate': 0.1,
	'showint':100,
	'mbs': 50,
	'vfrac': 0.1,
	'tfrac': 0.1,
	'sm': False,
	'bestk': 1,
	'hidac': (lambda x, y: tf.nn.relu(x,name=y)),
	'outac': (lambda x, y: tf.nn.softmax(x,name=y)),
	'layerlist': [11,9,6],
	'cfunc': "xent",
	'wgt_range': (-3,3),
	
}