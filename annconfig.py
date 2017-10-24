import tensorflow as tf
import tflowtools as TFT

########CONFIG########

YEAST_CONFIG = {
	'name': "yeast",
	'epochs': 1000,
	'lrate': "scale",
	'tint': 100,
	'showint': 1000,
	'mbs': 53,
	'wgt_range': (-.3,.3),
	'hidden_layers':[50,50],
	'hidac': (lambda x, y: tf.tanh(x,name=y)),
	'outac': (lambda x, y: tf.nn.softmax(x,name=y)),
	'case_generator': "yeast.txt",
	'stdeviation': True,
	'vfrac': 0.1 ,
	'tfrac': 0.1,
	'cfunc': 'rmse'
}

WINE_CONFIG = {
	'name': "wine",
	'epochs': 1000,
	'lrate': "scale",
	'tint': 100,
	'showint': 1000,
	'mbs': 41,
	'wgt_range': (-.3,.3),
	'hidden_layers':[50,50],
	'hidac': (lambda x, y: tf.tanh(x,name=y)),
	'outac': (lambda x, y: tf.nn.softmax(x,name=y)),
	'case_generator': "winequality_red.txt",
	'stdeviation': True,
	'vfrac': 0.1 ,
	'tfrac': 0.1,
	'cfunc': 'rmse'
}

GLASS_CONFIG = {
	'name': "glass",
	'epochs': 1000,
	'lrate': "scale",
	'tint': 100,
	'showint': 1000,
	'mbs': 107,
	'wgt_range': (-.3,.3),
	'hidden_layers':[50,50],
	'hidac': (lambda x, y: tf.tanh(x,name=y)),
	'outac': (lambda x, y: tf.nn.softmax(x,name=y)),
	'case_generator': "glass.txt",
	'stdeviation': True,
	'vfrac': 0.1 ,
	'tfrac': 0.1,
	'cfunc': 'rmse'
}

MNIST_CONFIG = {
	'name': "mnist",
	'epochs': 1000,
	'lrate': "scale",
	'tint': 100,
	'showint': 1000,
	'mbs': 50,
	'wgt_range': (-.3,.3),
	'hidden_layers':[50,50],
	'hidac': (lambda x, y: tf.tanh(x,name=y)),
	'outac': (lambda x, y: tf.nn.softmax(x,name=y)),
	'case_generator': "mnist.txt",
	'stdeviation': True,
	'vfrac': 0.1 ,
	'tfrac': 0.1,
	'cfunc': 'rmse'
}

IRIS_CONFIG = {
	'name': "iris",
	'epochs': 1000,
	'lrate': "scale",
	'tint': 100,
	'showint': 1000,
	'mbs': 50,
	'wgt_range': (-.3,.3),
	'hidden_layers':[50,50],
	'hidac': (lambda x, y: tf.tanh(x,name=y)),
	'outac': (lambda x, y: tf.nn.softmax(x,name=y)),
	'case_generator': "iris.txt",
	'stdeviation': True,
	'vfrac': 0.1 ,
	'tfrac': 0.1,
	'cfunc': 'rmse'
}

PARITY_CONFIG = {
	'name': "parity",
	'epochs': 1000,
	'lrate': "scale",
	'tint': 100,
	'showint': 100,
	'mbs': 1024,
	'wgt_range': (-1,1),
	'hidden_layers':[50,50],
	'hidac': (lambda x, y: tf.tanh(x,name=y)),
	'outac': (lambda x, y: tf.nn.softmax(x,name=y)),
	'case_generator': (lambda: TFT.gen_all_parity_cases(10)),
	'stdeviation': True,
	'vfrac': 0.1 ,
	'tfrac': 0.1,
	'cfunc': 'rmse'
}

SEGCOUNT_CONFIG = {
	'name': "segcount",
	'epochs': 1000,
	'lrate': "scale",
	'tint': 100,
	'showint': 100,
	'mbs': 100,
	'wgt_range': (-.3,.3),
	'hidden_layers':[50,50],
	'hidac': (lambda x, y: tf.tanh(x,name=y)),
	'outac': (lambda x, y: tf.nn.softmax(x,name=y)),
	'case_generator': (lambda: TFT.gen_segmented_vector_cases(25,1000,0,8)),
	'stdeviation': True,
	'vfrac': 0.1 ,
	'tfrac': 0.1,
	'cfunc': 'rmse'
}

BITCOUNT_CONFIG = {
	'name': "bitcount",
	'epochs': 3000,
	'lrate': "scale",
	'tint': 100,
	'showint': 100,
	'mbs': 100,
	'wgt_range': (-.3,.3),
	'hidden_layers':[50,50],
	'hidac': (lambda x, y: tf.tanh(x,name=y)),
	'outac': (lambda x, y: tf.nn.softmax(x,name=y)),
	'case_generator': (lambda: TFT.gen_segmented_vector_cases(25,1000,0,8)),
	'stdeviation': True,
	'vfrac': 0.1 ,
	'tfrac': 0.1,
	'cfunc': 'rmse'
}




