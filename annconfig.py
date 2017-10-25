import tensorflow as tf
import tflowtools as TFT

########CONFIG########

YEAST_CONFIG = {
	'name': "yeast",
	'steps': 53000,
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
	'cfunc': 'rmse',
	'mapbs': 20,
	'cfrac':1.0,
	'map_layers': [],
	'display_wgts': [],
	'display_biases': [],
	'dendrogram_layers': [1]
}

WINE_CONFIG = {
	'name': "wine",
	'steps': 41000,
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
	'cfunc': 'rmse',
	'mapbs': 0,
	'cfrac':1.0,
	'map_layers': [],
	'display_wgts': [],
	'display_biases': [],
	'dendrogram_layers': []
}

GLASS_CONFIG = {
	'name': "glass",
	'steps': 107000,
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
	'cfunc': 'rmse',
	'mapbs': 0,
	'cfrac':1.0,
	'map_layers': [],
	'display_wgts': [],
	'display_biases': [],
	'dendrogram_layers': []
}

MNIST_CONFIG = {
	'name': "mnist",
	'steps': 50000,
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
	'cfunc': 'rmse',
	'mapbs': 0,
	'cfrac':1.0,
	'map_layers': [],
	'display_wgts': [],
	'display_biases': [],
	'dendrogram_layers': []
}

IRIS_CONFIG = {
	'name': "iris",
	'steps': 50000,
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
	'cfunc': 'rmse',
	'mapbs': 0,
	'cfrac':1.0,
	'map_layers': [],
	'display_wgts': [],
	'display_biases': [],
	'dendrogram_layers': []
}

PARITY_CONFIG = {
	'name': "parity",
	'steps': 1024000,
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
	'cfunc': 'rmse',
	'mapbs': 20,
	'cfrac':1.0,
	'map_layers': [],
	'display_wgts': [],
	'display_biases': [],
	'dendrogram_layers': []
}

SEGCOUNTER_CONFIG = {
	'name': "segcounter",
	'steps': 100000,
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
	'cfunc': 'rmse',
	'mapbs': 20,
	'cfrac':1.0,
	'map_layers': [],
	'display_wgts': [],
	'display_biases': [],
	'dendrogram_layers': []
}

BITCOUNTER_CONFIG = {
	'name': "bitcounter",
	'steps': 200000,
	'lrate': "scale",
	'tint': 100,
	'showint': 100,
	'mbs': 50,
	'wgt_range': (-.1,.1),
	'hidden_layers':[50,50],
	'hidac': (lambda x, y: tf.tanh(x,name=y)),
	'outac': (lambda x, y: tf.nn.softmax(x,name=y)),
	'case_generator': (lambda: TFT.gen_vector_count_cases(500,15)),
	'stdeviation': True,
	'vfrac': 0.1 ,
	'tfrac': 0.1,
	'cfunc': 'rmse',
	'mapbs': 20,
	'cfrac':1.0,
	'map_layers': [],
	'display_wgts': [],
	'display_biases': [],
	'dendrogram_layers': []
}

AUTOENCODER_CONFIG = {
	'name': "autoencoder",
	'steps': 300000,
	'lrate': "scale",
	'tint': 100,
	'showint': 100,
	'mbs': 100,
	'wgt_range': (-.3,.3),
	'hidden_layers':[50,50],
	'hidac': (lambda x, y: tf.tanh(x,name=y)),
	'outac': (lambda x, y: tf.nn.softmax(x,name=y)),
	'case_generator': (lambda: TFT.gen_all_one_hot_cases(8)),
	'stdeviation': True,
	'vfrac': 0.1 ,
	'tfrac': 0.1,
	'cfunc': 'rmse',
	'mapbs': 0,
	'cfrac':1.0,
	'map_layers': [],
	'display_wgts': [],
	'display_biases': [],
	'dendrogram_layers': []
}






