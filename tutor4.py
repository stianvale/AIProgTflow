import tensorflow as tf
import numpy as np
import matplotlib.pyplot as PLT
import tflowtools as TFT
import math
import annconfig
import random

# **** Autoencoder ****
# We can extend the basic approach in tfex8 (tutor1.py) to handle a) a 3-layered neural network, and b) a collection
# of cases to be learned.  This is a specialized neural network designed to solve one type of classification
#  problem: converting an input string, through a single hidden layer, to a copy of itself on the output end.

##TODO: mapping
##TODO: dendrograms
##TODO: display weights
##TODO: display biases


class Gann():

    # nh = # hidden nodes (in the single hidden layer)
    # lr = learning rate

    def __init__(   self,lr=.1,mbs=100,cman=None,wgt_range=(-.1,.1), hidden_layers=[], hidac=(lambda x, y: tf.tanh(x,name=y)),
                    outac=(lambda x, y: tf.nn.softmax(x,name=y)), cfunc="rmse"):
        self.cman = cman
        self.cfunc = cfunc
        self.input_size = cman.get_input_size()
        self.output_size = cman.get_output_size()
        self.hidden_layers = hidden_layers
        self.hidac = hidac
        self.outac = outac
        self.learning_rate = lr
        self.global_step = 0
        self.mbs = mbs
        self.wgt_range = wgt_range
        self.training_cases = cman.get_training_cases()
        self.validation_cases = cman.get_validation_cases()
        self.testing_cases = cman.get_testing_cases()
        self.correct_percent = 0
        self.modules = []
        self.build_neural_network(mbs)
        self.mapvars = []
        self.mapvar_figures = []

    def add_module(self, module):
        self.modules.append(module)

    def add_mapvar(self,module_index,type='wgt'):
        self.mapvars.append(self.modules[module_index].getvar(type))
        self.mapvar_figures.append(PLT.figure())

    def build_neural_network(self,mbs):
        tf.reset_default_graph()
        self.input = tf.placeholder(tf.float64,shape=(None,self.input_size),name='Input')
        self.target = tf.placeholder(tf.float64,shape=(None,self.output_size),name='Target')
        self.hidden_layers.append(self.output_size)

        invar = self.input
        insize = self.input_size

        for i, outsize in enumerate(self.hidden_layers):
            gmod = None
            if(i != len(self.hidden_layers)-1):
                gmod = Gannmodule(self, i, invar, insize, outsize, self.hidac, self.wgt_range)
            else:
                gmod = Gannmodule(self, i, invar, insize, outsize, self.outac, self.wgt_range)

            invar = gmod.output; insize = gmod.outsize
        self.output = gmod.output


        self.error = None
        if(self.cfunc == "rmse"):
            self.error = tf.reduce_mean(tf.square(self.target - self.output),name='MSE')
        elif(self.cfunc == "xent"):
            self.error = tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.target, name="xEnt")
        self.predictor = self.output  # Simple prediction runs will request the value of outputs
        # Defining the training operator
        optimizer = None
        if (self.learning_rate == "scale"):
            optimizer = tf.train.GradientDescentOptimizer(1-self.correct_percent**6)
        else:
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

        #optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.trainer = optimizer.minimize(self.error,name='Backprop')

    #  This is the same as quickrun in tutorial # 1, but now it's a method, not a function.

    def run_one_step(self,operators, grabbed_vars=None, dir='probeview',
                  session=None, feed_dict=None, step=1, show_interval=1):
        sess = session if session else TFT.gen_initialized_session(dir=dir)

        results = sess.run([operators, grabbed_vars], feed_dict=feed_dict)
        if show_interval and (step % show_interval == 0):
            TFT.show_results(results[1], grabbed_vars, dir)
        return results[0], results[1], sess


    def do_training(self,epochs=100,test_interval=10,show_interval=50,mbs=100):
        errors = []
        self.val_error = []
        self.train_error = []
        if test_interval: self.avg_vector_distances = []
        self.current_session = sess = TFT.gen_initialized_session()
        for i in range(epochs):
            self.current_epoch = i
            error = 0
            grabvars = [self.error]
            step = self.global_step + i
            ncases = len(self.training_cases); nmb = math.ceil(ncases/mbs)
            for cstart in range(0,ncases,mbs):
                cend = min(ncases,cstart+mbs)
                minibatch = self.training_cases[cstart:cend]
                feeder = {self.input: [c[0] for c in minibatch], self.target: [c[1] for c in minibatch]}
                _,grabvals,_ = self.run_one_step([self.trainer],grabvars,step=step,show_interval=show_interval,session=sess,feed_dict=feeder)
                error += grabvals[0]

            errors.append([i,error])
            if (test_interval and i % test_interval == 0):
                self.do_testing(sess,scatter=False,mbs=len(self.training_cases),testset="training")
                self.do_testing(sess,scatter=False,mbs=len(self.validation_cases),testset="validation")
        self.do_testing(sess,scatter=False,mbs=mbs,testset="testing")


        #TFT.simple_plot(errors,xtitle="Epoch",ytitle="Error",title="")

        TFT.plot_training_history(self.train_error, self.val_error)
        




    # This particular testing is ONLY called during training, so it always receives an open session.
    def do_testing(self,session=None,scatter=True, mbs=100, testset="training"):
        error = 0
        sess = session if session else self.current_session
        hidden_activations = []
        grabvars = [self.input, self.predictor, self.error]

        cases = None
        if(testset == "training"):
            cases = self.training_cases
        elif(testset == "validation"):
            cases = self.validation_cases
        elif(testset == "testing"):
            cases = self.testing_cases

        inputs = [c[0] for c in cases]
        predictions = []
        ncases = len(cases)
        for cstart in range(0,ncases,mbs):
            cend = min(ncases,cstart+mbs)
            minibatch = cases[cstart:cend]

            feeder = {self.input: [c[0] for c in minibatch], self.target: [c[1] for c in minibatch]}
            _,grabvals,_ = self.run_one_step([self.predictor],grabvars,session=sess,
                                             feed_dict = feeder,show_interval=None)
            for val in grabvals[1]:
                predictions.append(val)


            hidden_activations.append(grabvals[0][0])

            error += grabvals[2]



        if(testset == "validation"):
            self.val_error.append([self.current_epoch, error])
        if(testset == "training"):
            self.train_error.append([self.current_epoch, error])
        # if scatter:
        #     PLT.figure()
        #     vs = hidden_activations if self.num_hiddens > 3 else TFT.pca(hidden_activations,2)
        #     TFT.simple_scatter_plot(hidden_activations,radius=8)

        
        correct = 0
        for i, pred in enumerate(predictions):

            if(np.argmax(pred) == np.argmax(cases[i][1])):
                correct += 1

        print(testset.capitalize() + " score: "+str(round(100*correct/len(predictions),2))+"%")
        self.correct_percent = correct/len(predictions)

        print(predictions[-1])
        print(cases[-1][1])


    def do_mapping(self,session=None,scatter=True, mbs=100, testset="mapping", mapbs = 10):
        
        sess = session if session else self.current_session
        grabvars = [self.input, self.predictor]

        grabvars += self.mapvars

        randNum = random.randint(0,len(self.training_cases)-mapbs-1)
        cases = self.training_cases[randNum:randNum+mapbs]

        inputs = [c[0] for c in cases]
        targets = [c[1] for c in cases]
        predictions = []

        feeder = {self.input: inputs, self.target: targets}
        _,grabvals,_ = self.run_one_step([self.predictor],grabvars,session=sess,
                                             feed_dict = feeder,show_interval=None)
        for val in grabvals[1]:
            predictions.append(val)


        fig_index = 0
        names = [x.name for x in grabvars[2:]]
        for grabval in grabvals[2:]:
            TFT.hinton_plot(grabval,fig=self.mapvar_figures[fig_index],title= names[fig_index])
            fig_index += 1
            print(grabvals)
            print(len(self.mapvar_figures))
            print(self.mapvars)
            a = input()

        a = input()







class Gannmodule():

    def __init__(self,ann,index,invariable,insize,outsize,acfunc,wgt_range):
        self.ann = ann
        self.insize=insize  # Number of neurons feeding into this module
        self.outsize=outsize # Number of neurons in this module
        self.input = invariable  # Either the gann's input variable or the upstream module's output
        self.index = index
        self.name = "Module-"+str(self.index)
        self.acfunc = acfunc
        self.wgt_range = wgt_range
        self.modules = []
        self.build()


    def build(self):
        mona = self.name; n = self.outsize
        low_lim = self.wgt_range[0]; up_lim = self.wgt_range[1]
        self.weights = tf.Variable(np.random.uniform(low_lim, up_lim, size=(self.insize,n)),
                                   name=mona+'-wgt',trainable=True) # True = default for trainable anyway
        self.biases = tf.Variable(np.random.uniform(low_lim, up_lim, size=n),
                                  name=mona+'-bias', trainable=True)  # First bias vector
        self.output = self.acfunc(tf.matmul(self.input,self.weights)+self.biases,mona+'-out')
        tf.Print(self.output,[self.output])
        self.ann.add_module(self)

    def getvar(self,type):  # type = (in,out,wgt,bias)
        return {'in': self.input, 'out': self.output, 'wgt': self.weights, 'bias': self.biases}[type]

    # spec, a list, can contain one or more of (avg,max,min,hist); type = (in, out, wgt, bias)
    def gen_probe(self,type,spec):
        var = self.getvar(type)
        base = self.name +'_'+type
        with tf.name_scope('probe_'):
            if ('avg' in spec) or ('stdev' in spec):
                avg = tf.reduce_mean(var)
            if 'avg' in spec:
                tf.summary.scalar(base + '/avg/', avg)
            if 'max' in spec:
                tf.summary.scalar(base + '/max/', tf.reduce_max(var))
            if 'min' in spec:
                tf.summary.scalar(base + '/min/', tf.reduce_min(var))
            if 'hist' in spec:
                tf.summary.histogram(base + '/hist/',var)


class Caseman():

    def __init__(self,cfunc,vfrac=0,tfrac=0, stdeviation=True):
        self.casefunc = cfunc
        self.validation_fraction = vfrac
        self.test_fraction = tfrac
        self.training_fraction = 1 - (vfrac + tfrac)
        self.stdeviation = stdeviation
        sep = ","

        if(cfunc == "winequality_red.txt"):
            sep = ";"

        if(not isinstance(cfunc, str)):
            self.generate_cases()
            self.organize_cases()
        else:
            self.add_cases_from_file(cfunc, sep)

    def generate_cases(self):
        self.cases = self.casefunc()  # Run the case generator.  Case = [input-vector, target-vector]

    def organize_cases(self):
        ca = np.array(self.cases)
        np.random.shuffle(ca) # Randomly shuffle all cases
        separator1 = round(len(self.cases) * self.training_fraction)
        separator2 = separator1 + round(len(self.cases)*self.validation_fraction)
        self.training_cases = ca[0:separator1]
        self.validation_cases = ca[separator1:separator2]
        self.testing_cases = ca[separator2:]

    def add_cases_from_file(self, filestring, sep):
        unique=[]
        count=[0 for i in range(10)]
        with open(filestring) as f:
            input_string = f.readlines()

        self.cases = []

        # if (filestring == "mnist.txt"):
        #     input_string = input_string[0:600]

        for line in input_string:
            line = line.strip()
            output = int(line.split(sep)[-1:][0])
            if (output not in unique):
                unique.append(output)

        unique.sort()

        for line in input_string:
            line = line.strip()
            output = int(line.split(sep)[-1:][0])
            result = [0 for i in unique]
            result[unique.index(output)] = 1
            count[unique.index(output)] += 1

            inpt = [float(i) for i in line.split(sep)[0:-1]]
            self.cases.append([inpt, result])


        self.organize_cases()


        if(self.stdeviation):
            for i in range(len(self.cases[0][0])):
                featurelist = [c[0][i] for c in self.cases]
                std = np.std(featurelist)
                avg = np.mean(featurelist)
                for case in self.cases:
                    case[0][i] = (case[0][i]-avg)/std


    def get_input_size(self): return len(self.cases[0][0])
    def get_output_size(self): return len(self.cases[0][1])
    def get_training_cases(self): return self.training_cases
    def get_validation_cases(self): return self.validation_cases
    def get_testing_cases(self): return self.testing_cases
    def get_cases(self): return self.cases


        #correct = tf.nn.in_top_k(tf.cast(logits,tf.float32), labels, k) # Return number of correct outputs
        #return tf.reduce_sum(tf.cast(correct, tf.int32))

# ********  Auxiliary functions for the autoencoder example *******

def vector_distance(vect1, vect2):
    return (sum([(v1 - v2) ** 2 for v1, v2 in zip(vect1, vect2)])) ** 0.5

def calc_avg_vect_dist(vectors):
    n = len(vectors);
    sum = 0
    for i in range(n):
        for j in range(i + 1, n):
            sum += vector_distance(vectors[i], vectors[j])
    return 2 * sum / (n * (n - 1))

#  A test of the autoencoder

def mainfunc(   steps=1000000,lrate="scale",tint=100,showint=10000,mbs=100, wgt_range=(-.3,.3), hidden_layers=[50,50],
                hidac=(lambda x, y: tf.tanh(x,name=y)), outac=(lambda x, y: tf.nn.softmax(x,name=y)), case_generator = "mnist.txt",
                stdeviation=False, vfrac=0.1, tfrac=0.1, cfunc="rmse", mapbs = 0, map_layers = []):

    cman = Caseman(cfunc=case_generator, vfrac=vfrac, tfrac=tfrac, stdeviation=stdeviation)
    ann = Gann(lr=lrate,cman=cman, mbs=mbs, wgt_range=wgt_range, hidden_layers=hidden_layers, hidac=hidac, outac=outac, cfunc=cfunc)
    PLT.ion()
    epochs = steps/mbs
    ann.do_training(epochs,test_interval=tint,show_interval=showint,mbs=mbs)

    for layer in map_layers:
        if(layer == 0):
            continue
        ann.add_mapvar(layer-1, "out")
    ##OBS: Skal vel ikke vise dendrogram mellom input og output i hele nettverket, men for et ønsket layer
    if(mapbs > 0):
        ann.do_mapping(mbs = mbs, mapbs = mapbs)

    PLT.ioff()
    TFT.close_session(ann.current_session, False)
    return ann

def configAndRun(name):

    key = name.upper() + '_CONFIG'
    myDict = getattr(annconfig, key)

    mainfunc(steps = myDict['steps'], lrate=myDict['lrate'], tint=myDict['tint'], showint=myDict['showint'], mbs=myDict['mbs'],
                wgt_range=myDict['wgt_range'], hidden_layers=myDict['hidden_layers'], hidac=myDict['hidac'], outac=myDict['outac'],
                case_generator=myDict['case_generator'], stdeviation=myDict['stdeviation'], vfrac=myDict['vfrac'], tfrac=myDict['tfrac'],
                cfunc=myDict['cfunc'], mapbs = myDict['mapbs'], map_layers=[1,2,3])

# mainfunc(epochs=1000,lrate="scale",tint=100,showint=10000,mbs=51, wgt_range=(-.1,.1), hidden_layers=[50,50],
#                 hidac=(lambda x, y: tf.tanh(x,name=y)), outac=(lambda x, y: tf.nn.softmax(x,name=y)), case_generator = "yeast.txt",
#                 stdeviation=False, vfrac=0.1, tfrac=0.1, cfunc="rmse")

configAndRun("segcounter")
