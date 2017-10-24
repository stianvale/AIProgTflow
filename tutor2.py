import tensorflow as tf
import numpy as np
import matplotlib.pyplot as PLT
import tflowtools as TFT
import math

# **** Autoencoder ****
# We can extend the basic approach in tfex8 (tutor1.py) to handle a) a 3-layered neural network, and b) a collection
# of cases to be learned.  This is a specialized neural network designed to solve one type of classification
#  problem: converting an input string, through a single hidden layer, to a copy of itself on the output end.

class autoencoder():

    # nh = # hidden nodes (in the single hidden layer)
    # lr = learning rate

    def __init__(self,nh=3,lr=.1,mbs=100,cman=None):
        self.learning_rate = lr
        self.num_hiddens = nh
        self.global_step = 0
        self.mbs = mbs
        self.cases = cman.get_cases()
        self.correct_percent = 0
        self.build_neural_network(nh,mbs)

    def build_neural_network(self,nh,mbs):
        ios = 8 # ios = input- and output-layer size
        nh = 50
        nh2 = 50
        #nh3 = 100
        out_size = 10
        low = -.3
        high = .3

        self.w1 = tf.Variable(np.random.uniform(low,high,size=(ios,nh)),name='Weights-1')  # first weight array
        self.w2 = tf.Variable(np.random.uniform(low,high,size=(nh,nh2)),name='Weights-2') # second weight array
        self.w3 = tf.Variable(np.random.uniform(low,high,size=(nh2,out_size)),name='Weights-3')
        self.b1 = tf.Variable(np.random.uniform(low,high,size=nh),name='Bias-1')  # First bias vector
        self.b2 = tf.Variable(np.random.uniform(low,high,size=nh2),name='Bias-2')  # Second bias vector
        self.b3 = tf.Variable(np.random.uniform(low,high,size=out_size),name='Bias-3')
        self.input = tf.placeholder(tf.float64,shape=(None,ios),name='Input')
        self.target = tf.placeholder(tf.float64,shape=(None,out_size),name='Target')
        self.hidden = tf.tanh(tf.matmul(self.input,self.w1) + self.b1,name="Hiddens")
        self.hidden2 = tf.tanh(tf.matmul(self.hidden,self.w2) + self.b2,name="Hiddens")
        self.output = tf.nn.softmax(tf.matmul(self.hidden2,self.w3) + self.b3, name = "Outputs")



        self.error = tf.reduce_mean(tf.square(self.target - self.output),name='MSE')
        #self.error = tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.target, name="xEnt")
        self.predictor = self.output  # Simple prediction runs will request the value of outputs
        # Defining the training operator

        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate*(1-self.correct_percent**6))
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
        if test_interval: self.avg_vector_distances = []
        self.current_session = sess = TFT.gen_initialized_session()
        for i in range(epochs):
            error = 0
            grabvars = [self.error]
            step = self.global_step + i
            ncases = len(self.cases); nmb = math.ceil(ncases/mbs)
            for cstart in range(0,ncases,mbs):
                cend = min(ncases,cstart+mbs)
                minibatch = self.cases[cstart:cend]
                feeder = {self.input: [c[0] for c in minibatch], self.target: [c[1] for c in minibatch]}
                _,grabvals,_ = self.run_one_step([self.trainer],grabvars,step=step,show_interval=show_interval,session=sess,feed_dict=feeder)
                error += grabvals[0]

            errors.append(error)
            if (test_interval and i % test_interval == 0):
                self.do_testing(sess,scatter=False,mbs=mbs)
        PLT.figure()
        TFT.simple_plot(errors,xtitle="Epoch",ytitle="Error",title="")
        if test_interval:
            PLT.figure()
            TFT.simple_plot(self.avg_vector_distances,xtitle='Epoch',
                              ytitle='Avg Hidden-Node Vector Distance',title='')


    # This particular testing is ONLY called during training, so it always receives an open session.
    def do_testing(self,session=None,scatter=True, mbs=100):
        sess = session if session else self.current_session
        hidden_activations = []
        grabvars = [self.hidden, self.predictor]
        inputs = [c[0] for c in self.cases]
        predictions = []
        ncases = len(self.cases)
        for cstart in range(0,ncases,mbs):
            cend = min(ncases,cstart+mbs)
            minibatch = self.cases[cstart:cend]

            feeder = {self.input: [c[0] for c in minibatch]}
            _,grabvals,_ = self.run_one_step([self.predictor],grabvars,session=sess,
                                             feed_dict = feeder,show_interval=None)
            for val in grabvals[1]:
                predictions.append(val)
            hidden_activations.append(grabvals[0][0])
        if scatter:
            PLT.figure()
            vs = hidden_activations if self.num_hiddens > 3 else TFT.pca(hidden_activations,2)
            TFT.simple_scatter_plot(hidden_activations,radius=8)
        #self.do_testing2(sess=sess)

#for counter
        # correct = 0
        # for i, pred in enumerate(predictions):
        #     if(np.argmax(pred) == sum(inputs[i])):
        #         correct += 1

        # print("Score: "+str(100*correct/len(predictions))+"%")

        # print(predictions[0:1])
        # print(inputs[0:1])

        # return hidden_activations

#for glass
        
        correct = 0
        for i, pred in enumerate(predictions):

            if(np.argmax(pred) == np.argmax(self.cases[i][1])):
                correct += 1

        print("Score: "+str(100*correct/len(predictions))+"%")
        self.correct_percent = correct/len(predictions)

        print(self.cases[-50][0])
        print(predictions[-50])
        print(self.cases[-50][1])

        return hidden_activations


    def do_testing2(self,sess,msg='Testing',bestk=1):
        inputs = [c[0] for c in self.cases]; targets = [c[1] for c in self.cases]
        #TFT.dendrogram(inputs, targets)
        feeder = {self.input: inputs, self.target: targets}
        if bestk is not None:
            self.test_func = self.gen_match_counter(self.predictor,[TFT.one_hot_to_int(list(v)) for v in targets],k=bestk)
        testres, grabvals, _ = self.run_one_step(self.test_func, self.grabvars, self.probes, session=sess,
                                           feed_dict=feeder,  show_interval=None)
        if bestk is None:
            print('%s Set Error = %f ' % (msg, testres))
        else:
            print('%s Set Correct Classifications = %f %%' % (msg, 100*(testres/len(cases))))
        return testres  # self.error uses MSE, so this is a per-case value when bestk=None

    def gen_match_counter(self, logits, labels, k=1):
        print(logits)
        labels = tf.convert_to_tensor([labels])
        print(labels)
        init = tf.global_variables_initializer()

        sess = tf.Session()
        sess.run(init)
        v = sess.run(labels)
        print(v) # will show you your variable.


class Caseman():

    def __init__(self,cfunc,sep=",",vfrac=0,tfrac=0, stdeviation=True):
        self.casefunc = cfunc
        self.validation_fraction = vfrac
        self.test_fraction = tfrac
        self.training_fraction = 1 - (vfrac + tfrac)
        self.stdeviation = stdeviation
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

        if (filestring == "mnist.txt"):
            input_string = input_string[0:600]

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

def autoex1(epochs=1000000,num_bits=15,lrate=1,tint=100,showint=10000,mbs=53):
    #case_generator = (lambda: TFT.gen_all_parity_cases(10))
    #case_generator = (lambda: TFT.gen_segmented_vector_cases(25,1000,0,8))
    #case_generator = (lambda: TFT.gen_vector_count_cases(500,15))
    case_generator = "yeast.txt"
    cman = Caseman(cfunc=case_generator, sep=",", vfrac=0, tfrac=0, stdeviation=True)
    ann = autoencoder(nh=num_bits,lr=lrate,cman=cman, mbs=mbs)
    PLT.ion()
    ann.do_training(epochs,test_interval=tint,show_interval=showint,mbs=mbs)
    ann.do_testing(scatter=True,mbs=mbs)  # Do a final round of testing to plot the hidden-layer activation vectors.
    PLT.ioff()
    TFT.close_session(ann.current_session, False)
    return ann

autoex1()


