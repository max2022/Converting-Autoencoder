from branchynet.net import BranchyNet
from branchynet.links import *
import chainer.functions as F
import chainer.links as L
from branchynet import utils, visualize
from chainer import cuda


from networks import lenet_mnist

branchyNet = lenet_mnist.get_network()
if cuda.available:
    branchyNet.to_gpu()
branchyNet.training()


from datasets import mnist
#The MNIST database of handwritten digits, has a training set of 60,000 examples,
# and a test set of 10,000 examples.
TRAIN_BATCHSIZE = 512
TEST_BATCHSIZE = 1

training_size=10000
x_train, y_train, x_test, y_test = mnist.get_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
x_train, y_train, x_test, y_test = x_train[:training_size*6], y_train[:training_size*6], x_test[:training_size], y_test[:training_size]
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


import dill
branchyNet = None
with open("_models/lenet_mnist.bn", "rb") as f:
    branchyNet = dill.load(f)


#set network to inference mode, this is for measuring baseline function. 
branchyNet.testing()
branchyNet.verbose = True

thresholds = [0.05]
#GPU
if cuda.available:
    branchyNet.to_gpu()
g_ts, g_accs, g_diffs, g_exits = utils.screen_branchy(branchyNet, x_test, y_test, thresholds,
                                                    batchsize=TEST_BATCHSIZE, verbose=True)

#convert to ms
g_diffs *= 1000.


print(g_accs)
print(g_diffs)
print(g_ts)
print(g_exits)
#grep numexit lenet.txt | awk -F '[' '{print $2}' | awk -F ']' '{print $1}' > lenet_test.txt
#awk '{if ($1=="0,"){$1="1"}else{$1="0"} print $1}' lenet_test.txt > lenet_final.txt
