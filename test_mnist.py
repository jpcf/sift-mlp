import Mlp
import numpy as np
import random
import csv
import utils
from sklearn import preprocessing

# Parameters
NUM_EPOCH  = 100
TRAIN_SAMP = 42000
TEST_SAMP  = 28000
N_PIXELS   = 784

# The Multilayer Perceptron Layer
ann = Mlp.Mlp(784, 100, 10, 0.1)

# --------------------------------- LOADING DATA ---------------------------------- #

# Read the dataset from files
dataTrain = np.zeros((TRAIN_SAMP, N_PIXELS))
dataTest  = np.zeros((TEST_SAMP,9))
classTrain = np.zeros((TRAIN_SAMP, 1), dtype=np.uint8)
classTest  = np.zeros((TEST_SAMP, 1), dtype=np.uint8)
f_csvTrain = open("./datasets/MNIST/train.csv", "r")
f_csvTest =  open("./datasets/MNIST/test.csv", "r")

csvreaderTrain = csv.reader(f_csvTrain, delimiter=",")
csvreaderTest  = csv.reader(f_csvTest,  delimiter=",")

i = 0
print("Loading Training Samples to memory...")
for row in csvreaderTrain:
	if (i != 0):
		# Load pixel values for that row, as well as class
		dataTrain[i-1,:]  = row[1:785]
		classTrain[i-1,:] = int(row[0])
		
		# Normalize data between 0 and 1
		dataTrain[i-1,:] /= 255
		
		# Print progress..
		if( i % 1000 == 0) : print("{0} training samples loaded...".format(i))

	i = i + 1

# Training Vector
print("Encoding Labels in One-Hot...")
classes = utils.oneHotEncode(classTrain, 10)
correctClassif = np.argmax(classes, axis=1)


# --------------------------------- TRAINING NETWORK ---------------------------------- #



for n in range(NUM_EPOCH):

	# Resetting the errors to zero
	numErr = 0

	# Training the network...
	for i in range(TRAIN_SAMP):
		prediction = ann.trainBP(dataTrain[i,:].reshape((N_PIXELS,1)), classes[i,:].reshape(10,1))
		if ( np.argmax(prediction) != correctClassif[i]):
			numErr += 1

		# Print progress...
		if( i % 10000 == 0) : print("{0} training samples processed...".format(i))

	# n = random.randint(0,DATASET_SZ-1)

	# Printing progress
	print("Training Epoch {0}".format(n))	
	print("Average error: ", 100*numErr/(TRAIN_SAMP))



