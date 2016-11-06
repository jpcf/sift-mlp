import Mlp
import numpy as np
import random
import csv
import utils
from sklearn import preprocessing

# Parameters
NUM_EPOCH = 1000
TEST_SAMP = 50
DATASET_SZ  = 214

# The Multilayer Perceptron Layer
ann = Mlp.Mlp(9, 50, 7, 0.3)

# Read the dataset from file
dataset = np.zeros((214, 9))
classes = np.zeros((214, 1), dtype=np.uint8)
csvfile = open("./datasets/Glass/glass.data", "r")
csvreader = csv.reader(csvfile, delimiter=",")

for row in csvreader:
	dataset[int(row[0])-1,:] = row[1:10]
	classes[int(row[0])-1,:] = int(row[10])

# Normalize data 
dataset = preprocessing.scale(dataset)

# Training Vector
classes = utils.oneHotEncode(classes, 7)
correctClassif = np.argmax(classes, axis=1)+1

# Testing error
numErr = 0

for n in range(NUM_EPOCH):

	# Printing progress
	if(n % 100 == 0) :
		print("Training Epoch {0}".format(n))	
		print("Average error: ", numErr/TEST_SAMP)

		# Resetting the errors to zero
		numErr = 0

	# Training the network...
	for i in range(DATASET_SZ):
		n = random.randint(0,DATASET_SZ-1)
		ann.trainBP(dataset[n,:].reshape((9,1)), classes[n,:].reshape(7,1))

	# Testing the new parameters
	for i in range(TEST_SAMP):
		# Choosing random sample to test
		n = random.randint(0,DATASET_SZ-1)

		# Acessing the prediction error
		if ( (np.argmax(ann.forwardPropagate(dataset[n].reshape((9,1))))+1) != correctClassif[n]):
			#print(output," --- ", classes[n])
			numErr += 1

