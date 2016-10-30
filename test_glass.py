import Mlp
import numpy as np
import random
import csv

# Parameters
NUM_EPOCH = 10000
TEST_SAMP = 20
DATASET_SZ  = 214

# The Multilayer Perceptron Layer
ann = Mlp.Mlp(9, 50, 7, 0.2)

# Read the dataset from file
dataset = np.zeros((214, 9))
classes = np.zeros((214, 1), dtype=np.uint8)
csvfile = open("./datasets/Glass/glass.data", "r")
csvreader = csv.reader(csvfile, delimiter=",")

for row in csvreader:
	dataset[int(row[0])-1,:] = row[1:10]
	classes[int(row[0])-1,:] = int(row[10])

trainVec = np.zeros((7,1))

for n in range(NUM_EPOCH):

	print("Training Epoch {0}".format(n))	

	# Training the network...
	for i in range(DATASET_SZ):
		n = random.randint(0,DATASET_SZ-1)
		ann.forwardPropagate(dataset[n,:].reshape((9,1)))
		trainVec[classes[n,:]-1] = 1
		#print(trainVec)
		ann.backPropagate(trainVec)
		trainVec[classes[n,:]-1] = 0

	# Testing the new parameters
	for i in range(TEST_SAMP):
		n = random.randint(0,DATASET_SZ-1)
		a = dataset[n].reshape((9,1))
		#print(a)
		output = ann.forwardPropagate(a)
		print(output, classes[n,:])
