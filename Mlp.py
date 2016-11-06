import numpy as np
from numba import jit

@jit
def relu(x):
	return np.maximum(x, 0, x)
@jit
def reluPrime(x):
	# This is just the Heaviside function
	return 0.5*(np.sign(x) + 1)
@jit
def sigmoid(x):
	return 1.0/(1+np.exp(-x))
@jit
def sigmoidPrime(x):
	return sigmoid(x)*(1-sigmoid(x) )

class Mlp:
	
	def __init__(self, inputDim, hidDim, outDim, learnRate):
		self.inputDim = inputDim
		self.hidDim   = hidDim
		self.outDim   = outDim
		self.learnRate= learnRate
		self.hidW     = np.random.rand(hidDim, inputDim+1)*2 -1 
		self.outW     = np.random.rand(outDim, hidDim+1)*2 -1
		self.activHid = np.zeros((hidDim,1))
		self.activOut = np.zeros((outDim,1))
		# The Delta Matrices
		self.deltaO   = np.zeros((outDim,1))
		self.deltaH   = np.zeros((hidDim,1))
		self.deltaX   = np.zeros((inputDim,1))

	@jit
	def forwardPropagate(self,inputPattern):
		self.activHid = np.dot(self.hidW, np.concatenate((inputPattern, np.array([[1]]))) )           
		self.activOut = np.dot(self.outW, sigmoid( np.concatenate((self.activHid, np.array([[1]])) ) ) )
		return sigmoid(self.activOut)
	
	@jit
	def trainBP(self, inputPattern, trainingPattern):
		#Perform forward propagation first
		self.activHid = np.dot(self.hidW, np.concatenate((inputPattern, np.array([[1]]))) )           
		self.activOut = np.dot(self.outW, sigmoid( np.concatenate((self.activHid, np.array([[1]])) ) ) )

		# Backpropagating the deltas, the trainingPatters must have the same shape as the activation output
		self.deltaO = (sigmoid(self.activOut) - trainingPattern)
		self.deltaH = np.multiply(sigmoidPrime(self.activHid), np.dot(self.outW[:,0:self.hidDim].T, self.deltaO))

		# The learning step. Note that the gradient is delta(j)*relu(activHid(i))
		self.outW -= self.learnRate*np.dot(self.deltaO, np.concatenate( (sigmoid(self.activHid), np.array([[1]])) ).T )
		self.hidW -= self.learnRate*np.dot(self.deltaH, np.concatenate( (sigmoid(inputPattern), np.array([[1]])) ).T )
	
		return sigmoid(self.activOut)
