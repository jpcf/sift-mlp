import numpy as np

def relu(x):
        return np.amax(np.concatenate((np.zeros_like(x), x), axis=1), axis=1, keepdims=True)

def reluPrime(x):
    # This is just the Heaviside function
    return 0.5*(np.sign(x) + 1)

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def sigmoidPrime(x):
    return sigmoid(x)*(1-sigmoid(x) )

class Mlp:

    def __init__(self, inputDim, hidDim, outDim, learnRate):
            self.inputDim = inputDim
            self.hidDim   = hidDim
            self.outDim   = outDim
            self.learnRate= learnRate
            self.hidW     = np.random.rand(hidDim, inputDim)
            self.outW     = np.random.rand(outDim, hidDim)
            self.activHid = np.zeros((hidDim,1))
            self.activOut = np.zeros((outDim,1))
            # The Delta Matrices
            self.deltaO   = np.zeros((outDim,1))
            self.deltaH   = np.zeros((hidDim,1))
            self.deltaX   = np.zeros((inputDim,1))
            self.prevX    = np.zeros((inputDim,1))

    def forwardPropagate(self,inputVec):
            self.activHid = np.dot(self.hidW, inputVec)
            self.activOut = np.dot(self.outW, relu(self.activHid))
            self.prevX    = inputVec
            return relu(self.activOut)

    def backPropagate(self, trainingPattern):
            # Backpropagating the deltas, the trainingPatters must have the same shape as the activation output
            self.deltaO = (relu(self.activOut) - trainingPattern)
            self.deltaH = np.multiply(reluPrime(self.activHid), np.dot(self.outW.T, self.deltaO))
            # The learning step. Note that the gradient is delta(j)*relu(activHid(i))
            self.outW = self.outW - self.learnRate*np.dot(relu(self.activHid), self.deltaO.T).T 
            self.hidW = self.hidW - self.learnRate*np.dot(self.prevX, self.deltaH.T).T
