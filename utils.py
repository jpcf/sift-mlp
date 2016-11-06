import numpy as np

def oneHotEncode(classNumArr, classNum):
	
	# Creates the array of one hot encoded outputs
	oneHotClassArr = np.zeros((classNumArr.shape[0], classNum))

	for i in range(classNumArr.shape[0]):
		oneHotClassArr[i,classNumArr[i,:]] = 1

	return oneHotClassArr
	
	
