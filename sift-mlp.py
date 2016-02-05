import cv2
import Mlp
import numpy as np
import csv
import time
import pickle

# Creates the SIFT object
sift = cv2.xfeatures2d.SIFT_create()

# The number of clusters, k
k = 2000 
termCrit = cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 10, 0.001
flags = cv2.KMEANS_PP_CENTERS
attempts = 3
iteration = 0

# The BOW K-means trainer
bow = cv2.BOWKMeansTrainer(k, termCrit, attempts, flags)

# The Multilayer Perceptron
mlp = Mlp.Mlp(k, 5*k, 17, 0.02)

startT = time.time()

with open('training_solutions_rev1.csv', 'rt') as csvfile:

    # Reads from the solution file
    lines = csv.reader(csvfile)
    next(lines)

    for line in lines:
        iteration += 1
        
        # Converts the image to grayscale
        img = cv2.imread('images_training_rev1/{}.jpg'.format(line[0]), cv2.IMREAD_GRAYSCALE)
        
        # Detecting and computing the descriptors
        keypoint = sift.detect(img, None)
        _, desc = sift.compute(img, keypoint, None)

        # Adds the descriptor to the BoW trainer
        bow.add(desc)

        if (iteration == 100) : 
            print("Iteration: ", iteration, "Elapsed Time: ", time.time() - startT)
            break


# Performs the Clustering operation and pickles it, in case we need it afterwards
print("Performing the Clustering...")
vocabulary = bow.cluster()
pickle.dump(vocabulary, open("vocab.p", "wb"))

# Now, we get the descriptors for the images, according to the vocabulary

bowExtract = cv2.BOWImgDescriptorExtractor( cv2.xfeatures2d,SiftDescriptorExtractor, cv2.FlannBasedMatcher)
bowExtract.setVocabulary(vocabulary)
