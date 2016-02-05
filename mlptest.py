import Mlp
import numpy as np
import random

MAX_ITER = 10000
ann = Mlp.Mlp(2, 5, 1, 0.02)

for i in range(0, MAX_ITER, 1):
    invec  = np.random.random_integers(0, 1, size=(2.,1.))
    predic = ann.forwardPropagate(invec)
    if( int(predic+0.5) != (invec[0] ^ invec[1])):
        print("Error @ ", i, " ", predic, "!=", (invec[0] ^ invec[1]))
        print(invec)
        print(int(predic+0.5))

    ann.backPropagate(invec[0] ^ invec[1])

    if ( i % 1000 == 0): print("Iteration: ", i)
