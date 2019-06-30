import numpy as np
#x :- training set
#y :- labels
#T :- threshold for perceptron repetition
def percep (x, y, T):
    theta = np.array([0,0])
    #theta0 :- theta-not
    theta0 = 0
    progression = []
    mistakes = 0
    for t in range(T):
        for i in range(3):
            if y[i]*np.dot(theta,x[i]) <= 0:
                mistakes += 1
                theta = theta + y[i]*x[i]
                theta0 = theta + y[i]
                progression.append(theta)
    return mistakes,progression,theta,theta0
