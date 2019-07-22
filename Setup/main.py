import numpy as np

def randomization(n):
    A = np.random.rand(n,1)
    raise NotImplementedError

def operations(h, w):
    A = np.random.rand(h,w)
    B = np.random.rand(h,w)
    return A,B,A+B
    raise NotImplementedError


def norm(A, B):
    s = np.linalg.norm(A+B)
    return s
    raise NotImplementedError


def neural_network(inputs, weights):
    x = inputs
    w = np.transpose(weights)
    a = np.matmul(w,x)
    return (np.tanh(a))
    raise NotImplementedError
