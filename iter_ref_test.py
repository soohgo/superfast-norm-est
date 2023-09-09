import numpy as np
import os
import time
from math import log, ceil
import scipy.linalg
import matplotlib.pyplot as plt
import get_data
from hadamardTransform import abridgedHadamardTransform

def iter_ref_res(M, niter, r, d, multiplier_type = 'Gaussian'):
    u, s, v = scipy.linalg.svd(M)
    opt_err = scipy.linalg.norm(M - u[:, :r].dot(np.diag(s[:r]).dot(v[:r, :])))
    del u,s,v
    Mhat = np.zeros(M.shape)
    result = []
    for i in range(niter):
        F = get_multiplier(2*d, M.shape[0], multiplier_type)
        H = get_multiplier(M.shape[1], d, multiplier_type)
        Y = (M - Mhat).dot(H)
        W = F.dot(M - Mhat)
        Q,_ = scipy.linalg.qr(Y, mode = 'economic')
        U,T = scipy.linalg.qr(F.dot(Q), mode = 'economic')
        X = scipy.linalg.pinv(T).dot(U.T.dot(W))
        Mhat = Mhat + Q.dot(X)
        # this step too control the rank growth by 
        # trimming the trailing singular values periodically
        # not necessary for only a few iterations
        # and the current implementation is NOT efficient!
        if (i+1)%9 == 0:
            u, sig, v = scipy.linalg.svd(Mhat)
            Mhat = u[:, :r].dot(np.diag(sig[:r]).dot(v[ :r, :]))
        result.append( scipy.linalg.norm(M - Mhat)/opt_err)
    return result

        
def get_multiplier(a, b, multiplier_type):
    if multiplier_type == 'Gaussian':
        if a > b:
            q, _ = scipy.linalg.qr(np.random.randn(a, b), mode = 'economic')
        else:
            q, _ = scipy.linalg.qr(np.random.randn(b, a), mode = 'economic')
            q = q.T
    else:
        x, y = a, b
        if a < b: 
            x, y = b, a
        dep = min(3, int(log(x, 2)))
        rand = np.random.choice(x, y, replace = False)
        subperm = np.zeros((x, y))
        for j in range(y):
            subperm[rand[j], j] = 1
        q = abridgedHadamardTransform(subperm, dep)
        if a < b:
            q = q.T
    return q


def test_wrapper(repeat, M, niter, r, d, result_file_name, multiplier_type = 'Gaussian'):
    result = np.zeros(niter)
    for i in range(repeat):
        temp = iter_ref_res(M, niter, r, d, multiplier_type)
        temp = np.array(temp).reshape(-1)
        result += repeat**-1*temp
    with open(result_file_name, 'w+') as f:
        temp = [str(a) for a in result]
        f.write(','.join(temp))
    return result