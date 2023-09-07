#!/usr/bin/env python3

import os
import time
#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import scipy.io
import scipy.linalg
from scipy.linalg import norm
from scipy.linalg import svd
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD
from get_data import *
from numpy.random import default_rng


class ErrorMatrix:
    def __init__(self):
        #set defaults
        self.approx_rank = 10
        #self.shape = (0, 0)

    def set_approx_rank(self, r):
        self.approx_rank = r

    def get_approx_rank(self):
        return self.approx_rank

    def new(self, data_type, m=None, n=None, r=None, opt=1):
        if data_type == 'integral':
            if n == None: n = 2048
            M = get_integral_equation_matrix(n)
        elif data_type == 'shaw':
            M = get_shaw_data_1024()
            #self.shape = (1024, 1024)
        elif data_type == 'gravity':
            #self.shape = (1024, 1024)
            M = get_gravity_data_1024()
            pass
        elif data_type == 'cauchy':
            if n == None: n = 2048
            M = get_cauchy_matrix(n)
        elif data_type == 'fast_decay':
            if n == None: n = 1024
            if r == None: r = 10
            M = get_fast_decay_matrix(n, r)
        elif data_type == 'slow_decay':
            if n == None: n = 1024
            #if r == None: r = 25
            if r == None: r = 10
            M = get_slow_decay_matrix(n, r)
        elif data_type == 'random':
            if m == None: m = 1024
            if n == None: n = 1024
            #M = scipy.sparse.random(m, n) #default density is 0.01
            #M = scipy.sparse.random(m, n, density=0.1)
            M = scipy.sparse.rand(m, n, density=0.01, format="csr", random_state=42).todense()
        else:
            exit(1)
        
        rho = self.approx_rank

        # --------- option 1-------
        if opt == 1:
            if scipy.sparse.issparse(M):
                U, s, Vh = scipy.sparse.linalg.svds(M, rho)
            else:
                U, s, Vh = svd(M, full_matrices = False)
            s_Trun = s[:rho]
            U_Trun = U[:,:rho]
            Vh_Trun= Vh[:rho,:]
            S_Trun = np.diag(s_Trun)
            M_hat= np.dot(U_Trun,np.dot(S_Trun, Vh_Trun))

        # --------- option 2-------
        elif opt == 2:
            svd_LRA =  TruncatedSVD(n_components = rho, random_state=7)
            svd_LRA.fit(M)
            s_truncated = svd_LRA.singular_values_
            M_hat = svd_LRA.fit_transform(M)

        # --------- option 3-------
        elif opt == 3:
            U, s, Vh = svds(M, rho)
            S= np.diag(s)
            M_hat= np.dot(U, np.dot(S1, Vh))

        else:
            exit(1)
        
        E = M - M_hat
        
        return E

    
class NormEstimator:
    def __init__(self):
        return 
    
    def get_init_vec_uniform(self, n, fill_fcn):
        num_entries = int(fill_fcn(n))
        v = np.zeros(n)
        rand = np.random.choice(n, num_entries, replace=False)
        for i in range(num_entries):
            v[rand[i]] = 1

        return v/num_entries
    
    def get_init_vec_inc(self, n, fill_fcn):
        num_entries = int(fill_fcn(n))
        v = np.zeros(n)
        rand = np.random.choice(n, num_entries, replace=False)
        for i in rand:
            v[i] = ((-1)**i)*(1 + i/(n-1))
        v = v/scipy.linalg.norm(v,1)

        return v

    def estimate_1norm(self, E, fill_fcn, init_mode=0):
    # init mode 0: uniform vector randomly sampled
    # init mode 1: vector increasing 1-to-2 randomly sampled and scaled
        m, n = E.shape

        if init_mode == 0:
            v = self.get_init_vec_uniform(n, fill_fcn)
        elif init_mode == 1:
            v = self.get_init_vec_inc(n, fill_fcn)

        j = None
        loop_ct = 0

        while True:
            loop_ct += 1
            u = np.asarray(np.dot(E, v)).reshape(-1)
            w = np.sign(u)
            x = np.asarray(np.dot(E.T, w)).reshape(-1)

            if j == np.argmax(np.abs(x)):
            # this is the same as the last iter, so we must be stuck in an infinite loop
            # caused by small numerical errors
                #break #return np.sum(np.abs(u)), v
                return np.sum(np.abs(u)), loop_ct

            j = np.argmax(np.abs(x)) #inf-norm of x
            x_nrm = abs(x[j])
            nrm = np.sum(np.abs(u)) #1-norm of u

            if x_nrm <= nrm:
                return nrm, loop_ct

            v = np.zeros(n)
            v[j] = 1

        return nrm, loop_ct
    
    def estimate_norm_infnorm(self, E, fill_fcn, init_mode='uniform'):
        return self.estimate_norm_1norm(E.T, fill_fcn, init_mode)



def test_estimates(data_type, fill_fcn, num_tests=100, init_mode = 0):
    Est = []
    I = []
    ErrMats = ErrorMatrix()
    Estimator = NormEstimator()
    for e in range(num_tests):
        E = ErrMats.new(data_type)
        E_norm = scipy.linalg.norm(E, 1)
        est, num_iters = Estimator.estimate_1norm(E, fill_fcn, init_mode=init_mode)
        if est == 0 and E_norm != 0:
            print("division by 0 error")
            Est.append(np.nan)
        elif np.isclose(E_norm, est):
            Est.append(1)
        else:
            Est.append(E_norm/est)
        I.append(num_iters)

    return Est, I

# usage example:
#
# from math import log2
# fill_fcn = lambda n: int(log2(n))
# test_estimates('cauchy', fill_fcn)
