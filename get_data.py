import os
import time
#import pandas as pd
import numpy as np
import scipy.io
import scipy.linalg

def get_integral_equation_matrix(k):
    n = k
    t = np.array([2*i*np.pi*n**-1 for i in range(n)])
    r = [(2.5+ np.cos(t[i]*3))**0.5 for i in range(n)]
    Ax = [np.cos(t[i])*r[i] for i in range(n)]
    Ay = [r[i]*np.sin(t[i]) for i in range(n)]
    Bx = [3*np.cos(t[i]) for i in range(n)]
    By = [3*np.sin(t[i]) for i in range(n)]

    def u_prime(t):
        return (2.5 + np.cos(3*t))**(1/2) * (-1 * np.sin(t)) - np.cos(t) * 3 * np.sin(3*t) * (2.5 + np.cos(3*t))**-0.5
    def v_prime(t):
        return (2.5 + np.cos(3*t))**(1/2) * (np.cos(t)) - np.sin(t) * 3 * np.sin(3*t) * (2.5 + np.cos(3*t))**-0.5
    ut = u_prime(t)
    vt = v_prime(t)


    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i][j] = (0.5 * np.log((Bx[i] - Ax[j])**2 + (By[i] - Ay[j])**2)) * np.sqrt(ut[j]**2 + vt[j]**2)
    return mat

def get_shaw_data():
    shaw_file_name = 'shaw_1000/shaw_1000.mtx'
    mat = scipy.io.mmread(shaw_file_name)
    mat = mat.toarray()
    return mat

def get_shaw_data_1024():
    shaw_file_name = 'shaw_1000/shaw_1000.mtx'
    mat = scipy.io.mmread(shaw_file_name)
    mat = mat.toarray()
    mat = np.concatenate((mat, np.zeros((1000, 24))), axis = 1)
    mat = np.concatenate((mat, np.zeros((24, 1024))), axis = 0)
    mat = np.random.permutation(mat)
    mat = np.random.permutation(mat.T).T
    return mat

def get_gravity_data():
    gravity_file_name = 'gravity_1000/gravity_1000.mtx'
    mat = scipy.io.mmread(gravity_file_name)
    mat = mat.toarray()
    return mat

def get_gravity_data_1024():
    gravity_file_name = 'gravity_1000/gravity_1000.mtx'
    mat = scipy.io.mmread(gravity_file_name)
    mat = mat.toarray()
    mat = np.concatenate((mat, np.zeros((1000, 24))), axis = 1)
    mat = np.concatenate((mat, np.zeros((24, 1024))), axis = 0)
    mat = np.random.permutation(mat)
    mat = np.random.permutation(mat.T).T
    return mat

def get_cauchy_matrix(n):
    a, b = 0, 100
    c, d = 100, 200
    x = a + (b-a)*np.random.rand(n)
    y = c + (d-c)*np.random.rand(n)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i, j] = (x[i] - y[j])**-1
    return mat


def get_fast_decay_matrix(n, r):
    u, _, v= scipy.linalg.svd(np.random.randn(n, n))
    sig = np.array([1]*r + [0.5**(i+1) for i in range(n-r)])
    return u.dot(np.diag(sig)).dot(v)

def get_slow_decay_matrix(n, r):
    u, _, v= scipy.linalg.svd(np.random.randn(n, n))
    sig = np.array([1]*r + [(2+i)**-2 for i in range(n-r)])
    return u.dot(np.diag(sig)).dot(v)

def main():
    start = time.time()
    # get_jester_data()
    # get_gisette_data()
    # get_dexter_data()
    get_integral_equation_matrix(2048)
    get_shaw_data_1024()
    get_gravity_data_1024()
    get_cauchy_matrix(2048)
    get_fast_decay_matrix(1024, 10)
    get_slow_decay_matrix(1024, 25)
    end = time.time()
    print( str( end - start) + ' seconds')

if __name__ == '__main__':
    main()
