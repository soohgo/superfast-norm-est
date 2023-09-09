import numpy as np
import ctypes
import time


ht = ctypes.CDLL('had_tran.so').hadamard_transform
ht.argtype = (ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double))
aht = ctypes.CDLL('had_tran.so').abridged_hadamard_transform
aht.argtype = (ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double))

def hadamardTransform(mat, scaling = True):
    """
    mat must be a numpy 2-d array
    return H dot mat
    """
    try:
        m, n = mat.shape
    except Exception as e:
        raise e

    if scaling:
        a = (ctypes.c_double * (m*n))(*(m**-0.5*mat.reshape(-1)))
    else:
        a = (ctypes.c_double * (m*n))(*mat.reshape(-1))

    if ht(m, n, a) == 1:
        return np.array(a).reshape(m, n)
    else:
        print('Hadamard Transform Failed.')
        return None

def abridgedHadamardTransform(mat, d, scaling = True):
    """
    mat must be a numpy 2-d array
    return AH dot mat
    """
    assert(type(d) == int)
    try:
        m, n = mat.shape
    except Exception as e:
        raise e

    if scaling:
        a = (ctypes.c_double * (m*n))(*(2**(-0.5*d)*mat.reshape(-1)))
    else:
        a = (ctypes.c_double * (m*n))(*mat.reshape(-1))

    if aht(m, n, d, a) == 1:
        return np.array(a).reshape(m, n)
    else:
        print('Abridged Hadamard Transfrom Failed.')
        return None