import numpy as np

def exp_mat_fun(size: int) -> np.array:
    exp_mat = np.array([
        [np.exp(-2j*np.pi/size*i*k)
        for i in range(size)]
        for k in range(size)
        ])
    return exp_mat



def my_ft(f: np.array) -> np.array:
    exp_mat = exp_mat_fun(len(f))
    return exp_mat.dot(f)
