import numpy as np
from numpy import pi, cos, sin


def dEdX_complex(dEdF, x):
    rE, iE = dEdF.real, dEdF.imag
    rx, ix = x.real, x.imag
    ax = abs(x)
    re_res = rE * (1/ax - rx**2 / ax**3) + iE * (- (rx*ix) / ax**3)
    im_res = rE * (- (rx*ix) / ax**3) + iE * (1/ax - ix**2 / ax**3)
    return re_res + 1j * im_res

def dEdX_complex_II(dEdF, x):
    return dEdF * (1/abs(x) - x**2 / abs(x)**3)

############# po tialto vymazat
def dEdX(x: np.array(tuple), f, c, y):
    dEdF = dEdF(f, c, y)
    dFdX = dFdX(x)
    dEdX = np.sum(dEdF * dFdX, axis=1)
    return dEdX

def dFdX(x):
    dFdX = map(dfdx, x)
    return np.array(list(dFdX))

def dfdx(x: tuple) -> np.array(np.array):
    x0, x1 = x
    ax = np.sqrt(x0**2 + x1**2)
    f0 = ((1/ax - x0**2 / ax**3), (- (x0*x1) / ax**3))
    f1 = ((- (x0*x1) / ax**3), (1/ax - x1**2 / ax**3))
    return (f0, f1)

def dEdF(f, c, y):
    dEdC = dEdC(c, y)
    dCdF = dCdF(f, c)
    dEdF = sum(my_mul(dEdC, dCdF), axis=(1, 2))
    return dEdF

def dCdF(f, c):
    res = [[[[0, 0], [0, 0]] for _ in range(len(c))] for _ in range(len(f))]
    con = 2*np.pi / len(f)
    for j in range(len(f)):
        for k in range(len(c)):
            for r in range(2):
                for l in range(2):
                    res[j][k][0][0] = np.cos(con * j*k)
                    res[j][k][0][1] = np.sin(con * j*k)
                    res[j][k][1][0] = - np.sin(con * j*k)
                    res[j][k][1][1] = np.cos(con * j*k)
    return res

def dEdC(c, y):
    dEdY = dEdY(c, y)
    dYdC = dYdC(c, y)
    dEdC = dEdY * dYdC

def dEdY():
    pass

def dYdC():
    pass


def my_mul(x, y):
    if len(x[0]) == 1:
        for i in range(len(x)):
            y[i] *= x[i]
    return 

###################### vymazat to hore

def dEdX_real(y: np.array, d: np.array, c: np.array, x: np.array, intensity) -> float:
    dim = len(x)
    dEdX = [[0, 0] for _ in range(dim)]
    for j in range(dim):
        for k in range(dim):
            dEdY = 2 * (y[k] - d[k])
            for r in range(2):
                dYdC = c[k][r] / (norm(c[k]) if not intensity else 1)
                for l in range(2): 
                    arg = 2*pi/dim * j*k
                    dCdF = dCdF_fun(r, l, arg)
                    for m in range(2):
                        dFdX = kronecker(l, m)/norm(x[j]) - x[j][l]*x[j][m] / norm(x[j])**3
                        dEdX[j][m] += dEdY * dYdC * dCdF * dFdX
    return vec_to_complex(dEdX)



def dEdF_real(y: np.array, d: np.array, c: np.array) -> float:
    dim = len(y)
    dEdF = [[0, 0] for _ in range(dim)]
    for j in range(dim):
        for k in range(dim):
            dEdY = 2 * (y[k] - d[k])
            for r in range(2):
                dYdC = c[k][r]
                for l in range(2): 
                    arg = 2*pi/dim * j*k
                    dCdF = dCdF_fun(r, l, arg)
                    dEdF[j][l] += dEdY * dYdC * dCdF
    return vec_to_complex(dEdF)

def dft(x):
    y = np.zeros(np.array(x).shape)
    for j in range(len(x)):
        for k in range(len(x)):
            for r in range(2):
                for l in range(2):
                    arg = 2*pi/len(x) * j*k
                    dCdF = dCdF_fun(r, l, arg)
                    y[j][l] += x[k][r] * dCdF
    return vec_to_complex(y)



def dEdC_real(y: np.array, d: np.array, c: np.array) -> float:
    dim = len(y)
    dEdC = [[0, 0] for _ in range(dim)]
    for k in range(dim):
        dEdY = 2 * (y[k] - d[k])
        for r in range(2):
            dYdC = c[k][r]
            dEdC[k][r] = dEdY * dYdC
    return vec_to_complex(dEdC)

def norm(x: np.array) -> float:
    return np.sqrt(sum(x**2))

def dCdF_fun(r: int, l: int, arg: float) -> float:
    if r == 0:
        if l == 0:
            return cos(arg)
        else:
            return sin(arg)
    if r == 1:
        if l == 0:
            return - sin(arg)
        else:
            return cos(arg)

def kronecker(l: int, m: int):
    return l == m


def complex_to_vec(x_arr):
    return np.array([(np.real(x), np.imag(x)) for x in x_arr])

def vec_to_complex(x_arr):
    return np.array([x[0] + 1j * x[1] for x in x_arr])



if __name__ == "__main__":
    pass