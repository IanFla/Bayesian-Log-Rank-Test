import numpy as np
import scipy.stats as st
from BLRT2.sampling.ritd import RightDirichlet
from BLRT2.sampling.itvd import IntervalDirichlet
from BLRT2.others.distribution import sur_null
from datetime import datetime as dt
import statsmodels.api as sm

import os
import multiprocessing
import pickle


def right(num, sizeX, sizeB, M):
    survival = sur_null()[num]
    T = survival.rvs(size=sizeX)
    C = st.expon().rvs(size=sizeX)
    C[C >= 1.0] = 1.0
    D = (T <= C)
    X = T * D + C * (1 - D)
    # print('censoring rate: {} (=1: {})'.format(1 - D.mean(), np.mean(X == 1)))

    start = dt.now()
    rd = RightDirichlet(X=X, D=D, size=sizeB, alpha=M)
    rd.prior(a=4, b=4)
    rd.imputation()
    rd.sampling(np.linspace(0, 1, 1000))
    end = dt.now()
    return rd, end - start


def interval(num, sizeX, sizeB, M):
    survival = sur_null()[num]
    L = survival.rvs(size=sizeX)
    C = st.expon().rvs(size=sizeX)
    C[C >= 1.0] = 1.0
    D = (L <= C)
    L = L * D + C * (1 - D)
    R = L.copy()
    R[D == 0] = np.inf
    # print('censoring rate: {} (=1: {})'.format(1 - D.mean(), np.mean(L == 1)))

    start = dt.now()
    td = IntervalDirichlet(L=L, R=R, size=sizeB, alpha=M)
    td.imputation()
    td.sampling(np.linspace(0, 1, 1000))
    end = dt.now()
    return td, end - start


def divi(num, den):
    den[num == 0] = 1
    return num / den


def neff(arr):
    n = len(arr)
    acf = sm.tsa.acf(arr, nlags=100, fft=True, adjusted=True)
    sums = 0
    for k in range(1, len(acf)):
        sums = sums + (n - k) * acf[k] / n

    return n / (1 + 2 * sums)


def statistic(seed, num, mode, sizeX, sizeB, M):
    if mode == 0:
        np.random.seed(seed)
        d1, t1 = right(num, sizeX, sizeB, M)
        np.random.seed(19971107 + seed)
        d2, t2 = right(num, sizeX, sizeB, M)
        X1 = d1.X
        X2 = d2.X
    else:
        np.random.seed(seed)
        d1, t1 = interval(num, sizeX, sizeB, M)
        np.random.seed(19971107 + seed)
        d2, t2 = interval(num, sizeX, sizeB, M)
        X1 = d1.L
        X2 = d2.L

    grid_x = np.linspace(0.0, 1, 1000)
    N1 = np.sum(grid_x[:-1].reshape([-1, 1]) <= X1, axis=1)
    N2 = np.sum(grid_x[:-1].reshape([-1, 1]) <= X2, axis=1)
    NN = N1 * N2 / (N1 + N2)
    H1 = divi(d1.S[:, 1:] - d1.S[:, :-1], d1.S[:, :-1])
    H2 = divi(d2.S[:, 1:] - d2.S[:, :-1], d2.S[:, :-1])
    Q1 = np.sum(NN * H1, axis=1)
    Q2 = np.sum(NN * H2, axis=1)

    t1 = t1.seconds + t1.microseconds * 1e-6
    t2 = t2.seconds + t2.microseconds * 1e-6
    n1 = neff(Q1)
    n2 = neff(Q2)
    return [[n1, n2], [t1, t2]]


def run(it):
    Ms = [0.001, 10, 100, 1000]
    result = []
    for M in Ms:
        print(it, M)
        res = [
            statistic(it, 0, 0, 100, 10000, M),
            statistic(it, 0, 1, 100, 10000, M)
        ]
        result.append(res)

    return result


def main():
    # os.environ['OMP_NUM_THREADS'] = '1'
    # with multiprocessing.Pool(processes=32) as pool:
    #     begin = dt.now()
    #     its = np.arange(100)
    #     R = pool.map(run, its)
    #     end = dt.now()
    #     print((end - begin).seconds)

    R = []
    for it in range(20):
        R.append(run(it))

    with open('../data/efficiency2', 'wb') as file:
        pickle.dump(R, file)


if __name__ == '__main__':
    main()
