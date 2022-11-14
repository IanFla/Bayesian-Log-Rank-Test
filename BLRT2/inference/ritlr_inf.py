import numpy as np
from BLRT2.sampling.ritb import RightBootstrap
from BLRT2.sampling.ritd import RightDirichlet
from lifelines.statistics import logrank_test

from BLRT2.others.distribution import sur_null_inf
import scipy.stats as st
import os
import multiprocessing
from datetime import datetime as dt
import pickle


class RightLogRank:
    def __init__(self, X1, X2, D1, D2):
        self.X1, self.X2 = X1, X2
        self.D1, self.D2 = D1, D2
        self.p = []

    def clrt(self):
        res = logrank_test(self.X1, self.X2, self.D1, self.D2)
        self.p.append(res.p_value)

    def wlrt(self, p, q):
        res = logrank_test(self.X1, self.X2, self.D1, self.D2, weightings='fleming-harrington', p=p, q=q)
        self.p.append(res.p_value)

    @staticmethod
    def divi(num, den):
        den[num == 0] = 1
        return num / den

    def dlrt(self, size, alpha=0.001, a1=1.0, b1=1.0, a2=1.0, b2=1.0, m=1000):
        rd1 = RightDirichlet(self.X1, self.D1, size, alpha=alpha)
        rd2 = RightDirichlet(self.X2, self.D2, size, alpha=alpha)
        tau = min(rd1.X[0], rd2.X[0])
        grid_x = np.linspace(0.0, tau, m)
        rd1.prior(a=a1, b=b1)
        rd1.imputation()
        rd1.sampling(grid_x)
        rd2.prior(a=a2, b=b2)
        rd2.imputation()
        rd2.sampling(grid_x)

        N1 = np.sum(grid_x[:-1].reshape([-1, 1]) <= rd1.X, axis=1)
        N2 = np.sum(grid_x[:-1].reshape([-1, 1]) <= rd2.X, axis=1)
        NN = N1 * N2 / (N1 + N2)
        H1 = self.divi(rd1.S[:, 1:] - rd1.S[:, :-1], rd1.S[:, :-1])
        H2 = self.divi(rd2.S[:, 1:] - rd2.S[:, :-1], rd2.S[:, :-1])
        Q1 = np.sum(NN * H1, axis=1)
        Q2 = np.sum(NN * H2, axis=1)

        PoP = 2 * min(np.mean(Q1.reshape([-1, 1]) > Q2), np.mean(Q1.reshape([-1, 1]) < Q2))
        self.p.append(PoP)
        return (Q1.reshape([-1, 1]) - Q2).flatten()

    def blrt(self, size, mode=1, bay=True):
        rb1 = RightBootstrap(self.X1, self.D1, size)
        rb2 = RightBootstrap(self.X2, self.D2, size)
        if mode == 1:
            rb1.samplingI(bay=bay)
            rb2.samplingI(bay=bay)
        elif mode == 2:
            rb1.samplingW(bay=bay)
            rb2.samplingW(bay=bay)
        elif mode == 3:
            rb1.samplingB(bay=bay)
            rb2.samplingB(bay=bay)
        else:
            print('mode error!!! ')

        H1 = rb1.P / np.cumsum(rb1.P, axis=1)
        H2 = rb2.P / np.cumsum(rb2.P, axis=1)
        N11 = np.sum(rb1.T.reshape([-1, 1]) <= rb1.X, axis=1)
        N12 = np.sum(rb1.T.reshape([-1, 1]) <= rb2.X, axis=1)
        N21 = np.sum(rb2.T.reshape([-1, 1]) <= rb1.X, axis=1)
        N22 = np.sum(rb2.T.reshape([-1, 1]) <= rb2.X, axis=1)
        NN1 = N11 * N12 / (N11 + N12)
        NN2 = N21 * N22 / (N21 + N22)
        Q1 = np.sum((NN1 * H1)[:, rb1.T != np.inf], axis=1)
        Q2 = np.sum((NN2 * H2)[:, rb2.T != np.inf], axis=1)

        PoP = 2 * min(np.mean(Q1.reshape([-1, 1]) > Q2), np.mean(Q1.reshape([-1, 1]) < Q2))
        self.p.append(PoP)


def main(it, num=1, scaleC=1.0, sizeX=100, sizeB=1000):
    print(it)
    np.random.seed(19971107 + it)
    survival1 = survival2 = sur_null_inf()[num]
    # survival1, survival2 = sur_nonnull()[num]
    T1 = survival1.rvs(size=sizeX)
    T2 = survival2.rvs(size=sizeX)
    C1 = st.expon(scale=scaleC).rvs(size=sizeX)
    C2 = st.expon(scale=scaleC).rvs(size=sizeX)
    C1[C1 >= 1.0] = 1.0
    C2[C2 >= 1.0] = 1.0
    D1 = (T1 <= C1)
    D2 = (T2 <= C2)
    X1 = T1 * D1 + C1 * (1 - D1)
    X2 = T2 * D2 + C2 * (1 - D2)
    # print('censoring rate 1: {} (=1: {})'.format(1 - D1.mean(), np.mean(X1 == 1)))
    # print('censoring rate 2: {} (=1: {})'.format(1 - D2.mean(), np.mean(X2 == 1)))

    rlr = RightLogRank(X1, X2, D1, D2)
    rlr.clrt()
    M = [1, 5, 10, 20, 50, 100, 200, 500, 1000, 10000]
    Q = rlr.dlrt(size=sizeB, alpha=0.001, a1=0.001, b1=0.001, a2=0.001, b2=0.001, m=10 * sizeX)
    std = [np.std(Q)]
    for m in M:
        Q1 = rlr.dlrt(size=sizeB, alpha=m, a1=2, b1=2, a2=2, b2=2, m=10 * sizeX)
        Q2 = rlr.dlrt(size=sizeB, alpha=m, a1=2, b1=2, a2=2, b2=1, m=10 * sizeX)
        Q3 = rlr.dlrt(size=sizeB, alpha=m, a1=2, b1=2, a2=2, b2=3, m=10 * sizeX)
        std.extend([np.std(Q1), np.std(Q2), np.std(Q3)])

    return [rlr.p, std]


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    with multiprocessing.Pool(processes=60) as pool:
        begin = dt.now()
        its = np.arange(10000)
        R = pool.map(main, its)
        end = dt.now()
        print((end - begin).seconds)

    with open('../data/std_a2', 'wb') as file:
        pickle.dump(R, file)
