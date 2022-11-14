import numpy as np
import scipy.stats as st
from BLRT3.others.distribution import sur_nonnull
from BLRT3.inference.logrank import RightLogRank

import os
import multiprocessing
from datetime import datetime as dt
from functools import partial
import pickle


def experiment(num, scaleC, sizeX, sizeB, seed):
    np.random.seed(seed)
    T1 = sur_nonnull()[num][0].rvs(size=sizeX)
    T2 = sur_nonnull()[num][1].rvs(size=sizeX)
    k = [[0.5, 0.5], [1.0, 1.0], [2.0, 2.0]][num]
    s = [[1.0, 0.5], [1.0, 0.5], [1, 0.75]][num]

    C1 = st.expon(scale=scaleC).rvs(size=sizeX)
    C2 = st.expon(scale=scaleC).rvs(size=sizeX)
    C1[C1 >= 1.0] = 1.0
    C2[C2 >= 1.0] = 1.0
    D1 = (T1 <= C1)
    D2 = (T2 <= C2)
    X1 = T1 * D1 + C1 * (1 - D1)
    X2 = T2 * D2 + C2 * (1 - D2)

    Alpha = [10, 100, 1000]
    a = 4.0
    rlr = RightLogRank(X1, X2, D1, D2)
    rlr.clrt()

    for alpha in Alpha:
        rlr.dlrt(size=sizeB, alpha=alpha, m=1000, k=k, s=[s[1], s[0]], a=[a, a], b=[a, a])

    return np.append([1 - D1.mean(), 1 - D2.mean(), np.mean(X1 == 1), np.mean(X2 == 1)], rlr.p)


def run(it, sizeX):
    print(it)
    seed = 1997 + 1107 * it
    result = [experiment(num=num, scaleC=1.0, sizeX=sizeX, sizeB=1000, seed=seed) for num in range(3)]
    return result


def main(sizeX):
    os.environ['OMP_NUM_THREADS'] = '1'
    with multiprocessing.Pool(processes=30) as pool:
        begin = dt.now()
        its = np.arange(10000)
        R = pool.map(partial(run, sizeX=sizeX), its)
        end = dt.now()
        print((end - begin).seconds)

    with open('../data/weibull3_{}'.format(sizeX), 'wb') as file:
        pickle.dump(R, file)


if __name__ == '__main__':
    main(100)
