import numpy as np
import scipy.stats as st
from BLRT2.others.distribution import sur_nonnull
from BLRT2.inference.ritlr import RightLogRank

import os
import multiprocessing
from datetime import datetime as dt
from functools import partial
import pickle


def experiment(survival, scaleC, sizeX, sizeB, seed):
    np.random.seed(seed)
    T1 = survival[0].rvs(size=sizeX)
    T2 = survival[1].rvs(size=sizeX)
    C1 = st.expon(scale=scaleC).rvs(size=sizeX)
    C2 = st.expon(scale=scaleC).rvs(size=sizeX)
    C1[C1 >= 1.0] = 1.0
    C2[C2 >= 1.0] = 1.0
    D1 = (T1 <= C1)
    D2 = (T2 <= C2)
    X1 = T1 * D1 + C1 * (1 - D1)
    X2 = T2 * D2 + C2 * (1 - D2)

    rlr = RightLogRank(X1, X2, D1, D2)
    rlr.clrt()
    rlr.dlrt(size=sizeB, alpha=0.001, m=10 * sizeX)
    return np.append([1 - D1.mean(), 1 - D2.mean(), np.mean(X1 == 1), np.mean(X2 == 1)], rlr.p)


def run(it, sizeX):
    print(it)
    seed = 1997 + 1107 * it
    result = [experiment(survival, scaleC=1.0, sizeX=sizeX, sizeB=1000, seed=seed) for survival in sur_nonnull()]
    return result


def main(sizeX):
    os.environ['OMP_NUM_THREADS'] = '1'
    with multiprocessing.Pool(processes=32) as pool:
        begin = dt.now()
        its = np.arange(10000)
        R = pool.map(partial(run, sizeX=sizeX), its)
        end = dt.now()
        print((end - begin).seconds)

    with open('../data/non-null({})'.format(sizeX), 'wb') as file:
        pickle.dump(R, file)


if __name__ == '__main__':
    main(100)
