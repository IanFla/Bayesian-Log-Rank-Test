import numpy as np
from BLRT4.sampling.cases import survival_null, survival_nonnull
from BLRT4.inference.LRT import LogRankTest

import os
import multiprocessing
from datetime import datetime as dt
from functools import partial
import pickle


def experiment(curves, scaleC, sizeX, sizeB, seed):
    np.random.seed(seed)
    T1 = curves[0].rvs(size=sizeX)
    T2 = curves[1].rvs(size=sizeX)
    C1 = np.random.exponential(scale=scaleC, size=sizeX)
    C2 = np.random.exponential(scale=scaleC, size=sizeX)
    C1[C1 >= 1.0] = 1.0
    C2[C2 >= 1.0] = 1.0
    D1 = (T1 <= C1)
    D2 = (T2 <= C2)
    X1 = T1 * D1 + C1 * (1 - D1)
    X2 = T2 * D2 + C2 * (1 - D2)

    priors = [[1, 1, 1], [1, 1, 0.5]]
    lrt = LogRankTest(X1, X2, D1, D2)
    lrt.clrt()
    for prior in priors:
        lrt.blrt(size=sizeB, M=5, prior=prior, size_G=1000)

    return np.append([1 - D1.mean(), 1 - D2.mean(), np.mean(X1 == 1), np.mean(X2 == 1)], lrt.p)


def run(it, sizeX):
    print(it)
    seed = 1997 + 1107 * it
    Curves = np.vstack([survival_null(), survival_nonnull()])
    result = [experiment(curves=Curves[num], scaleC=1.0, sizeX=sizeX, sizeB=1000, seed=seed)
              for num in range(len(Curves))]
    return result


def main(sizeX):
    os.environ['OMP_NUM_THREADS'] = '1'
    with multiprocessing.Pool(processes=30) as pool:
        begin = dt.now()
        its = np.arange(1000)
        R = pool.map(partial(run, sizeX=sizeX), its)
        end = dt.now()
        print((end - begin).seconds)

    with open('../results/DPM_{}_'.format(sizeX), 'wb') as file:
        pickle.dump(R, file)


if __name__ == '__main__':
    main(100)
