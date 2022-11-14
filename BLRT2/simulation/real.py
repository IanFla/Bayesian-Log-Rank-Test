import numpy as np
from BLRT2.inference.ritlr import RightLogRank
import pandas as pd
import lifelines as lf

import os
import multiprocessing
from datetime import datetime as dt
import pickle


def read(name):
    df1 = pd.read_csv('../data/{}0.csv'.format(name), index_col=0)
    X1 = 1.0 * df1.X1.values
    D1 = df1.D1.values
    df2 = pd.read_csv('../data/{}1.csv'.format(name), index_col=0)
    X2 = 1.0 * df2.X2.values
    D2 = df2.D2.values
    # Xmax = np.max(np.append(X1, X2))
    return [X1, X2, D1, D2]


def plot(data, name, ax):
    bdwth = 15 if name == 'DRS' else 70
    naf = lf.NelsonAalenFitter().fit(data[0], data[2], label='h0 (control)')
    naf.plot_hazard(bandwidth=bdwth, ax=ax)
    naf = lf.NelsonAalenFitter().fit(data[1], data[3], label='h1 (treatment)')
    naf.plot_hazard(bandwidth=bdwth, ax=ax)
    md = np.median(np.append(data[0], data[1]))
    ax.plot(md, 0, 'x', label='median')
    ax.legend()
    ax.set_title(name)


def shuffle(data):
    X1, X2, D1, D2 = data
    X = np.append(X1, X2)
    D = np.append(D1, D2)
    n = X1.size
    md = np.median(X)
    index_nd = np.random.permutation(np.arange(X.size))
    index_ed = np.random.permutation(np.arange(X.size)[X > md])
    index_ld = np.random.permutation(np.arange(X.size)[X < md])
    X_nd = X.copy()
    D_nd = D.copy()
    X_nd = X_nd[index_nd]
    D_nd = D_nd[index_nd]
    data_nd = [X_nd[:n], X_nd[n:], D_nd[:n], D_nd[n:]]
    X_ed = X.copy()
    D_ed = D.copy()
    X_ed[X_ed > md] = X_ed[index_ed]
    D_ed[X_ed > md] = D_ed[index_ed]
    data_ed = [X_ed[:n], X_ed[n:], D_ed[:n], D_ed[n:]]
    X_ld = X.copy()
    D_ld = D.copy()
    X_ld[X_ld < md] = X_ld[index_ld]
    D_ld[X_ld < md] = D_ld[index_ld]
    data_ld = [X_ld[:n], X_ld[n:], D_ld[:n], D_ld[n:]]
    return [data_nd, data_ed, data_ld]


def experiment(data, sizeB):
    X1, X2, D1, D2 = data
    rlr = RightLogRank(X1, X2, D1, D2)
    rlr.clrt()
    rlr.dlrt(size=sizeB, alpha=0.001, m=10 * X1.size)
    return np.append([1 - D1.mean(), 1 - D2.mean(), np.mean(X1 == 1), np.mean(X2 == 1)], rlr.p)


def run(it):
    print(it)
    np.random.seed(1997 + 1107 * it)
    data = read('DRS')
    Data = shuffle(data)
    data = read('CGD')
    Data.extend(shuffle(data))
    result = [experiment(data, sizeB=1000) for data in Data]
    return result


def main():
    os.environ['OMP_NUM_THREADS'] = '1'
    with multiprocessing.Pool(processes=32) as pool:
        begin = dt.now()
        its = np.arange(10000)
        R = pool.map(run, its)
        end = dt.now()
        print((end - begin).seconds)

    with open('../data/real', 'wb') as file:
        pickle.dump(R, file)


if __name__ == '__main__':
    main()
