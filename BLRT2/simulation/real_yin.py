import numpy as np
import scipy.stats as st
from BLRT2.inference.ritlr import RightLogRank
import pandas as pd
from matplotlib import pyplot as plt


def read(name):
    df1 = pd.read_csv('../data/{}0.csv'.format(name), index_col=0)
    X1 = 1.0 * df1.X1.values
    D1 = df1.D1.values
    df2 = pd.read_csv('../data/{}1.csv'.format(name), index_col=0)
    X2 = 1.0 * df2.X2.values
    D2 = df2.D2.values
    Xmax = max(X1.max(), X2.max())
    return [X1 / Xmax, X2 / Xmax, D1, D2]


def experiment(data, sizeB):
    plt.style.use('ggplot')
    X1, X2, D1, D2 = data
    rlr = RightLogRank(X1, X2, D1, D2)
    rlr.clrt()
    Q = rlr.dlrt(size=sizeB, alpha=0.001, m=10 * X1.size)
    fig, ax = plt.subplots(figsize=[5, 4])
    ax.hist(Q, bins=100, density=True)
    ax.plot([0, 0], [0, 0.13], '--')
    ax.set_xlabel('$Q$')
    ax.set_ylabel('Relative Frequency')
    ax.set_ylim([0, 0.125])
    fig.tight_layout()
    fig.show()


def experiment2(data, sizeB):
    plt.style.use('ggplot')
    X1, X2, D1, D2 = data
    rlr = RightLogRank(X1, X2, D1, D2)
    rlr.clrt()
    Q = rlr.dlrt(size=sizeB, alpha=0.001, m=10 * X1.size)
    mean = Q.mean()
    std = Q.std()
    grid_x1 = np.linspace(mean - 4 * std, mean - 2 * std, 1000)
    grid_x2 = np.linspace(mean + 2 * std, mean + 4 * std, 1000)
    grid_x = np.linspace(2 * std, 4 * std, 1000)

    fig, ax = plt.subplots(figsize=[5, 4])
    ax.plot(grid_x, np.flip(np.log(np.mean(grid_x1.reshape([-1, 1]) >= Q, axis=1))), label='tail 1')
    ax.plot(grid_x, np.log(np.mean(grid_x2.reshape([-1, 1]) <= Q, axis=1)), label='tail 2')
    ax.plot(grid_x, np.log(1 - st.norm(loc=mean, scale=std).cdf(grid_x2)), '--')
    ax.set_xlabel('Tail')
    ax.set_ylabel('Log Tail Probability')
    ax.legend()
    fig.tight_layout()
    fig.show()


# def main():
#     np.random.seed(19971107)
#     data = read('CGD')
#     R = []
#     for i in range(1000):
#         print(i)
#         R.append(experiment(data, sizeB=1000))
#
#     R = np.array(R)
#     print(R.mean(axis=0))


def main():
    np.random.seed(19971107)
    data = read('CGD')
    experiment(data, sizeB=1000)


if __name__ == '__main__':
    main()
