import numpy as np
import scipy.stats as st
from BLRT2.others.distribution import sur_null
from BLRT2.inference.ritlr import RightLogRank
from matplotlib import pyplot as plt


def experiment(survival, scaleC, sizeX, sizeB, seed):
    np.random.seed(seed)
    T1 = survival.rvs(size=sizeX)
    T2 = survival.rvs(size=sizeX)
    C1 = st.expon(scale=scaleC).rvs(size=sizeX)
    C2 = st.expon(scale=scaleC).rvs(size=sizeX)
    C1[C1 >= 1.0] = 1.0
    C2[C2 >= 1.0] = 1.0
    D1 = (T1 <= C1)
    D2 = (T2 <= C2)
    X1 = T1 * D1 + C1 * (1 - D1)
    X2 = T2 * D2 + C2 * (1 - D2)

    rlr = RightLogRank(X1, X2, D1, D2)
    Q = rlr.dlrt(size=sizeB, alpha=0.001, m=10 * sizeX)
    fig, ax = plt.subplots(figsize=[5, 4])
    ax.hist(Q, bins=100, density=True)
    if sizeX == 30:
        ax.plot([0, 0], [0, 0.17], '--')
        ax.set_ylim([0, 0.165])
    elif sizeX == 100:
        ax.plot([0, 0], [0, 0.09], '--')
        ax.set_ylim([0, 0.085])
    else:
        ax.plot([0, 0], [0, 0.055], '--')
        ax.set_ylim([0, 0.05])

    grid_x = np.linspace(Q.min(), Q.max(), 1000)
    mean = 4 * np.median(Q) - 3 * np.mean(Q)
    ax.plot(grid_x, st.norm(loc=mean, scale=Q.std()).pdf(grid_x), '--')
    ax.set_xlabel('$Q$')
    ax.set_ylabel('Relative Frequency')
    fig.tight_layout()
    fig.show()


def experiment2(survival, scaleC, sizeX, sizeB, seed):
    np.random.seed(seed)
    T1 = survival.rvs(size=sizeX)
    T2 = survival.rvs(size=sizeX)
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
    Q = rlr.dlrt(size=sizeB, alpha=0.001, m=10 * X1.size)
    # mean = Q.mean()
    mean = np.median(Q)
    std = Q.std()
    grid_x1 = np.linspace(mean - 3 * std, mean - 1.5 * std, 1000)
    grid_x2 = np.linspace(mean + 1.5 * std, mean + 3 * std, 1000)
    grid_x = np.linspace(1.5 * std, 3 * std, 1000)

    fig, ax = plt.subplots(figsize=[5, 4])
    ax.plot(grid_x, np.flip(np.log(np.mean(grid_x1.reshape([-1, 1]) >= Q, axis=1))), label='Left tail')
    ax.plot(grid_x, np.log(np.mean(grid_x2.reshape([-1, 1]) <= Q, axis=1)), label='Right tail')
    ax.plot(grid_x, np.log(1 - st.norm(loc=mean, scale=std).cdf(grid_x2)), ':', label='Tail of normal distribution')
    ax.set_xlabel('$Q$')
    ax.set_ylabel('Log Tail Probabilities')
    ax.legend(loc=3)
    fig.tight_layout()
    fig.show()


def main():
    plt.style.use('ggplot')
    experiment2(sur_null()[0], scaleC=1.0, sizeX=30, sizeB=1000, seed=19971107)
    experiment2(sur_null()[0], scaleC=1.0, sizeX=100, sizeB=1000, seed=1234567890)
    experiment2(sur_null()[0], scaleC=1.0, sizeX=300, sizeB=1000, seed=1234)


if __name__ == '__main__':
    main()
