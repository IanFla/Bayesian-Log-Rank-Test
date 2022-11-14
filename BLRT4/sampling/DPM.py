import numpy as np

from matplotlib import pyplot as plt
from BLRT4.sampling.cases import survival_nonnull
import lifelines as lf
from datetime import datetime as dt


class DirProMixture:
    def __init__(self, X, D, size, M=1.0, alpha=1.0, beta=1.0):
        X = np.array(X)
        D = np.array(D)
        index = np.argsort(-(X + D * 1e-10))
        self.X = X[index]
        self.D = D[index]
        self.size = size
        self.M = M
        self.alpha = alpha
        self.beta = beta

        self.Theta = None
        self.S = None

    def step(self, theta):
        for i in np.random.permutation(np.arange(theta.size)):
            if self.D[i]:
                prob = theta * np.exp(-theta * self.X[i])
                prob[i] = self.M * self.alpha * (self.beta ** self.alpha) / \
                          ((self.beta + self.X[i]) ** (self.alpha + 1))
                theta[i] = np.random.gamma(shape=self.alpha + 1, scale=1 / (self.beta + self.X[i]))
            else:
                prob = np.exp(-theta * self.X[i])
                prob[i] = self.M * (self.beta ** self.alpha) / ((self.beta + self.X[i]) ** self.alpha)
                theta[i] = np.random.gamma(shape=self.alpha, scale=1 / (self.beta + self.X[i]))

            theta[i] = np.random.choice(theta, p=prob / prob.sum())

        return theta

    def gibbs(self, thin=1, burn=100):
        theta = np.random.gamma(shape=self.alpha, scale=1 / self.beta, size=self.X.size)
        for b in range(burn):
            theta = self.step(theta)

        self.Theta = np.zeros([self.size, self.X.size])
        for s in range(self.size):
            for t in range(thin):
                theta = self.step(theta)

            self.Theta[s] = theta

    def sampling(self, grid_x, num=1000):
        W = np.random.beta(a=1, b=self.M + self.X.size, size=self.size * num).reshape([self.size, num])
        W[:, 1:] = W[:, 1:] * np.cumprod(1 - W, axis=1)[:, :-1]
        W = W / W.sum(axis=1).reshape([-1, 1])

        prob = np.append(self.M, np.ones(self.X.size))
        C = np.random.choice(a=self.X.size + 1, size=self.size * num, p=prob / prob.sum()).reshape([self.size, num])
        T = 1.0 * np.zeros(self.size * num).reshape([self.size, num])
        T[C == 0] = np.random.gamma(shape=self.alpha, scale=1 / self.beta, size=np.sum(C == 0))
        for s in range(self.size):
            T[s][C[s] != 0] = self.Theta[s][C[s][C[s] != 0] - 1]

        self.S = np.zeros([self.size, grid_x.size])
        for n in range(num):
            self.S += W[:, n].reshape([-1, 1]) * np.exp(-T[:, n].reshape([-1, 1]) * grid_x)


def draw(dpm, grid_x, ax, c, label):
    t0 = dt.now()
    dpm.gibbs(thin=1, burn=100)
    t1 = dt.now()
    print(t1 - t0)
    dpm.sampling(grid_x, num=1000)
    print(dt.now() - t1)
    print(np.mean(dpm.Theta))
    ax[0].hist(dpm.Theta.flatten(), bins=100, color=c, histtype='step', label=label)
    ax[1].plot(grid_x, dpm.S.mean(axis=0), c + '-', label=label)
    ax[1].plot(grid_x, np.quantile(dpm.S, q=1 - 0.05 / 2, axis=0), c + '--')
    ax[1].plot(grid_x, np.quantile(dpm.S, q=0.05 / 2, axis=0), c + '--')


def main(num, i, scaleC=1.0, sizeX=100, sizeB=1000, M=1.0):
    survival = survival_nonnull()[num][i]
    T = survival.rvs(size=sizeX)
    C = np.random.exponential(scale=scaleC, size=sizeX)
    C[C >= 1.0] = 1.0
    D = (T <= C)
    X = T * D + C * (1 - D)
    print('censoring rate: {} (=1: {})'.format(1 - D.mean(), np.mean(X == 1)))

    fig, ax = plt.subplots(1, 2, figsize=[16, 6])
    grid_x = np.linspace(0, 1, 1000)
    kmf = lf.KaplanMeierFitter(alpha=0.05, label='kmf').fit(X, D, timeline=grid_x)
    kmf.plot_survival_function(ax=ax[1], ci_show=True, label='NPMLE', c='gray')

    c = ['k', 'b', 'g', 'y', 'r', 'c', 'm']
    for j, a in enumerate([1, 1, 1, 1, 1]):
        dpm = DirProMixture(X, D, size=sizeB, M=M, alpha=a, beta=0.5 * a)
        draw(dpm, grid_x, ax, c[j], label=a)

    ax[0].legend()
    ax[1].legend()
    fig.tight_layout()
    fig.show()


if __name__ == '__main__':
    main(num=1, i=1, M=5)
