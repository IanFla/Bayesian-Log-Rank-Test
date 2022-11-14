import numpy as np
import scipy.stats as st
import scipy.optimize as opt
import numdifftools as nd

from matplotlib import pyplot as plt
from BLRT4.sampling.cases import survival_nonnull, Weibull
import lifelines as lf


class MixDirProcess:
    def __init__(self, X, D, size, M, G0):
        X = np.array(X)
        D = np.array(D)
        index = np.argsort(-(X + D * 1e-10))
        self.X = X[index]
        self.D = D[index]
        self.size = size
        self.M = M
        self.G0 = G0

        self.cache = None
        self.Theta = None
        self.T = None
        self.S = None

    def prior(self, a=4.0, b=4.0):
        Tset = list(set(self.X[self.D == 1]))
        logSsum = np.sum([np.log(self.G0.svl(np.array([T]))[0]) for T in Tset])
        index = np.arange(self.X.size)[self.D == 0]

        def log_g(x):
            if index.size > 0:
                return np.sum([np.log(1e-200 + i + self.M * (self.G0.svl(np.array([self.X[i]]))[0] ** x))
                               for i in index], axis=0)
            else:
                return np.zeros_like(x)

        def neglogpdf(x):
            y = np.inf * np.ones_like(x)
            flag = (x > 0)
            y[flag] = -(st.gamma(a=a, scale=1 / b).logpdf(x[flag]) + len(Tset) * np.log(x[flag])
                        + logSsum * x[flag] + log_g(x[flag]))
            return y

        mode = opt.minimize(fun=neglogpdf, x0=np.array(2.0)).x[0]
        std = 1 / np.sqrt(nd.Hessian(neglogpdf)(mode))[0][0]
        envelope = st.t(loc=mode, scale=2 * std, df=1)
        target = lambda x: neglogpdf(x) + envelope.logpdf(x)
        points = envelope.rvs(size=1000)
        res = opt.minimize(fun=target, x0=points[np.argmin(target(points))]).x[0]
        rejection = lambda x: np.exp(-target(x) + target(np.array([res]))[0])
        self.cache = [neglogpdf, envelope, rejection]

    def imputation(self):
        neglogpdf, envelope, rejection = self.cache
        size0 = int(5 * self.size)
        while True:
            Theta0 = envelope.rvs(size0)
            acc_ps = rejection(Theta0)
            Theta1 = Theta0[np.random.uniform(size=size0) <= acc_ps]
            if Theta1.size < self.size:
                rate = acc_ps.mean()
                print('acceptance rate: {}'.format(rate))
                size0 = int(2 * self.size / rate)
            else:
                break

        self.Theta = np.random.choice(a=Theta1, size=self.size, replace=False)

        self.T = np.zeros([self.size, self.X.size])
        self.T[:, self.D == 1] = self.X[self.D == 1]
        for i in np.arange(self.X.size)[self.D == 0]:
            ps = np.append(self.M, np.ones(i))
            choice = np.random.choice(i + 1, size=self.size, p=ps / ps.sum())
            LB = 1 - self.G0.svl(np.array([self.X[i]]))[0] ** self.Theta[choice == 0]
            u = LB + (1 - LB) * np.random.uniform(size=np.sum(choice == 0))
            self.T[choice == 0, i] = self.G0.ppf(1 - (1 - u) ** (1 / self.Theta[choice == 0]))
            self.T[choice != 0, i] = self.T[choice != 0, choice[choice != 0] - 1]

    def sampling(self, grid_x):
        if grid_x[0] != 0.0:
            grid_x = np.append(0.0, grid_x)

        self.S = np.zeros([self.size, grid_x.size])
        self.S[:, 0] = 1.0
        a = (self.X.size + self.M) * np.ones(self.size)
        for i, grid in enumerate(grid_x[1:]):
            c = a.copy()
            a = np.sum(grid < self.T, axis=1) + self.M * (self.G0.svl(np.array([grid]))[0] ** self.Theta)
            flag = (a > 0)
            b = c[flag] - a[flag]
            b[b == 0] = 1e-10
            ratio_S = np.random.beta(a=a[flag], b=b)
            self.S[flag, i + 1] = ratio_S * self.S[flag, i]


def draw(mdp, grid_x, ax, c, label):
    mdp.prior(a=4.0, b=4.0)
    mdp.imputation()
    mdp.sampling(grid_x)
    grid = np.linspace(0.001, 6, 1000)
    ax[0].plot(grid, 1.0 * np.exp(mdp.cache[0](grid).min() - mdp.cache[0](grid)), color=c, label=label)
    hist, grid = np.histogram(mdp.Theta, bins=30)
    hist = np.array([hist, hist]).T.flatten()
    grid = np.append(grid[0], np.append(np.array([grid[1:-1], grid[1:-1]]).T.flatten(), grid[-1]))
    ax[0].plot(grid, hist / hist.max(), color=c)
    ax[1].plot(grid_x, mdp.S.mean(axis=0), c + '-', label=label)
    ax[1].plot(grid_x, np.quantile(mdp.S, q=1 - 0.05 / 2, axis=0), c + '--')
    ax[1].plot(grid_x, np.quantile(mdp.S, q=0.05 / 2, axis=0), c + '--')


def main(num, i, scaleC=1.0, sizeX=100, sizeB=2000, G0=None):
    survival = survival_nonnull()[num][i]
    T = survival.rvs(size=sizeX)
    C = st.expon(scale=scaleC).rvs(size=sizeX)
    C[C >= 1.0] = 1.0
    D = (T <= C)
    X = T * D + C * (1 - D)
    print('censoring rate: {} (=1: {})'.format(1 - D.mean(), np.mean(X == 1)))

    fig, ax = plt.subplots(1, 2, figsize=[16, 6])
    grid_x = np.linspace(0, 1, 1000)
    kmf = lf.KaplanMeierFitter(alpha=0.05, label='kmf').fit(X, D, timeline=grid_x)
    kmf.plot_survival_function(ax=ax[1], ci_show=True, label='NPMLE', c='gray')

    c = ['k', 'b', 'g', 'y', 'r', 'c', 'm']
    for j, M in enumerate([0.001, 70, 10000]):
        mdp = MixDirProcess(X=X, D=D, size=sizeB, M=M, G0=G0)
        draw(mdp, grid_x, ax, c[j], label=M)

    ax[0].legend()
    ax[1].legend()
    fig.tight_layout()
    fig.show()


if __name__ == '__main__':
    main(num=0, i=1, G0=Weibull(tau=0.5, scale=0.5))
    main(num=1, i=1, G0=Weibull(tau=1, scale=0.5))
    main(num=2, i=1, G0=Weibull(tau=2, scale=0.75))
