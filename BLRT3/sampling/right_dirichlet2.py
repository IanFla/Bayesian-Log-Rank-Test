import numpy as np
import scipy.stats as st
import scipy.optimize as opt
import numdifftools as nd

from matplotlib import pyplot as plt
from BLRT3.others.distribution2 import sur_nonnull
import lifelines as lf


class RightDirichlet:
    def __init__(self, X, D, size, alpha, tl):
        X = np.array(X)
        D = np.array(D)
        index = np.argsort(-(X + D * 1e-10))
        self.X = X[index]
        self.D = D[index]
        self.size = size
        self.alpha = alpha
        self.tl = tl

        self.cache = None
        self.theta = None
        self.T = None
        self.S = None

    def prior(self, a=4.0, b=4.0):
        Tset = list(set(self.X[self.D == 1]))
        logSsum = np.sum([np.log(self.tl.survival(np.array([T]))[0]) for T in Tset])
        index = np.arange(self.X.size)[self.D == 0]

        def log_g(x):
            if index.size > 0:
                return np.sum([np.log(1e-200 + i + self.alpha * (self.tl.survival(np.array([self.X[i]]))[0] ** x))
                               for i in index], axis=0)
            else:
                np.zeros_like(x)

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
        res = opt.minimize(fun=neglogpdf, x0=np.array(2.0)).x[0]
        rejection = lambda x: np.exp(-target(x) + target(np.array([res]))[0])
        self.cache = [neglogpdf, envelope, rejection]

    def imputation(self):
        neglogpdf, envelope, rejection = self.cache
        size0 = int(5 * self.size)
        while True:
            theta0 = envelope.rvs(size0)
            acc_ps = rejection(theta0)
            theta1 = theta0[np.random.uniform(size=size0) <= acc_ps]
            if theta1.size < self.size:
                rate = acc_ps.mean()
                print('acceptance rate: {}'.format(rate))
                size0 = int(2 * self.size / rate)
            else:
                break

        self.theta = np.random.choice(a=theta1, size=self.size, replace=False)

        self.T = np.zeros([self.size, self.X.size])
        self.T[:, self.D == 1] = self.X[self.D == 1]
        for i in np.arange(self.X.size)[self.D == 0]:
            ps = np.append(self.alpha, np.ones(i))
            ps /= ps.sum()
            choice = np.random.choice(i + 1, size=self.size, p=ps)
            LB = 1 - self.tl.survival(np.array([self.X[i]]))[0] ** self.theta[choice == 0]
            u = LB + (1 - LB) * st.uniform.rvs(size=np.sum(choice == 0))
            self.T[choice == 0, i] = self.tl.ppf(1 - (1 - u) ** (1 / self.theta[choice == 0]))
            self.T[choice != 0, i] = self.T[choice != 0, choice[choice != 0] - 1]

    def sampling(self, grid_x):
        if grid_x[0] != 0.0:
            grid_x = np.append(0.0, grid_x)

        self.S = np.zeros([self.size, grid_x.size])
        self.S[:, 0] = 1.0
        a = (self.X.size + self.alpha) * np.ones(self.size)
        for i, grid in enumerate(grid_x[1:]):
            c = a.copy()
            a = np.sum(grid < self.T, axis=1) + self.alpha * (self.tl.survival(np.array([grid]))[0] ** self.theta)
            flag = (a > 0)
            b = c[flag] - a[flag]
            b[b == 0] = 1e-10
            ratio_S = np.random.beta(a=a[flag], b=b)
            self.S[flag, i + 1] = ratio_S * self.S[flag, i]


def draw(rd, grid_x, ax, c, label):
    rd.prior()
    rd.imputation()
    rd.sampling(grid_x)
    grid = np.linspace(0.001, 6, 1000)
    ax[0].plot(grid, 1.0 * np.exp(rd.cache[0](grid).min() - rd.cache[0](grid)), color=c, label=label)
    ax[1].plot(grid_x, rd.S.mean(axis=0), c + '-', label=label)
    ax[1].plot(grid_x, np.quantile(rd.S, q=1 - 0.05 / 2, axis=0), c + '--')
    ax[1].plot(grid_x, np.quantile(rd.S, q=0.05 / 2, axis=0), c + '--')


def main(num, i, scaleC=1.0, sizeX=100, sizeB=1000):
    survival = sur_nonnull()[num][i]
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

    c = ['k', 'b', 'g', 'c', 'y', 'r', 'm']
    for j, alpha in enumerate([0.001, 1, 10, 50, 100, 1000, 10000]):
        rd = RightDirichlet(X=X, D=D, size=sizeB, alpha=alpha, tl=sur_nonnull()[num][1 - i])
        draw(rd, grid_x, ax, c[j], label=alpha)

    for a in ax:
        a.legend()

    fig.tight_layout()
    fig.show()


if __name__ == '__main__':
    main(num=3, i=0)
    main(num=3, i=1)
