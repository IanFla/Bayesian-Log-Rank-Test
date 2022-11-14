import numpy as np
import scipy.stats as st
import scipy.optimize as opt
import numdifftools as nd
from BLRT2.others.distribution import TruncGamma, TruncExpon

from matplotlib import pyplot as plt
from BLRT2.others.distribution import sur_null, sur_nonnull
import lifelines as lf


class RightDirichlet:
    def __init__(self, X, D, size, alpha):
        X = np.array(X)
        D = np.array(D)
        index = np.argsort(-(X + D * 1e-10))
        self.X = X[index]
        self.D = D[index]
        self.size = size
        self.alpha = sum(1 - D) if alpha == 'adapt' else alpha
        self.cache = None
        self.rejection = None
        self.theta = None
        self.T = None
        self.S = None

    def prior(self, a=0.001, b=0.001):
        Tset = list(set(self.X[self.D == 1]))
        a, b = a + len(Tset), b + sum(Tset)

        index = np.arange(self.X.size)[self.D == 0]
        log_g = lambda x: np.sum([np.log(i + self.alpha * np.exp(-self.X[i] * x))
                                  for i in index], axis=0) if index.size > 0 else np.zeros_like(x)
        neglogpdf = lambda x: -(st.gamma(a=a, scale=1 / b).logpdf(x) + log_g(x))
        mode = opt.minimize(fun=neglogpdf, x0=np.array(2.0)).x[0]
        std = 1 / np.sqrt(nd.Hessian(neglogpdf)(mode))[0][0]
        lim = st.gamma(a=a, scale=1 / b).ppf(1 - 0.001)
        grid_theta = np.array([0.0, mode - 4 * std, mode - 3 * std, mode - 2 * std, mode - std, mode,
                               mode + std, mode + 2 * std, mode + 3 * std, mode + 4 * std, lim])
        grid_theta = grid_theta[(grid_theta >= 0.0) & (grid_theta <= lim)]
        self.cache = [log_g, neglogpdf, grid_theta[-1]]
        grid_g = log_g(grid_theta)
        acc_p = lambda x: np.exp(log_g(x) - np.interp(x, grid_theta, grid_g))
        neg_slope = np.append(-(grid_g[1:] - grid_g[:-1]) / (grid_theta[1:] - grid_theta[:-1]), 0.0)

        bs = b + neg_slope
        grid_theta_ = np.append(grid_theta, 1e100)
        G = np.array([TruncGamma(a=a, b=bs[i], l=grid_theta_[i], r=grid_theta_[i + 1]).pdf(grid_theta_[i: i + 2])
                      for i in range(bs.size)])
        G = np.diag(G[:, 1]) - np.diag(G[1:, 0], k=1)
        G[-1, :] = 1.0
        y = np.zeros(grid_theta.size)
        y[-1] = 1.0
        ps = np.linalg.inv(G).dot(y)
        if ps[-1] == 1.0:
            print('prior error!!! ')

        self.rejection = [grid_theta_, acc_p, a, bs, ps]

    def imputation(self):
        grid_theta_, acc_p, a, bs, ps = self.rejection
        size0 = int(2 * self.size)
        while True:
            index0, sizes = np.unique(np.random.choice(a=ps.size, size=size0, p=ps), return_counts=True)
            theta0 = np.hstack([TruncGamma(a=a, b=bs[index0[i]], l=grid_theta_[index0[i]], r=grid_theta_[index0[i] + 1])
                               .rvs(size=sizes[i]) for i in range(index0.size)])
            acc_ps = acc_p(theta0)
            theta1 = theta0[np.random.uniform(size=theta0.size) <= acc_ps]
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
            self.T[choice == 0, i] = TruncExpon(theta=self.theta[choice == 0], size=np.sum(choice == 0), L=self.X[i])
            self.T[choice != 0, i] = self.T[choice != 0, choice[choice != 0] - 1]

    def sampling(self, grid_x):
        if grid_x[0] != 0.0:
            grid_x = np.append(0.0, grid_x)

        self.S = np.zeros([self.size, grid_x.size])
        self.S[:, 0] = 1.0
        a = (self.X.size + self.alpha) * np.ones(self.size)
        for i, grid in enumerate(grid_x[1:]):
            c = a.copy()
            a = np.sum(grid < self.T, axis=1) + self.alpha * np.exp(-self.theta * grid)
            flag = (a > 0)
            b = c[flag] - a[flag]
            b[b == 0] = 1e-10
            ratio_S = np.random.beta(a=a[flag], b=b)
            self.S[flag, i + 1] = ratio_S * self.S[flag, i]


def draw(rd, grid_x, ax, c, label):
    # rd.prior()
    rd.prior(a=4, b=4)
    rd.imputation()
    rd.sampling(grid_x)
    ax.plot(grid_x, rd.S.mean(axis=0), c + '-', label=label + ' (mean)')
    ax.plot(grid_x, np.quantile(rd.S, q=1 - 0.05 / 2, axis=0), c + '--', label=label + ' (upper)')
    ax.plot(grid_x, np.quantile(rd.S, q=0.05 / 2, axis=0), c + '--', label=label + ' (lower)')


def main(num, scaleC=1.0, sizeX=100, sizeB=1000):
    # survival = sur_null()[num]
    survival = sur_null()[num]
    T = survival.rvs(size=sizeX)
    C = st.expon(scale=scaleC).rvs(size=sizeX)
    C[C >= 1.0] = 1.0
    D = (T <= C)
    X = T * D + C * (1 - D)
    print('censoring rate: {} (=1: {})'.format(1 - D.mean(), np.mean(X == 1)))

    grid_x = np.linspace(0, 1, 1000)
    kmf = lf.KaplanMeierFitter(alpha=0.05, label='kmf').fit(X, D, timeline=grid_x)

    # fig, ax = plt.subplots(1, 2, figsize=[10, 4])
    # rd = RightDirichlet(X=X, D=D, size=sizeB, alpha=0.001)
    # rd.prior()
    # rd.imputation()
    # log_g, neglogpdf, lim = rd.cache
    # grid = np.linspace(0, lim, 1000)
    # ax[0].plot(grid, log_g(grid))
    # ax[1].plot(grid, 1 * np.exp(neglogpdf(grid).min() - neglogpdf(grid)))
    # ax[1].hist(rd.theta, bins=20, density=True)
    # fig.tight_layout()
    # fig.show()

    fig, ax = plt.subplots(figsize=[10, 8])
    for alpha in [0.001, 1, 10, 100, 1000]:
        rd = RightDirichlet(X=X, D=D, size=sizeB, alpha=alpha)
        rd.prior(a=4, b=4)
        log_g, neglogpdf, lim = rd.cache
        grid = np.linspace(0, lim, 1000)
        ax.plot(grid, 1 * np.exp(neglogpdf(grid).min() - neglogpdf(grid)), label=alpha)

    ax.legend()
    fig.tight_layout()
    fig.show()

    fig, ax = plt.subplots(figsize=[10, 8])
    kmf.plot_survival_function(ax=ax, ci_show=True, label='NPMLE', c='gray')
    draw(RightDirichlet(X=X, D=D, size=sizeB, alpha=0.001), grid_x, ax, 'k', 'alpha=0.001')
    draw(RightDirichlet(X=X, D=D, size=sizeB, alpha=1), grid_x, ax, 'b', 'alpha=1')
    draw(RightDirichlet(X=X, D=D, size=sizeB, alpha=10), grid_x, ax, 'g', 'alpha=10')
    draw(RightDirichlet(X=X, D=D, size=sizeB, alpha=100), grid_x, ax, 'y', 'alpha=100')
    draw(RightDirichlet(X=X, D=D, size=sizeB, alpha=1000), grid_x, ax, 'r', 'alpha=1000')
    # draw(RightDirichlet(X=X, D=D, size=sizeB, alpha='adapt'), grid_x, ax, 'c', 'alpha=adapt')
    ax.legend()
    fig.tight_layout()
    fig.show()


if __name__ == '__main__':
    np.random.seed(19971107)
    main(num=0)
