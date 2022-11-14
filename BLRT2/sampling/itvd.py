import numpy as np
import scipy.stats as st
from BLRT2.others.distribution import TruncGamma, TruncExpon

from matplotlib import pyplot as plt
from BLRT2.others.distribution import sur_null, sur_nonnull
import lifelines as lf


class IntervalDirichlet:
    def __init__(self, L, R, size, alpha, a=0.001, b=0.001):
        self.L = np.array(L)
        self.R = np.array(R)
        self.index = np.arange(self.L.size)[self.L < self.R]
        self.size = size
        self.alpha = sum(self.L != self.R) if alpha == 'adapt' else alpha
        self.a = a
        self.b = b
        self.theta = None
        self.T = None
        self.S = None

    def theta_T(self, T):
        Tset = list(set(T))
        theta = TruncGamma(a=self.a + len(Tset), b=self.b + sum(Tset)).rvs(size=1)
        return theta

    def T_theta(self, theta, T):
        for i in np.random.permutation(self.index):
            flag = 1.0 * ((T >= self.L[i]) & (T <= self.R[i]))
            flag[i] = 0.0
            ps = np.append(self.alpha * (np.exp(-theta * self.L[i]) - np.exp(-theta * self.R[i])), flag)
            ps /= ps.sum()
            choice = np.random.choice(T.size + 1, size=1, p=ps)
            if choice == 0:
                T[i] = TruncExpon(theta=theta, size=1, L=self.L[i], R=self.R[i])
            else:
                T[i] = T[choice - 1]

        return T

    def imputation(self, thin=1, burn=100, theta0=2.0):
        T0 = TruncExpon(theta=theta0, size=self.L.size, L=self.L, R=self.R)
        for b in range(burn):
            T1 = self.T_theta(theta0, T0)
            theta1 = self.theta_T(T1)
            T0 = T1.copy()
            theta0 = theta1.copy()

        self.T = np.zeros([self.size, self.L.size])
        self.theta = np.zeros(self.size)
        for i in range(self.size):
            for t in range(thin):
                T1 = self.T_theta(theta0, T0)
                theta1 = self.theta_T(T1)
                T0 = T1.copy()
                theta0 = theta1.copy()

            self.T[i] = T0
            self.theta[i] = theta0

    def sampling(self, grid_x):
        if grid_x[0] != 0.0:
            grid_x = np.append(0.0, grid_x)

        self.S = np.zeros([self.size, grid_x.size])
        self.S[:, 0] = 1.0
        a = (self.L.size + self.alpha) * np.ones(self.size)
        for i, grid in enumerate(grid_x[1:]):
            c = a.copy()
            a = np.sum(grid < self.T, axis=1) + self.alpha * np.exp(-self.theta * grid)
            flag = (a > 0)
            b = c[flag] - a[flag]
            b[b == 0] = 1e-10
            ratio_S = np.random.beta(a=a[flag], b=b)
            self.S[flag, i + 1] = ratio_S * self.S[flag, i]


def draw(td, grid_x, ax, c, label):
    td.imputation()
    td.sampling(grid_x)
    ax.plot(grid_x, td.S.mean(axis=0), c + '-', label=label + ' (mean)')
    ax.plot(grid_x, np.quantile(td.S, q=1 - 0.05 / 2, axis=0), c + '--', label=label + ' (upper)')
    ax.plot(grid_x, np.quantile(td.S, q=0.05 / 2, axis=0), c + '--', label=label + ' (lower)')


def main(num, scaleC=1.0, sizeX=100, sizeB=1000):
    survival = sur_nonnull()[num][0]
    L = survival.rvs(size=sizeX)
    C = st.expon(scale=scaleC).rvs(size=sizeX)
    C[C >= 1.0] = 1.0
    D = (L <= C)
    L = L * D + C * (1 - D)
    R = L.copy()
    R[D == 0] = np.inf
    print('censoring rate: {} (=1: {})'.format(1 - D.mean(), np.mean(L == 1)))

    grid_x = np.linspace(0, 1, 1000)
    kmf = lf.KaplanMeierFitter().fit_interval_censoring(L, R, tol=1e-10)

    fig, ax = plt.subplots(figsize=[10, 8])
    kmf.plot_survival_function(ax=ax, label='NPMLE', c='gray')
    draw(IntervalDirichlet(L=L, R=R, size=sizeB, alpha=0.001), grid_x, ax, 'k', 'alpha=0.001')
    draw(IntervalDirichlet(L=L, R=R, size=sizeB, alpha=1), grid_x, ax, 'b', 'alpha=1')
    draw(IntervalDirichlet(L=L, R=R, size=sizeB, alpha=10), grid_x, ax, 'g', 'alpha=10')
    draw(IntervalDirichlet(L=L, R=R, size=sizeB, alpha=100), grid_x, ax, 'y', 'alpha=100')
    draw(IntervalDirichlet(L=L, R=R, size=sizeB, alpha=10000), grid_x, ax, 'r', 'alpha=10000')
    draw(IntervalDirichlet(L=L, R=R, size=sizeB, alpha='adapt'), grid_x, ax, 'c', 'alpha=adapt')
    ax.legend()
    fig.tight_layout()
    fig.show()


if __name__ == '__main__':
    np.random.seed(19971107)
    main(num=1)
