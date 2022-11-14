import numpy as np

from matplotlib import pyplot as plt
from BLRT2.others.distribution import sur_null
from BLRT2.sampling.ritd import RightDirichlet, draw
import scipy.stats as st
import lifelines as lf


class RightBootstrap:
    def __init__(self, X, D, size):
        X = np.array(X)
        D = np.array(D)
        index = np.argsort(-(X + D * 1e-10))
        self.X = X[index]
        self.D = D[index]
        if self.D[0] == 0:
            self.X[0] = np.inf
            self.D[0] = 1

        self.T = self.X[self.D == 1]
        self.C = self.X[self.D == 0]
        self.W = 1.0 * np.ones(self.T.size)
        for c in self.C:
            flag = (self.T >= c)
            self.W[flag] += self.W[flag] / self.W[flag].sum()

        self.size = size
        self.P = None

    def samplingI(self, bay=True):
        TT = np.zeros([self.size, self.X.size])
        TT[:, self.D == 1] = np.arange(self.T.size)
        for i in np.arange(self.X.size)[self.D == 0]:
            choice = np.random.choice(a=i, size=self.size)
            TT[:, i] = TT[np.arange(self.size), choice]

        ones = np.ones_like(self.X)
        if bay:
            P = np.random.dirichlet(alpha=ones, size=self.size)
        else:
            P = np.random.multinomial(n=self.X.size, pvals=ones / ones.sum(), size=self.size) / self.X.size

        self.P = np.ones([self.size, self.T.size])
        for i in np.arange(self.T.size):
            self.P[:, i] = (P * (TT == i)).sum(axis=1)

    def samplingW(self, bay=True):
        if bay:
            self.P = np.random.dirichlet(alpha=self.W, size=self.size)
        else:
            self.P = np.random.multinomial(n=self.X.size, pvals=self.W / self.W.sum(), size=self.size) / self.X.size

    def samplingB(self, bay=True):
        ones = np.ones_like(self.X)
        if bay:
            P = np.random.dirichlet(alpha=ones, size=self.size)
        else:
            P = np.random.multinomial(n=self.X.size, pvals=ones / ones.sum(), size=self.size) / self.X.size

        self.P = P[:, self.D == 1]
        for i in np.arange(self.X.size)[self.D == 0]:
            flag = (self.T >= self.X[i])
            W = self.P[:, flag] / self.P[:, flag].sum(axis=1, keepdims=True)
            if not bay:
                W[np.isnan(W)] = np.ones_like(W[np.isnan(W)]) / np.sum(flag)

            self.P[:, flag] += P[:, i].reshape([-1, 1]) * W

    def plot(self, S, ax, label, c):
        if self.T[0] == np.inf:
            x = np.append(np.array([self.T[1:], self.T[1:]]).T.flatten(), 0.0)
            y = np.append(S[0], np.array([S[1:], S[1:]]).T.flatten())
        else:
            x = np.append(np.array([self.T, self.T]).T.flatten(), 0.0)
            y = np.append(0.0, np.array([S, S]).T.flatten())

        ax.plot(x, y, c, label=label)

    def draw(self, ax, label, c, alpha=0.05):
        S = np.cumsum(self.P, axis=1)
        self.plot(S.mean(axis=0), ax, label=label + ' (mean)', c=c + '-')
        self.plot(np.quantile(S, q=alpha / 2, axis=0), ax, label=label + ' (lower)', c=c + '--')
        self.plot(np.quantile(S, q=1 - alpha / 2, axis=0), ax, label=label + ' (upper)', c=c + '--')


def main(num, scaleC=1.0, sizeX=100, sizeB=1000, alpha=0.05):
    survival = sur_null()[num]
    T = survival.rvs(size=sizeX)
    C = st.expon(scale=scaleC).rvs(size=sizeX)
    C[C >= 1.0] = 1.0
    D = (T <= C)
    X = T * D + C * (1 - D)
    print('censoring rate: {} (=1: {})'.format(1 - D.mean(), np.mean(X == 1)))

    grid_x = np.linspace(0, X[D].max(), 1000)
    kmf = lf.KaplanMeierFitter(alpha=alpha, label='kmf').fit(X, D, timeline=grid_x)
    rb = RightBootstrap(X=X, D=D, size=sizeB)

    fig, ax = plt.subplots(figsize=[10, 8])
    kmf.plot_survival_function(ax=ax, ci_show=True, label='NPMLE', c='gray')
    rb.samplingI(bay=True)
    rb.draw(ax=ax, label='NIB', alpha=alpha, c='b')
    rb.samplingW(bay=True)
    rb.draw(ax=ax, label='KM', alpha=alpha, c='g')
    rb.samplingB(bay=True)
    rb.draw(ax=ax, label='B', alpha=alpha, c='r')
    draw(RightDirichlet(X=X, D=D, size=sizeB, alpha=0.001), grid_x, ax, 'k', 'DP')
    ax.legend()
    fig.tight_layout()
    fig.show()


if __name__ == '__main__':
    main(num=1, scaleC=0.8, sizeX=100, sizeB=10000)
