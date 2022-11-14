import numpy as np

from matplotlib import pyplot as plt
import lifelines as lf


class Weibull:
    def __init__(self, tau, scale):
        self.tau = tau
        self.scale = scale

    def hzd(self, x):
        return self.tau * (x ** (self.tau - 1)) / (self.scale ** self.tau)

    def cdf(self, x):
        return 1 - np.exp(-(x / self.scale) ** self.tau)

    def svl(self, x):
        return np.exp(-(x / self.scale) ** self.tau)

    def pdf(self, x):
        return self.hzd(x) * self.svl(x)

    def ppf(self, u):
        return self.scale * ((np.log(1 / (1 - u))) ** (1 / self.tau))

    def rvs(self, size, lb=0.0):
        lb = np.array([lb])
        LB = self.cdf(lb)[0]
        u = np.random.uniform(size=size)
        x = self.ppf(LB + (1.0 - LB) * u)
        return x


class DoubleLinear:
    def __init__(self, a1, b1, c, a2, b2):
        self.a1 = a1
        self.b1 = b1
        self.c = c
        self.a2 = a2
        self.b2 = b2

    def hzd(self, x):
        h = np.zeros_like(x)
        flag1 = (x >= 0) & (x <= self.c)
        flag2 = (x > self.c) & (x <= 1)
        h[flag1] = self.a1 + self.b1 * x[flag1]
        h[flag2] = self.a2 + self.b2 * (x[flag2] - self.c)
        return h

    def cdf(self, x):
        z = 100.0 * np.ones_like(x)
        flag1 = (x >= 0) & (x <= self.c)
        flag2 = (x > self.c) & (x <= 1)
        z[flag1] = self.a1 * x[flag1] + 0.5 * self.b1 * (x[flag1] ** 2)
        z[flag2] = self.a1 * self.c + 0.5 * self.b1 * (self.c ** 2) + \
                   self.a2 * (x[flag2] - self.c) + 0.5 * self.b2 * ((x[flag2] - self.c) ** 2)
        u = 1 - 1 / np.exp(z)
        return u

    def svl(self, x):
        return 1 - self.cdf(x)

    def pdf(self, x):
        return self.hzd(x) * self.svl(x)

    def ppf(self, u):
        x = 2.0 * np.ones_like(u)
        z = - np.log(1 - u)
        c1 = self.a1 * self.c + 0.5 * self.b1 * (self.c ** 2)
        c2 = c1 + self.a2 * (1 - self.c) + 0.5 * self.b2 * ((1 - self.c) ** 2)
        flag1 = (z <= c1)
        flag2 = (z > c1) & (z <= c2)
        if self.b1 != 0:
            x[flag1] = (np.sqrt(self.a1 ** 2 + 2 * self.b1 * z[flag1]) - self.a1) / self.b1
        else:
            x[flag1] = z[flag1] / self.a1

        if self.b2 != 0:
            x[flag2] = (np.sqrt(self.a2 ** 2 + 2 * self.b2 * (z[flag2] - c1)) - self.a2) / self.b2 + self.c
        else:
            x[flag2] = (z[flag2] - c1) / self.a2 + self.c

        return x

    def rvs(self, size, lb=0.0):
        lb = np.array([lb])
        LB = self.cdf(lb)[0]
        u = np.random.uniform(size=size)
        x = self.ppf(LB + (1.0 - LB) * u)
        return x


def survival_null():
    curves = [[Weibull(tau=0.5, scale=1), Weibull(tau=0.5, scale=1)],
              [DoubleLinear(a1=2, b1=0, c=0.5, a2=0.5, b2=0), DoubleLinear(a1=2, b1=0, c=0.5, a2=0.5, b2=0)],
              [DoubleLinear(a1=1, b1=10, c=0.2, a2=3, b2=0), DoubleLinear(a1=1, b1=10, c=0.2, a2=3, b2=0)],
              [Weibull(tau=1, scale=1), Weibull(tau=1, scale=1)],
              [Weibull(tau=2, scale=1), Weibull(tau=2, scale=1)]]

    return curves


def survival_nonnull():
    curves = [[Weibull(tau=0.5, scale=1), Weibull(tau=0.5, scale=0.5)],
              [DoubleLinear(a1=1, b1=0, c=0.5, a2=1, b2=2), DoubleLinear(a1=1.5, b1=0, c=0.5, a2=1.5, b2=3)],
              [DoubleLinear(a1=0.25, b1=3, c=0.5, a2=1.75, b2=1), DoubleLinear(a1=1.75, b1=0, c=0.5, a2=1.75, b2=1)],
              [DoubleLinear(a1=2, b1=0, c=0.2, a2=2, b2=-2), DoubleLinear(a1=2, b1=0, c=0.2, a2=2, b2=4)],
              [DoubleLinear(a1=0.5, b1=1.5, c=1, a2=0, b2=0), DoubleLinear(a1=2, b1=-1.5, c=1, a2=0, b2=0)],
              [DoubleLinear(a1=0.5, b1=1, c=0.5, a2=1, b2=1), DoubleLinear(a1=1.5, b1=0, c=0.5, a2=0.5, b2=0)],
              [Weibull(tau=1, scale=1), Weibull(tau=1, scale=0.5)],
              [Weibull(tau=2, scale=1), Weibull(tau=2, scale=0.75)],
              [DoubleLinear(a1=1, b1=0, c=0.5, a2=2, b2=0), DoubleLinear(a1=1.5, b1=0, c=0.5, a2=3, b2=0)],
              [DoubleLinear(a1=1, b1=0, c=0.25, a2=3, b2=0), DoubleLinear(a1=2, b1=0, c=0.25, a2=2, b2=0)],
              [DoubleLinear(a1=1, b1=10, c=0.2, a2=3, b2=0), DoubleLinear(a1=3, b1=-10, c=0.2, a2=1, b2=0)],
              [DoubleLinear(a1=1, b1=2.5, c=0.8, a2=3, b2=0), DoubleLinear(a1=3, b1=-2.5, c=0.8, a2=1, b2=0)]]

    return curves


def main(nrow, ncol, curves):
    plt.style.use('ggplot')

    fig, ax = plt.subplots(nrow, ncol, figsize=[3 * ncol, 3 * nrow])
    ax = ax.flatten()
    for i, survival in enumerate(curves()):
        T1 = survival[0].rvs(size=100000)
        T2 = survival[1].rvs(size=100000)
        D1 = (T1 <= 1)
        D2 = (T2 <= 1)
        X1 = T1 * D1 + (1 - D1)
        X2 = T2 * D2 + (1 - D2)

        grid_x = np.linspace(1e-10, 1, 1000)
        naf1 = lf.NelsonAalenFitter().fit(X1, D1, timeline=grid_x)
        naf2 = lf.NelsonAalenFitter().fit(X2, D2, timeline=grid_x)
        ax[i].plot(grid_x, survival[0].hzd(grid_x))
        naf1.plot_hazard(ax=ax[i], bandwidth=0.05)
        ax[i].plot(grid_x, survival[1].hzd(grid_x))
        naf2.plot_hazard(ax=ax[i], bandwidth=0.05)
        ax[i].set_ylim([0, 6])

    fig.tight_layout()
    fig.show()

    fig, ax = plt.subplots(nrow, ncol, figsize=[3 * ncol, 3 * nrow])
    ax = ax.flatten()
    for i, survival in enumerate(curves()):
        T1 = survival[0].rvs(size=100000)
        T2 = survival[1].rvs(size=100000)
        ax[i].hist(T1, bins=100, density=True, histtype='step', range=[0, 10])
        ax[i].hist(T2, bins=100, density=True, histtype='step', range=[0, 10])

        grid_x = np.linspace(0.1, 1.2, 1000)
        ax[i].plot(grid_x, survival[0].pdf(grid_x))
        ax[i].plot(grid_x, survival[1].pdf(grid_x))
        ax[i].set_xlim([grid_x.min(initial=np.inf), grid_x.max(initial=-np.inf)])
        ax[i].set_ylim([0, 1.2 * survival[1].pdf(grid_x).max()])

    fig.show()


if __name__ == '__main__':
    main(2, 3, survival_null)
    main(3, 4, survival_nonnull)
