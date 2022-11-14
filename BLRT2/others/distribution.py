import numpy as np
import scipy.stats as st

from matplotlib import pyplot as plt
import lifelines as lf


class TruncGamma:
    def __init__(self, a, b, l=0.0, r=np.inf):
        self.l = l
        self.r = r
        self.gamma = st.gamma(a=a, scale=1 / b)

    def rvs(self, size):
        l = self.gamma.cdf(self.l)
        r = self.gamma.cdf(self.r)
        unif = np.random.uniform(low=l, high=r, size=size)
        return self.gamma.ppf(unif)

    def pdf(self, x):
        p = self.gamma.cdf(self.r) - self.gamma.cdf(self.l)
        ind = 1.0 * ((x >= self.l) & (x <= self.r))
        return ind * self.gamma.pdf(x) / p


def TruncExpon(theta, size, L=0.0, R=np.inf):
    l, r = np.array(1 - np.exp(-theta * L)), np.array(1 - np.exp(-theta * R))
    return -np.log(1 - (l + (r - l) * np.random.uniform(size=size))) / theta


class TwoLine:
    def __init__(self, a1, b1, c, a2, b2):
        self.a1 = a1
        self.b1 = b1
        self.c = c
        self.a2 = a2
        self.b2 = b2

    def hazard(self, x):
        h = np.zeros_like(x)
        flag1 = (x >= 0) & (x <= self.c)
        flag2 = (x > self.c) & (x <= 1)
        h[flag1] = self.a1 + self.b1 * x[flag1]
        h[flag2] = self.a2 + self.b2 * (x[flag2] - self.c)
        return h

    def rvs(self, size):
        u = st.uniform.rvs(size=size)
        z = np.log(1 / u)
        c1 = self.a1 * self.c + 0.5 * self.b1 * (self.c ** 2)
        c2 = c1 + self.a2 * (1 - self.c) + 0.5 * self.b2 * ((1 - self.c) ** 2)
        flag1 = (z <= c1)
        flag2 = (z > c1) & (z <= c2)
        samples = 2 * np.ones(size)
        if self.b1 != 0:
            samples[flag1] = (np.sqrt(self.a1 ** 2 + 2 * self.b1 * z[flag1]) - self.a1) / self.b1
        else:
            samples[flag1] = z[flag1] / self.a1

        if self.b2 != 0:
            samples[flag2] = (np.sqrt(self.a2 ** 2 + 2 * self.b2 * (z[flag2] - c1)) - self.a2) / self.b2 + self.c
        else:
            samples[flag2] = (z[flag2] - c1) / self.a2 + self.c

        return samples


class ExponWeib:
    def __init__(self, a, c, scale):
        self.exponweib = st.exponweib(a=a, c=c, scale=scale)

    def hazard(self, x):
        return self.exponweib.pdf(x) / (1 - self.exponweib.cdf(x))

    def rvs(self, size):
        return self.exponweib.rvs(size=size)


def sur_null():
    curves = [ExponWeib(a=1, c=0.5, scale=1),
              TwoLine(a1=2, b1=0, c=0.5, a2=0.5, b2=0),
              TwoLine(a1=1, b1=10, c=0.2, a2=3, b2=0),
              ExponWeib(a=1, c=1, scale=1),
              ExponWeib(a=1, c=2, scale=1)]
    return curves


def sur_nonnull():
    curves = [[ExponWeib(a=1, c=0.5, scale=1), ExponWeib(a=1, c=0.5, scale=0.5)],
              [TwoLine(a1=1, b1=0, c=0.5, a2=1, b2=2), TwoLine(a1=1.5, b1=0, c=0.5, a2=1.5, b2=3)],
              [TwoLine(a1=0.25, b1=3, c=0.5, a2=1.75, b2=1), TwoLine(a1=1.75, b1=0, c=0.5, a2=1.75, b2=1)],
              [TwoLine(a1=2, b1=0, c=0.2, a2=2, b2=-2), TwoLine(a1=2, b1=0, c=0.2, a2=2, b2=4)],
              [TwoLine(a1=0.5, b1=1.5, c=1, a2=0, b2=0), TwoLine(a1=2, b1=-1.5, c=1, a2=0, b2=0)],
              [TwoLine(a1=0.5, b1=1, c=0.5, a2=1, b2=1), TwoLine(a1=1.5, b1=0, c=0.5, a2=0.5, b2=0)],
              [ExponWeib(a=1, c=1, scale=1), ExponWeib(a=1, c=1, scale=0.5)],
              [ExponWeib(a=1, c=2, scale=1), ExponWeib(a=1, c=2, scale=0.75)],
              [TwoLine(a1=1, b1=0, c=0.5, a2=2, b2=0), TwoLine(a1=1.5, b1=0, c=0.5, a2=3, b2=0)],
              [TwoLine(a1=1, b1=0, c=0.25, a2=3, b2=0), TwoLine(a1=2, b1=0, c=0.25, a2=2, b2=0)],
              [TwoLine(a1=1, b1=10, c=0.2, a2=3, b2=0), TwoLine(a1=3, b1=-10, c=0.2, a2=1, b2=0)],
              [TwoLine(a1=1, b1=2.5, c=0.8, a2=3, b2=0), TwoLine(a1=3, b1=-2.5, c=0.8, a2=1, b2=0)]]
    return curves


def main():
    plt.style.use('ggplot')

    # fig, ax = plt.subplots(figsize=[6, 4])
    # grid_x = np.linspace(3.5, 13.5, 1000)
    # tg = TruncGamma(a=1, b=0.25, l=4.0, r=13.0)
    # ax.plot(grid_x, tg.pdf(grid_x))
    # ax.hist(tg.rvs(size=10000), bins=50, density=True, histtype='step', label='TG')
    # ax.hist(TruncExpon(theta=0.25, size=10000, L=4.0, R=13.0), bins=50, density=True, histtype='step', label='TE')
    # ax.legend()
    # fig.show()

    # fig, ax = plt.subplots(3, 3, figsize=[12, 8])
    # ax = ax.flatten()
    plt.rcParams["figure.figsize"] = (4, 3)
    for i, survival in enumerate(sur_null()):
        T = survival.rvs(size=100000)
        D = (T <= 1)
        X = T * D + (1 - D)

        grid_x = np.linspace(0, 1, 1000)
        # naf = lf.NelsonAalenFitter().fit(X, D, timeline=grid_x, label='naf')
        plt.plot(grid_x, survival.hazard(grid_x), label=r'$h_0(t)=h_1(t)$')
        # naf.plot_hazard(ax=ax[i], bandwidth=0.05)
        plt.ylim([0, 6])
        plt.legend()
        # plt.title('H0C{}'.format(i + 1))
        plt.xlabel(r'$t$')
        plt.ylabel(r'$h_i(t)$')
        plt.tight_layout()
        plt.show()

    # fig.tight_layout()
    # fig.show()

    # fig, ax = plt.subplots(3, 4, figsize=[15, 8])
    # ax = ax.flatten()
    for i, survival in enumerate(sur_nonnull()):
        T1 = survival[0].rvs(size=100000)
        T2 = survival[1].rvs(size=100000)
        D1 = (T1 <= 1)
        D2 = (T2 <= 1)
        X1 = T1 * D1 + (1 - D1)
        X2 = T2 * D2 + (1 - D2)

        grid_x = np.linspace(0, 1, 1000)
        # naf1 = lf.NelsonAalenFitter().fit(X1, D1, timeline=grid_x, label='naf1')
        # naf2 = lf.NelsonAalenFitter().fit(X2, D2, timeline=grid_x, label='naf2')
        plt.plot(grid_x, survival[0].hazard(grid_x), label=r'$h_0(t)$')
        # naf1.plot_hazard(ax=ax[i], bandwidth=0.05)
        plt.plot(grid_x, survival[1].hazard(grid_x), label=r'$h_1(t)$')
        # naf2.plot_hazard(ax=ax[i], bandwidth=0.05)
        plt.ylim([0, 6])
        plt.legend()
        plt.xlabel(r'$t$')
        plt.ylabel(r'$h_i(t)$')
        # plt.title('H1C{}'.format(i + 1))
        plt.tight_layout()
        plt.show()

    # fig.tight_layout()
    # fig.show()


if __name__ == '__main__':
    main()
