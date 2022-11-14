import numpy as np
from BLRT3.sampling.right_dirichlet import RightDirichlet
from lifelines.statistics import logrank_test

from BLRT3.others.distribution import sur_null
import scipy.stats as st


class RightLogRank:
    def __init__(self, X1, X2, D1, D2):
        self.X1, self.X2 = X1, X2
        self.D1, self.D2 = D1, D2
        self.p = []

    def clrt(self):
        res = logrank_test(self.X1, self.X2, self.D1, self.D2)
        self.p.append(res.p_value)

    def wlrt(self, p, q):
        res = logrank_test(self.X1, self.X2, self.D1, self.D2, weightings='fleming-harrington', p=p, q=q)
        self.p.append(res.p_value)

    @staticmethod
    def divi(num, den):
        den[num == 0] = 1
        return num / den

    def dlrt(self, size, alpha=0.001, m=1000, k=None, s=None, a=None, b=None):
        rd1 = RightDirichlet(self.X1, self.D1, size, alpha=alpha, k=k[0], s=s[0])
        rd2 = RightDirichlet(self.X2, self.D2, size, alpha=alpha, k=k[1], s=s[1])
        tau = min(rd1.X[0], rd2.X[0])
        grid_x = np.linspace(0.0, tau, m)
        rd1.prior(a=a[0], b=b[0])
        rd1.imputation()
        rd1.sampling(grid_x)
        rd2.prior(a=a[1], b=b[1])
        rd2.imputation()
        rd2.sampling(grid_x)

        N1 = np.sum(grid_x[:-1].reshape([-1, 1]) <= rd1.X, axis=1)
        N2 = np.sum(grid_x[:-1].reshape([-1, 1]) <= rd2.X, axis=1)
        NN = N1 * N2 / (N1 + N2)
        H1 = self.divi(rd1.S[:, 1:] - rd1.S[:, :-1], rd1.S[:, :-1])
        H2 = self.divi(rd2.S[:, 1:] - rd2.S[:, :-1], rd2.S[:, :-1])
        Q1 = np.sum(NN * H1, axis=1)
        Q2 = np.sum(NN * H2, axis=1)

        Q = (Q1.reshape([-1, 1]) - Q2).flatten()
        PoP = 2 * min(np.mean(Q > 0), np.mean(Q < 0))
        self.p.append(PoP)
        # mean = 4 * np.median(Q) - 3 * np.mean(Q)
        # Prob = st.norm(loc=mean, scale=Q.std()).cdf(0)
        # PoP = 2 * min(Prob, 1 - Prob)
        # self.p.append(PoP)
        return Q


def main(num, scaleC=1.0, sizeX=100, sizeB=1000):
    survival1 = survival2 = sur_null()[num]
    T1 = survival1.rvs(size=sizeX)
    T2 = survival2.rvs(size=sizeX)
    C1 = st.expon(scale=scaleC).rvs(size=sizeX)
    C2 = st.expon(scale=scaleC).rvs(size=sizeX)
    C1[C1 >= 1.0] = 1.0
    C2[C2 >= 1.0] = 1.0
    D1 = (T1 <= C1)
    D2 = (T2 <= C2)
    X1 = T1 * D1 + C1 * (1 - D1)
    X2 = T2 * D2 + C2 * (1 - D2)
    print('censoring rate 1: {} (=1: {})'.format(1 - D1.mean(), np.mean(X1 == 1)))
    print('censoring rate 2: {} (=1: {})'.format(1 - D2.mean(), np.mean(X2 == 1)))

    rlr = RightLogRank(X1, X2, D1, D2)
    rlr.clrt()
    rlr.dlrt(size=sizeB, alpha=0.001, m=1000, k=[1.0, 1.0], s=[1.0, 1.0], a=[4.0, 4.0], b=[4.0, 4.0])
    print(rlr.p)


if __name__ == '__main__':
    main(1)
