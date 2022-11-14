import numpy as np
from BLRT4.sampling.MDP import MixDirProcess
from BLRT4.sampling.UIP import UniInfPrior
from lifelines.statistics import logrank_test

from BLRT4.sampling.cases import survival_null, survival_nonnull, Weibull


class LogRankTest:
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

    def blrt(self, size, M=5, prior=None, size_G=1000):
        mdp1 = MixDirProcess(self.X1, self.D1, size, M=M, G0=Weibull(prior[0][0], prior[0][1]))
        mdp2 = MixDirProcess(self.X2, self.D2, size, M=M, G0=Weibull(prior[1][0], prior[1][1]))
        tau = min(mdp1.X[0], mdp2.X[0])
        grid_x = np.linspace(0.0, tau, size_G)
        mdp1.prior(a=4.0, b=4.0)
        mdp1.imputation()
        mdp1.sampling(grid_x)
        mdp2.prior(a=4.0, b=4.0)
        mdp2.imputation()
        mdp2.sampling(grid_x)

        N1 = np.sum(grid_x[:-1].reshape([-1, 1]) <= mdp1.X, axis=1)
        N2 = np.sum(grid_x[:-1].reshape([-1, 1]) <= mdp2.X, axis=1)
        NN = N1 * N2 / (N1 + N2)
        H1 = self.divi(mdp1.S[:, 1:] - mdp1.S[:, :-1], mdp1.S[:, :-1])
        H2 = self.divi(mdp2.S[:, 1:] - mdp2.S[:, :-1], mdp2.S[:, :-1])
        Q1 = np.sum(NN * H1, axis=1)
        Q2 = np.sum(NN * H2, axis=1)

        Q = (Q1.reshape([-1, 1]) - Q2).flatten()
        PoP = 2 * min(np.mean(Q > 0), np.mean(Q < 0))
        self.p.append(PoP)
        return Q


def main(H, num, scaleC=1.0, sizeX=100, sizeB=1000):
    if H:
        survival1, survival2 = survival_nonnull()[num]
    else:
        survival1, survival2 = survival_null()[num]

    history1 = []
    history2 = []
    for i in range(5):
        T1 = survival1.rvs(size=sizeX)
        T2 = survival2.rvs(size=sizeX)
        C1 = np.random.exponential(scale=scaleC, size=sizeX)
        C2 = np.random.exponential(scale=scaleC, size=sizeX)
        C1[C1 >= 1.0] = 1.0
        C2[C2 >= 1.0] = 1.0
        D1 = (T1 <= C1)
        D2 = (T2 <= C2)
        X1 = T1 * D1 + C1 * (1 - D1)
        X2 = T2 * D2 + C2 * (1 - D2)
        history1.append([X1, D1])
        history2.append([X2, D2])

    uip1 = UniInfPrior(history1)
    uip1.fit(10)
    uip2 = UniInfPrior(history2)
    uip2.fit(10)
    prior = [uip1.params, uip2.params]
    print(prior)

    T1 = survival1.rvs(size=sizeX)
    T2 = survival2.rvs(size=sizeX)
    C1 = np.random.exponential(scale=scaleC, size=sizeX)
    C2 = np.random.exponential(scale=scaleC, size=sizeX)
    C1[C1 >= 1.0] = 1.0
    C2[C2 >= 1.0] = 1.0
    D1 = (T1 <= C1)
    D2 = (T2 <= C2)
    X1 = T1 * D1 + C1 * (1 - D1)
    X2 = T2 * D2 + C2 * (1 - D2)
    print('censoring rate 1: {} (=1: {})'.format(1 - D1.mean(), np.mean(X1 == 1)))
    print('censoring rate 2: {} (=1: {})'.format(1 - D2.mean(), np.mean(X2 == 1)))

    lrt = LogRankTest(X1, X2, D1, D2)
    lrt.clrt()
    lrt.blrt(size=sizeB, M=100, prior=prior, size_G=1000)
    print(lrt.p)


if __name__ == '__main__':
    main(0, 1)
    main(1, 1)
