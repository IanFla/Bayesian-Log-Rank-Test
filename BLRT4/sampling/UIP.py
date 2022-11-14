import numpy as np
import scipy.optimize as opt

from BLRT4.sampling.cases import survival_nonnull, Weibull
import matplotlib.pyplot as plt


class UniInfPrior:
    def __init__(self, history, hyper=1000):
        self.history = history
        self.B = len(history)
        self.hyper = hyper
        self.weights = np.ones(self.B) / self.B
        self.params = np.array([1.0, 1.0])
        self.timeline = np.sort(np.hstack([list(set(data[0][data[1]])) for data in history]))
        self.estimates = [self.estimate(data) for data in self.history]

    def estimate(self, data):
        X, D = data
        ind = np.argsort(X)
        X = X[ind]
        D = D[ind]

        num = np.sum(X[D].reshape([-1, 1]) == self.timeline, axis=0)
        den = np.sum(X.reshape([-1, 1]) >= self.timeline, axis=0)
        num[den == 0] = 0.0
        den[den == 0] = 1.0
        dL = num / den
        L1 = np.cumsum(dL)

        dL = np.sum(X[D].reshape([-1, 1]) == X[D], axis=0) / \
             np.sum(X.reshape([-1, 1]) >= X[D], axis=0)
        L2 = np.cumsum(dL)
        L2 = np.interp(self.timeline, np.append(0, X[D]), np.append(0, L2))

        num = np.sum(X.reshape([-1, 1]) >= self.timeline, axis=0) ** 2
        den = len(X) * np.sum(X[D].reshape([-1, 1]) == self.timeline, axis=0)
        num[den == 0] = 0.0
        den[den == 0] = 1.0
        IdL = num / den

        T = np.append(np.append(0.0, X[D]), np.inf)
        ind = np.argmax(self.timeline.reshape([-1, 1]) <= T, axis=1)
        square = (T[ind] - T[ind - 1]) ** 0.5
        square[square == np.inf] = 0.0
        # square = 0.1 * np.ones_like(self.timeline)
        return [L1, L2, IdL, square]

    def combine1(self, weights):
        weights = weights / weights.sum()
        L = np.sum([weights[b] * self.estimates[b][0] for b in range(self.B)], axis=0)
        P = np.append(0, 1 - np.exp(-L))
        return P[1:] - P[:-1]

    def combine2(self, num):
        uniforms = 1 - np.linspace(1 / (2 * num), 1 - 1 / (2 * num), num)
        L = np.sum([self.weights[b] * self.estimates[b][1] for b in range(self.B)], axis=0)
        return np.interp(np.log(1 / uniforms), np.append(0, L), np.append(0, self.timeline))

    def empirical_hazard(self, t):
        timeline = np.append(np.append(0, self.timeline), np.inf)
        L = np.sum([self.weights[b] * self.estimates[b][1] for b in range(self.B)], axis=0)
        L = np.append(np.append(0, L), L[-1])
        ind = np.argmin(t.reshape([-1, 1]) >= timeline, axis=1)
        return (L[ind] - L[ind - 1]) / (timeline[ind] - timeline[ind - 1])

    def fun1(self, params):
        t = self.combine2(num=self.hyper)
        h = self.empirical_hazard(t)
        t = t[h > 0]
        h = h[h > 0]
        loss = np.mean((params[0] + (np.exp(params[0]) - 1) * np.log(t) -
                       np.exp(params[0]) * params[1] - np.log(h)) ** 2)
        return loss

    def fun2(self, weights):
        p = self.combine1(weights)
        IdL = np.sum([weights[b] * self.estimates[b][2] * self.estimates[b][3] for b in range(self.B)], axis=0)
        loss = np.sum(p * ((np.exp(-(self.timeline / self.params[1]) ** self.params[0]) *
                            (self.params[1] ** self.params[0]) /
                            (self.params[0] * (self.timeline ** (self.params[0] - 1))) - IdL) ** 2))
        return loss

    def fit(self, N):
        cons1 = {'type': 'ineq', 'fun': lambda x: x - 1e-2}
        cons2 = {'type': 'eq', 'fun': lambda x: np.array([np.sum(x) - 1.0])}
        for n in range(N):
            self.params = np.exp(opt.minimize(fun=self.fun1, x0=np.log(self.params), method='SLSQP',
                                              options={'ftol': 1e-10, 'maxiter': 1e5}).x)
            self.weights = opt.minimize(fun=self.fun2, x0=self.weights, method='SLSQP',
                                        constraints=[cons1, cons2], options={'ftol': 1e-10, 'maxiter': 1e5}).x


def main(num, i, B=5, scaleC=1.0, sizeX=100):
    survival = survival_nonnull()[num][i]
    history = []
    for b in range(B):
        T = survival.rvs(size=sizeX)
        C = np.random.exponential(scale=scaleC, size=sizeX)
        C[C >= 1.0] = 1.0
        D = (T <= C)
        X = T * D + C * (1 - D)
        history.append([X, D])

    uip = UniInfPrior(history)
    uip.fit(N=10)
    print(uip.params)
    print(uip.weights)
    grid_x = np.linspace(0.1, 1, 1000)
    plt.plot(grid_x, survival.hzd(grid_x), 'b')
    plt.plot(grid_x, Weibull(uip.params[0], uip.params[1]).hzd(grid_x), 'r')
    plt.xlim([0, 1.1])
    plt.ylim([0, 1.1 * Weibull(uip.params[0], uip.params[1]).hzd(grid_x).max()])
    plt.show()


if __name__ == '__main__':
    main(num=0, i=0)
    main(num=0, i=1)
    main(num=1, i=0)
    main(num=1, i=1)
    main(num=2, i=0)
    main(num=2, i=1)
