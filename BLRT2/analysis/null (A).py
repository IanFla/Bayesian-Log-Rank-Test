import numpy as np
from matplotlib import pyplot as plt
import pickle


def read(sizeX):
    file = open('../data/null({})'.format(sizeX), 'rb')
    data = pickle.load(file)
    return np.array(data)


def compare(data, ax):
    for i, a in enumerate(ax):
        a.plot(data[:, i, 0], data[:, i, 1], '.')
        a.plot(data[:, i, 0], data[:, i, 0])
        a.set_xlabel('p-value')
        a.set_ylabel('Prob. of Null')
        a.set_title('H0C' + str(i + 1))


def main(sizeX, alpha):
    plt.style.use('ggplot')

    data = read(sizeX)
    data_cr = data[:, :, :4]
    data_pv = data[:, :, 4:]

    # fig, ax = plt.subplots(3, 3, figsize=[12, 8])
    # compare(data_pv[:, :, [2, 3]], ax.flatten())
    # fig.tight_layout()
    # fig.show()

    print('censoring rate: ')
    print(data_cr.mean(axis=0)[:, :2])
    print('size (alpha={}): '.format(alpha))
    print(np.mean(data_pv <= alpha, axis=0))


def main2(n):
    plt.style.use('ggplot')
    sizeX = [30, 100, 300]
    grid_x = np.linspace(0, 1, 1000).reshape([-1, 1])
    for i, size in enumerate(sizeX):
        data = read(size)
        data_pv = data[:, :, 4:]
        fig, ax = plt.subplots(figsize=[4.5, 4])
        ax.plot(grid_x.flatten(), np.mean(grid_x >= data_pv[:, n - 1, 0], axis=1))
        ax.plot(grid_x.flatten(), np.mean(grid_x >= data_pv[:, n - 1, 1], axis=1))
        ax.legend(['CLRT', 'BLRT'])
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel('Rejection Proportions')
        # plt.title('H0C{}({})'.format(n, size))
        fig.tight_layout()
        fig.show()

    # fig.tight_layout()
    # fig.show()


if __name__ == '__main__':
    main(sizeX=100, alpha=0.05)
    main2(1)
