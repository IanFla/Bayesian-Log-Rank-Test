import numpy as np
from matplotlib import pyplot as plt
import pickle


def read(sizeX):
    file = open('../data/null_inf_adj_{}'.format(sizeX), 'rb')
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

    fig, ax = plt.subplots(3, 3, figsize=[12, 8])
    compare(data_pv[:, :, [2, 3]], ax.flatten())
    fig.tight_layout()
    fig.show()

    print('censoring rate: ')
    print(data_cr.mean(axis=0))
    print('size (alpha={}): '.format(alpha))
    print(np.mean(data_pv <= alpha, axis=0))


def main2():
    plt.style.use('ggplot')
    data = read(100)
    data_pv = data[:, :, 4:]
    print(np.round(data[:, :, :4].mean(axis=0), 3))
    print(np.round((data_pv <= 0.05).mean(axis=0), 3))

    # params = [[10, 1, 1], [100, 1, 1], [1000, 1, 1], [10, 1, 2], [100, 1, 2], [1000, 1, 2]]
    # for i in range(5):
    #     fig, ax = plt.subplots(2, 3, figsize=[10, 6])
    #     ax = ax.flatten()
    #     for j, a in enumerate(ax):
    #         a.plot(data_pv[:, i, 0], data_pv[:, i, j + 1], '.')
    #         a.plot(data_pv[:, i, 0], data_pv[:, i, 0])
    #         a.set_xlabel('$p$-value')
    #         a.set_ylabel('Prob. of Null')
    #         a.set_title(params[j])
    #
    #     fig.tight_layout()
    #     fig.show()


if __name__ == '__main__':
    main2()
