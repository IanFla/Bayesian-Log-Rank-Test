import numpy as np
from matplotlib import pyplot as plt
import pickle


def read(sizeX):
    file = open('../data/non-null({})'.format(sizeX), 'rb')
    data = pickle.load(file)
    return np.array(data)


def compare(data, ax):
    for i, a in enumerate(ax):
        a.plot(data[:, i, 0], data[:, i, 1], '.')
        a.plot(data[:, i, 0], data[:, i, 0])
        a.set_xlabel('p-value')
        a.set_ylabel('Prob. of Null')
        # a.set_xlim([0, 0.1])
        # a.set_ylim([0, 0.1])
        a.set_title('H0C' + str(i + 1))


def main(sizeX, alpha):
    plt.style.use('ggplot')

    data = read(sizeX)
    data_cr = data[:, :, :4]
    data_pv = data[:, :, 4:]

    # fig, ax = plt.subplots(3, 4, figsize=[15, 10])
    # compare(data_pv[:, :, [0, 2]], ax.flatten())
    # fig.tight_layout()
    # fig.show()

    print('censoring rate: ')
    print(data_cr.mean(axis=0)[:, :2])
    print('size (alpha={}): '.format(alpha))
    print(np.mean(data_pv <= alpha, axis=0))


if __name__ == '__main__':
    main(sizeX=100, alpha=0.05)
