from BLRT2.simulation.real import read, plot
import numpy as np
import lifelines as lf
from matplotlib import pyplot as plt
import pickle


def read2():
    file = open('../data/real', 'rb')
    data = np.array(pickle.load(file))
    tmp = data.copy()
    data[:, :3] = tmp[:, 3:]
    data[:, 3:] = tmp[:, :3]
    return data


def plot2(data, name, ax):
    km = lf.KaplanMeierFitter().fit(data[0], data[2], label='$\hat{S}_0$ (control)')
    km.plot_survival_function(ax=ax)
    km = lf.KaplanMeierFitter().fit(data[1], data[3], label='$\hat{S}_1$ (treatment)')
    km.plot_survival_function(ax=ax)
    md = np.median(np.append(data[0], data[1]))
    # ax.plot(md, 0, 'x', label='median')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\hat{S}_i(t)$')
    ax.legend(loc=3)
    # ax.set_title(name)


def compare(data, ax):
    titles = ['CGD (nd)', 'CGD (ed)', 'CGD (ld)', 'DRS (nd)', 'DRS (ed)', 'DRS (ld)']
    for i, a in enumerate(ax):
        a.plot(data[:, i, 0], data[:, i, 1], '.')
        a.plot(data[:, i, 0], data[:, i, 0])
        a.set_xlabel('p-value')
        a.set_ylabel('Prob. of Null')
        if i not in [0, 3]:
            a.set_xlim([0, 0.105])
            a.set_ylim([0, 0.105])

        if i == 4:
            a.set_xlim([0, 0.0105])
            a.set_ylim([0, 0.0105])

        a.set_title(titles[i])


def main(alpha):
    plt.style.use('ggplot')

    fig, ax = plt.subplots(1, 2, figsize=[9, 4])
    data = read('CGD')
    plot(data, 'CGD', ax[0])
    data = read('DRS')
    plot(data, 'DRS', ax[1])
    fig.tight_layout()
    fig.show()

    # fig, ax = plt.subplots(1, 2, figsize=[9, 4])
    fig, ax = plt.subplots(figsize=[5, 4])
    data = read('CGD')
    plot2(data, 'CGD', ax)
    fig.tight_layout()
    fig.show()
    fig, ax = plt.subplots(figsize=[5, 4])
    data = read('DRS')
    plot2(data, 'DRS', ax)
    fig.tight_layout()
    fig.show()

    data = read2()
    data_cr = data[:, :, :4]
    data_pv = data[:, :, 4:]

    fig, ax = plt.subplots(2, 3, figsize=[10, 6])
    compare(data_pv, ax.flatten())
    fig.tight_layout()
    fig.show()

    print('censoring rate: ')
    print(data_cr.mean(axis=0)[:, :2])
    print('size (alpha={}): '.format(alpha))
    print(np.mean(data_pv <= alpha, axis=0))


if __name__ == '__main__':
    main(alpha=0.05)
