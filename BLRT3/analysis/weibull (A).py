import numpy as np
from matplotlib import pyplot as plt
import pickle


def read(sizeX=100):
    file = open('../data/weibull_{}'.format(sizeX), 'rb')
    data = pickle.load(file)
    return np.array(data)


def main():
    data = read()
    rejection = (data[:, :, 5:] <= 0.05).mean(axis=0)
    print(np.round(rejection, 3))


if __name__ == '__main__':
    main()
