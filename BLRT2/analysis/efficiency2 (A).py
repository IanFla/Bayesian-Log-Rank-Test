import numpy as np
from matplotlib import pyplot as plt
import pickle


def read():
    file = open('../data/efficiency2', 'rb')
    data = pickle.load(file)
    return np.array(data)


def main():
    data = read()
    data[:, :, :, 0, :][data[:, :, :, 0, :] > 10000] = 10000
    data = data.mean(axis=0).mean(axis=3)
    print(np.round(data, 2))


if __name__ == '__main__':
    main()
