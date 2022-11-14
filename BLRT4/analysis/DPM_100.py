import numpy as np
import pickle


def read(sizeX=100):
    file = open('../results/DPM_{}'.format(sizeX), 'rb')
    data = pickle.load(file)
    return np.array(data)


def main():
    data = read()
    data[:, :, 4:] = (data[:, :, 4:] <= 0.05)
    data = data[:, :, 5]
    print(np.round(data.mean(axis=0), 3))


if __name__ == '__main__':
    main()
