import numpy as np


def muLaw(x, mu=256):
    x = x.astype(float)
    y = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    y = np.digitize(y, 2 * np.arange(mu) / mu - 1) - 1
    return y.astype(int)


def iMuLaw(y, mu=256):
    # y = y.astype(float)
    y = 2 * y / mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x


if __name__ == '__main__':
    x = [-0.87, 0.95, 0]
    x = np.asarray(x)
    x = muLaw(x)
    print(x)
    print(iMuLaw(x))
