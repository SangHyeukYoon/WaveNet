import tensorflow as tf
from keras.layers import Conv1D
import numpy as np

from ResidualBlock import ResBlock
from WaveNet import WaveNet
from PreProcess import PreProcess


def test1():
    model = WaveNet(trainable=False)
    model.load_weights('..\\model\\model')

    filePath = 'C:\\Users\\nyoon\\Documents\\VCTK-Corpus\\wav48\\p225\\'
    preProcess = PreProcess(filePath, 16000 * 2)

    seed = preProcess.getSeed()
    seed = tf.reshape(seed, (1, -1, 256))

    syn = seed
    for i in range(1000):
        out = model(syn)
        proba = tf.argmax(out[0, -1:, :], axis=-1)
        print(proba)
        proba = tf.one_hot(proba, 256)
        proba = tf.expand_dims(proba, axis=0)
        syn = tf.concat([syn, proba], axis=1)


def test2():
    model = WaveNet(trainable=False)
    model.load_weights('..\\model\\model')

    for s in range(256):
        seed = s
        seed = tf.one_hot(seed, 256)
        seed = tf.reshape(seed, (1, -1, 256))

        out = seed
        result = []
        for i in range(100):
            out = model.synthesize(out)
            proba = tf.argmax(out[0, -1:, :], axis=-1)

            result.append(int(proba))

            proba = tf.one_hot(proba, 256)
            out = tf.expand_dims(proba, axis=0)

        print(s)
        print(result)
        print()


if __name__ == '__main__':
    test2()
    # model = WaveNet(trainable=True)
    # inputs = tf.random.normal((1, 1, 256))
    #
    # output = model.synthesize(inputs)
    # print(output.shape)
    # val = tf.argmax(output, axis=-1)
    # print(val)
    #
    # output = model(inputs)
    # val = tf.argmax(output, axis=-1)
    # print(val)

    # block = ResBlock(128, 4, 4)
    # inputs = np.zeros((1, 1, 128))
    #
    # output, skips = block.feed(inputs)
    # print(output.shape)
    # print(skips.shape)

    # conv = Conv1D(16, 3, dilation_rate=4, padding='causal')
    # inputs = tf.random.normal((1, 12, 128))
    #
    # output = conv(inputs)
    # print(output[0, 8, 0])
    #
    # output = conv(inputs[:, :9, :])
    # print(output[0, -1, 0])
