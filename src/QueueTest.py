import tensorflow as tf
from keras.layers import Conv1D
import numpy as np
import soundfile as sf

from ResidualBlock import ResBlock
from WaveNet import WaveNet
from PreProcess import PreProcess
from MuLaw import muLaw, iMuLaw


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


def test3():
    model = WaveNet(trainable=False)
    model.load_weights('..\\model\\model')

    filePath = 'C:\\Users\\nyoon\\Documents\\VCTK-Corpus'
    preProcess = PreProcess(filePath, 16000 * 3)

    testInput, c = preProcess.getSeed()
    testInput = tf.expand_dims(testInput, axis=0)

    length = testInput.shape[1]
    result = testInput[:, :1, :]

    for i in range(length - 1):
        output = model.synthesize(testInput[:, i:i+1, :], c)
        result = tf.concat([result, output], axis=1)

        if i % 100 == 0:
            print(i)

    result = tf.argmax(result[0, :, :], axis=-1)
    result = iMuLaw(np.array(result))

    sf.write('..\\syn\\test3.wav', result, 16000, subtype='PCM_16')


def test4():
    model = WaveNet(trainable=False)
    model.load_weights('..\\model\\model')

    filePath = 'C:\\Users\\nyoon\\Documents\\VCTK-Corpus'
    preProcess = PreProcess(filePath, 16000 * 3)

    testInput, c = preProcess.getSeed()
    testInput = testInput[16000:, :]
    testInput = tf.expand_dims(testInput, axis=0)

    length = testInput.shape[1]
    result = testInput[:, :1, :]

    if length <= 16000:
        return

    for i in range(16000):
        output = model.synthesize(testInput[:, i:i+1, :], c)

        output = tf.argmax(output[0], axis=-1)
        output = tf.one_hot(output, 256)
        output = tf.reshape(output, (1, -1, 256))

        result = tf.concat([result, output], axis=1)

        if i % 100 == 0:
            print(i)

    for i in range(16000 * 2):
        output = model.synthesize(result[:, -1:, :], c)

        output = tf.argmax(output[0], axis=-1)
        output = tf.one_hot(output, 256)
        output = tf.reshape(output, (1, -1, 256))

        result = tf.concat([result, output], axis=1)

        if i % 100 == 0:
            print(16000 + i)

    result = tf.argmax(result[0, :, :], axis=-1)
    result = iMuLaw(np.array(result))

    sf.write('..\\syn\\test3.wav', result, 16000, subtype='PCM_16')


if __name__ == '__main__':
    test4()
