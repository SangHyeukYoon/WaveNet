import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

import numpy as np
import soundfile as sf

from PreProcess import PreProcess
from WaveNet import WaveNet
from MuLaw import muLaw, iMuLaw


if __name__ == '__main__':
    filePath = 'C:\\Users\\nyoon\\Documents\\VCTK-Corpus'
    preProcess = PreProcess(filePath, 16000 * 2)

    model = WaveNet(trainable=False)
    model.load_weights('..\\model\\model')

    seed = 11
    seed = tf.one_hot(seed, 256)
    seed = tf.reshape(seed, (1, -1, 256))

    numSpk = preProcess.getNumSpk()

    c = 10
    c = tf.one_hot(c, numSpk)
    c = tf.reshape(c, (1, -1, numSpk))

    syn = seed
    out = seed
    for i in range(1000):
        out = model.synthesize(out, c)
        out = tf.argmax(out[0], axis=1)
        # print(out)
        out = tf.one_hot(out, 256)
        out = tf.reshape(out, (1, -1, 256))
        syn = tf.concat([syn, out], axis=1)

        if i % 100 == 0:
            print()
            # print('{}'.format(i), end='')
            t = i // 100
            res = tf.argmax(syn[0, (t-1)*100:t*100, :], axis=1)
            print(res)

        # print('.', end='')

    temp = tf.argmax(syn[0], axis=1)
    temp = iMuLaw(np.array(temp))
    sf.write('..\\syn\\syn.wav', temp, 16000, subtype='PCM_16')
