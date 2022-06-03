import tensorflow as tf
from tensorflow.python.keras.losses import CategoricalCrossentropy
from tensorflow.python.keras.optimizers import adam_v2
import numpy as np
import soundfile as sf

from PreProcess import PreProcess
from WaveNet import WaveNet
from MuLaw import iMuLaw

@tf.function
def train_step(model, x, c, loss_fn, optimizer):
    input_len = x.shape[1] - 1
    with tf.GradientTape() as tape:
        y = model(x[:, :input_len, :], c)
        loss = loss_fn(x[:, 1:, :], y)

    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return loss


if __name__ == '__main__':
    filePath = 'C:\\Users\\nyoon\\Documents\\VCTK-Corpus'
    preProcess = PreProcess(filePath, 16000 * 3)

    model = WaveNet(trainable=True)
    loss_fn = CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    numWav = preProcess.getNumWav()
    step = int(numWav / 100)

    for e in range(300):
        loss = 0.0

        print('epoch: {}'.format(e))
        print('[', end='')

        i = 0
        for x, c in preProcess.getTrainData():
            loss += train_step(model, x, c, loss_fn, optimizer)

            i += 1

            if i % step == 0:
                print('*', end='')

        print(']\nloss: {}'.format(loss / numWav))

        model.save_weights('..\\model\\model')
