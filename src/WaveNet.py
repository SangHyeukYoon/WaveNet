import tensorflow as tf
from tensorflow.python import keras
from keras.layers import Conv1D, ReLU

import numpy as np

from ResidualBlock import ResBlock


class WaveNet(keras.Model):

    def __init__(self, trainable=True, *args, **kwargs):
        super().__init__(trainable=trainable, *args, **kwargs)

        self.firstLayer = Conv1D(128, 1, padding='causal', trainable=trainable)

        self.resBlocks = []
        for _ in range(2):
            for i in range(10):
                self.resBlocks.append(ResBlock(128, 3, 2 ** i, trainable=trainable))

        self.finalLayers = [
            ReLU(), Conv1D(128, 1, padding='same', trainable=trainable),
            ReLU(), Conv1D(256, 1, padding='same', activation='softmax', trainable=trainable)
        ]

    @tf.function
    def call(self, inputs, conds):
        x = self.firstLayer(inputs)

        skips = np.zeros(x.shape, dtype=float)
        for block in self.resBlocks:
            x, h = block(x, conds)
            skips += h

        x = skips
        for layer in self.finalLayers:
            x = layer(x)

        return x

    def synthesize(self, inputs, conds):
        x = self.firstLayer(inputs)

        skips = np.zeros(x.shape, dtype=float)
        for block in self.resBlocks:
            x, h = block.feed(x, conds)
            skips += h

        x = skips
        for layer in self.finalLayers:
            x = layer(x)

        return x


if __name__ == '__main__':
    input_shape = (1, 16000, 1)
    x = tf.random.normal(input_shape)

    waveNet = WaveNet()
    x = waveNet(x)
    print(x.shape)
