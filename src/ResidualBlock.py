import tensorflow as tf
from tensorflow.python import keras
from keras.layers import Conv1D


class ResBlock(keras.Model):

    def __init__(self, filters, kernel_size, dilation_rate, trainable=True, *args, **kwargs):
        super().__init__(trainable=trainable, *args, **kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate

        self.dilatedConv = Conv1D(filters, kernel_size, dilation_rate=dilation_rate, padding='causal')
        self.gatedConv = Conv1D(filters, kernel_size, dilation_rate=dilation_rate, padding='causal')
        self.oneConv = Conv1D(filters, 1, padding='same')

        self.condConv = Conv1D(filters, 1, padding='same')
        self.gatedCondConv = Conv1D(filters, 1, padding='same')

        if not trainable:
            self.queue = self._init_queue()

    @tf.function
    def call(self, inputs, conds):
        t = tf.tanh(self.dilatedConv(inputs) + self.condConv(conds))
        s = tf.sigmoid(self.gatedConv(inputs) + self.gatedCondConv(conds))

        z = t * s
        z = self.oneConv(z)

        return z + inputs, z

    def feed(self, inputs, conds):
        # Pop previous value and push new input value.
        self.queue = tf.concat([self.queue[:, 1:, :], inputs], axis=1)  # inputs: (1, 1, 256)
        q = self.queue

        t = tf.tanh(self.dilatedConv(q) + self.condConv(conds))
        s = tf.sigmoid(self.gatedConv(q) + self.gatedCondConv(conds))
        z = t * s
        z = self.oneConv(z)
        z = z[:, -1:, :]

        return z + inputs, z

    def _init_queue(self):
        return tf.zeros((1, (self.kernel_size - 1) * self.dilation_rate + 1, self.filters))


if __name__ == '__main__':
    input_shape = (1, 16000, 1)
    x = tf.random.normal(input_shape)

    resBlock = ResBlock(32, 4, 2)
    x, s = resBlock(x)

    print(x.shape)
    print(s.shape)

