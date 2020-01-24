import tensorflow as tf
from tensorflow.keras.layers import *

class Model(tf.keras.Model):
    def __init__(self, boardSize):
        super().__init__()

        self.conv2d = Conv2D(24, (int(boardSize/2), int(boardSize/2)), strides=(1, 1), padding='same')
        self.concat = Concatenate()

        self.dense1 = Dense(10, activation='relu')
        self.dense2 = Dense(12, activation='relu')
        self.dense3 = Dense(8, activation='relu')

        self.lambda1 = Lambda(lambda x: (x[0][x[1]]))

    def __call__(self, board, start, end, training=False):
        b = self.conv2d(board)
        b = self.concat([board, b])
        print(b.shape)

        current = start
        moves = [[1, 1], [1, 0], [1, -1], [0, 1], [0, -1], [-1, 1], [-1, 0], [-1, -1]]

        index = tf.cast([start], tf.int32)
        index = tf.squeeze(index)
        print(index.shape)
        s = self.lambda1((b, index))

        while current[0] != end[0] or current[1] != end[1]:
            gen = [[current[0] + moves[i][0], current[1] + moves[i][1]] for i in range(8)]
            vals = tf.convert_to_tensor([b[gen[i][0]][gen[i][1]] for i in range(8)])
            
            x = self.dense1(vals)
            x = self.dense2(x)
            x = self.dense3(x)

            chosen = tf.math.argmax(x)

            current = gen[chosen]
            s += vals[chosen]

        return s

        