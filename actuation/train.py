import sys
import tensorflow as tf
from argparse import ArgumentParser
from tensorflow import GradientTape
from tensorflow.keras.optimizers import Adam

class Train():
    def __init__(self, args):
        # initialize tensorflow
        # print(tf.config.experimental.list_physical_devices('GPU'))
        # tf.config.experimental.set_memory_growth(gpu, True)
        sys.setrecursionlimit(10000)

        # import classes
        sys.path.append("..")
        from models.model import Model
        from preprocessing.process import Processor

        # convert args to variables
        self.learningRate = args.lr
        self.batchSize = args.bs
        self.epochs = args.ep
        self.boardSize = args.dim

        # initialize model and optimizer
        self.model = Model(10)
        self.optimizer = Adam(args.lr)

        # initialize preprocessor
        self.processor = Processor(self.batchSize)
        self.boards, self.starts, self.ends = self.processor()

    def __call__(self):
        for epoch in range(self.epochs):
            for i in range(100000):
                self._update(self.boards[i], self.starts[i], self.ends[i])
            print('epoch completed')

    def _update(self, board, start, end):
        with GradientTape() as tape:
            loss = self.model(board, start, end, training=True)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))


if __name__ == '__main__':
    # create argument parser
    parser = ArgumentParser()
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--bs', default=1, type=int)
    parser.add_argument('--ep', default=1000, type=int)
    parser.add_argument('--dim', default=10, type=int)

    args = parser.parse_args()

    # train model
    train = Train(args)
    train()