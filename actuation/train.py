from argparse import ArgumentParser

class Train():
    def __init__(self, args):
        # convert args to variables
        self.learningRate = args.lr
        self.batchSize = args.bs
        self.epochs = args.ep

        # initialize model and optimizer
        self.model = Model()


    def __call__(self):

if __name__ == '__main__':
    # create argument parser
    parser = ArgumentParser()
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--bs', default=64, type=int)
    parser.add_argument('--ep', default=1000, type=int)

    args = parser.parse_args()

    # train model
    train = Train()
    train()