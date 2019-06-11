import sys
import numpy as np
import tensorflow as tf


class Logger(object):
    def __init__(self, filename="default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()


def plot_binary(codes):
    size, scale = 1000, 10
    commutes = pd.Series(codes)

    commutes.plot.hist(grid=True, bins=200, rwidth=0.9,
         color='#607c8e')
    plt.title('codes distributions')
    plt.xlabel('code')
    plt.ylabel('Counts')
    plt.grid(axis='y', alpha=0.75)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


# return -1 if x < 0, 1 if x > 0, random -1 or 1 if x ==0
def sign(x):
    s = np.sign(x)
    tmp = s[s == 0]
    s[s==0] = np.random.choice([-1, 1], tmp.shape)
    return s


def reduce_shaper(t):
    return tf.reshape(tf.reduce_sum(t, 1), [tf.shape(t)[0], 1])


if __name__ == "__main__":
    x = np.random.choice([-1, 0, 1], [5, 5])
    print(x)
    print((sign(x)))
