from math import sqrt

from numpy import (array, unravel_index, nditer, linalg, random, subtract,
                   power, exp, pi, zeros, arange, outer, meshgrid, dot,
                   logical_and, mean, std, cov, argsort, linspace, transpose)
from collections import defaultdict, Counter
from warnings import warn
from sys import stdout
from time import time



def _incremental_index_verbose(m):
    # Yields numbers from 0 to m-1 printing the status on the stdout.
    progress = f'\r [ {0:{len(str(m))}} / {m} ] - Time Elapsed: {0:3.0f}% ? s'
    stdout.write(progress)
    beginning = time()
    for i in range(m):
        yield i
        it_per_sec = (time() - beginning) 
        progress = f'\r [ {i+1:{len(str(m))}} / {m} ]'
        progress += f' {100*(i+1)/m:3.0f}%'
        progress += f' - Time Elapsed: {it_per_sec:2.5f} s'
        stdout.write(progress)


def fast_norm(x):
    # Norm-2 of 1D numpy array
    return sqrt(dot(x, x.T))


def asymptotic_decay(learning_rate, t, max_iter):
    # Decay function of the learning process.
    # 
    # learning_rate : float     = current learning rate.
    # t : int                   = current iteration.
    # max_iter : int             maximum number of iterations for the training.
   
    return learning_rate / (1+t/(max_iter/2))


class Som(object):
    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=0.5,random_seed=None):

        # Initializes a Self Organizing Maps
        # A map with 5*sqrt(N) neurons, where N is the number of samples, 
        # should perform well.
        # 
        # x : int         = x dimension of the SOM.
        # y : int         = y dimension of the SOM.
        # input_len : int = Number of the elements of the vectors in input.

        # sigma : float, optional (default=1.0)
        #     Spread of the neighborhood function, needs to be adequate
        #     to the dimensions of the map.

        # decay_function : function (default=None)
        #     Function that reduces learning_rate and sigma at each iteration
        #     the default function is:
        #                 learning_rate / (1+t/(max_iterarations/2))
     
        # neighborhood_function : function =  Function that weights the neighborhood of a position in the map
    
        # random_seed : int, optional (default=None) = Random seed to use.
        
        if sigma >= x or sigma >= y:
            warn('Warning: sigma is higher than map dimensions!')

        self.random_generator = random.RandomState(random_seed)

        self.learning_rate = learning_rate
        self.sigma = sigma
        self.input_len = input_len

        # random initialization
        self.weights = self.random_generator.rand(x, y, input_len)*2-1

        for i in range(x):
            for j in range(y):
                # normalization
                norm = fast_norm(self.weights[i, j])
                self.weights[i, j] = self.weights[i, j] / norm

        self.neigx = arange(x)
        self.neigy = arange(y) 
        self.decay_function = asymptotic_decay
        self.neighborhood = self.bubble
        self.activation_map = zeros((x, y))

    def get_weights(self):
        return self.weights

    def _activate(self, x):
        """Updates matrix activation_map, in this matrix
           the element i,j is the response of the neuron i,j to x."""
        s = subtract(x, self.weights)  # x - w
        it = nditer(self.activation_map, flags=['multi_index'])
        while not it.finished:
            # || x - w ||
            self.activation_map[it.multi_index] = fast_norm(s[it.multi_index])
            it.iternext()

    def bubble(self, c, sigma):
        # Spread sigma with center in c
        ax = logical_and(self.neigx > c[0]-sigma,
                         self.neigx < c[0]+sigma)
        ay = logical_and(self.neigy > c[1]-sigma,
                         self.neigy < c[1]+sigma)
        return outer(ax, ay)*1.


    def check_iteration_number(self, num_iteration):
        if num_iteration < 1:
            raise ValueError('num_iteration must be > 1')

    def check_input_len(self, data):
        # Checks that the data in input is of the correct shape
        data_len = len(data[0])
        if self.input_len != data_len:
            msg = 'Received %d features, expected %d.' % (data_len,
                                                          self.input_len)
            raise ValueError(msg)

    def winner(self, x):
        # Coordinates for the winning neuron of a sample x
        self._activate(x)
        return unravel_index(self.activation_map.argmin(),
                             self.activation_map.shape)
    
    def get_winners(self,data):
        # Coordinates for winning neurons of many samples
        return [self.winner(x) for x in data]

    def get_winner_dict(self,data,labels):
        # Returns a dictionary matching labels from sample data to its coordinates

        winners = self.get_winners(data)
        dict = { list: [] for list in winners }
        for cnt, n in enumerate(winners):
            dict[(n[0],n[1])].append(labels[cnt])
        return dict

    def update(self, x, win, t, max_iteration):
        # Updates the weights of neurons

        eta = self.decay_function(self.learning_rate, t, max_iteration)
        # sigma and learning rate decrease with the same rule
        sig = self.decay_function(self.sigma, t, max_iteration)
        # improves the performances
        g = self.neighborhood(win, sig)*eta
        it = nditer(g, flags=['multi_index'])

        while not it.finished:
            # eta * neighborhood_function * (x-w)
            x_w = (x - self.weights[it.multi_index])
            self.weights[it.multi_index] += g[it.multi_index] * x_w
            it.iternext()

    def random_weights_init(self, data):
        # Initialize weights with random data from sample
        self.check_input_len(data)
        it = nditer(self.activation_map, flags=['multi_index'])
        while not it.finished:
            rand_i = self.random_generator.randint(len(data))
            self.weights[it.multi_index] = data[rand_i]
            norm = fast_norm(self.weights[it.multi_index])
            self.weights[it.multi_index] = self.weights[it.multi_index]
            it.iternext()
    
    def pca_weights_init(self, data):
        """Initializes the weights to span the first two principal components.
        This initialization doesn't depend on random processes and
        makes the training process converge faster.
        It is strongly reccomended to normalize the data before initializing
        the weights and use the same normalization for the training data.
        """
        if self.input_len == 1:
            msg = 'The data needs at least 2 features for pca initialization'
            raise ValueError(msg)
        self.check_input_len(data)
        if len(self.neigx) == 1 or len(self.neigy) == 1:
            msg = 'PCA initialization inappropriate:' + \
                  'One of the dimensions of the map is 1.'
            warn(msg)
        pc_length, pc = linalg.eig(cov(transpose(data)))
        pc_order = argsort(pc_length)
        for i, c1 in enumerate(linspace(-1, 1, len(self.neigx))):
            for j, c2 in enumerate(linspace(-1, 1, len(self.neigy))):
                self.weights[i, j] = c1*pc[pc_order[0]] + c2*pc[pc_order[1]]

    def train_random(self, data, num_iteration):
        # Trains the SOM picking samples at random from data.
        print("Training...")
        self.check_iteration_number(num_iteration)
        self.check_input_len(data)
        iterations = range(num_iteration)

        # Verbose output
        iterations = _incremental_index_verbose(num_iteration)

        for iteration in iterations:
            # pick a random sample
            rand_i = self.random_generator.randint(len(data))
            self.update(data[rand_i], self.winner(data[rand_i]),
                        iteration, num_iteration)
        print("\n...ready!")

    def train_batch(self, data, num_iteration):
    #    Trains using all the vectors in data sequentially.
        print("Training...")
        self.check_iteration_number(num_iteration)
        self.check_input_len(data)
        iterations = range(num_iteration)
        iterations = _incremental_index_verbose(num_iteration)

        for iteration in iterations:
            idx = iteration % (len(data)-1)
            self.update(data[idx], self.winner(data[idx]),
                        iteration, num_iteration)
        print("\n...ready!")


    def distance_map(self):
        # Distance map of weights. Each cell contains the sum of distances to neighbors

        um = zeros((self.weights.shape[0], self.weights.shape[1]))
        it = nditer(um, flags=['multi_index'])
        while not it.finished:
            for ii in range(it.multi_index[0]-1, it.multi_index[0]+2):
                for jj in range(it.multi_index[1]-1, it.multi_index[1]+2):
                    if (ii >= 0 and ii < self.weights.shape[0] and
                            jj >= 0 and jj < self.weights.shape[1]):
                        w_1 = self.weights[ii, jj, :]
                        w_2 = self.weights[it.multi_index]
                        um[it.multi_index] += fast_norm(w_1-w_2)
            it.iternext()
        um = um/um.max()
        return um

