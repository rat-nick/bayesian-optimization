from math import sin
from math import pi
import numpy as np
from numpy import arange
from numpy import vstack, hstack
from numpy import argmax
from numpy import asarray
from numpy.random import normal
from random import uniform, choice
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
from matplotlib import pyplot
from benchmark import benchmark_func
from scipy.stats import kde
import seaborn as sns
from search_space import SearchSpace, Axis

desired_value = 1


class BayesOptimizer:

    def __init__(self, search_space, model, sample_size, function, optimal_value, epsilon=0.01):
        self.search_space = search_space
        self.model = model
        self.sample_size = sample_size
        self.function = function
        self.optimal_value = optimal_value
        self.epsilon = epsilon

    def optimize(self):
        """Method for opitmizing the objective function"""
        # sample the search space
        X = self.search_space.sample_multiple(self.sample_size)
        # print(X)
        y = asarray([self.function(x) for x in X])
        # print(y)
        i = 0
        while(True):
            # select next point to sample
            x = self.__opt_acquisition(X, y)
            # evaluate the point with the fuction
            actual = self.function(x)
            #print("next sample is with cost", x, actual)
            # summarize the finding
            est, _ = self.__surrogate([x])
            print("Surrogate estimate", est)
            print("Actual value ", actual)
            if abs(actual - self.optimal_value) < self.epsilon:
                print("Converged at ", i, " iteration")
                break
            # add the data to the dataset
            X = vstack((X, [x]))
            y = hstack((y, actual))
            # update the model
            self.model.fit(X, y)
            self.optimal_point = argmax(y)
            i += 1

    def __acquisition(self, X, Xsamples):
        # calculate the best surrogate score found so far
        yhat, _ = self.__surrogate(X)
        best = max(yhat)
        # calculate mean and stdev via surrogate function
        mu, std = self.__surrogate(Xsamples)
        #mu = mu[:, 0]
        # calculate the probability of improvement
        probs = norm.cdf((mu - best) / (std+1E-9))
        return probs

    def __opt_acquisition(self, X, y):
        Xsamples = self.search_space.sample_multiple(100)
        # calculate the acquisition function for each sample
        scores = self.__acquisition(X, Xsamples)
        # locate the index of the largest scores
        ix = argmax(scores)
        return Xsamples[ix]

    def __surrogate(self, X):
        # catch any warning generated when making a prediction
        with catch_warnings():
            simplefilter("ignore")
            return self.model.predict(X, return_std=True)


def plot(X, y, model):
    # scatter plot of inputs and real objective function
    sns.kdeplot(data=X, cut=0, bw_adjust=.1, label='distribution')
    pyplot.legend()
    pyplot.show()

    pyplot.scatter(X, y, label="samples")
    pyplot.legend()
    # line plot of surrogate function across domain
    Xsamples = asarray(arange(extent_min, extent_max, 0.1))

    Xsamples = Xsamples.reshape(len(Xsamples), 1)
    ysamples, _ = surrogate(model, Xsamples)
    pyplot.plot(Xsamples, ysamples, "tab:orange", label="surrogate model")
    pyplot.legend()
    # density = kde.gaussian_kde(X)
    # xs = np.linspace(extent_min, extent_max, 200)
    pyplot.show()


# # reshape into rows and cols
# #X = X.reshape(len(X), 1)
# y = y.reshape(len(y), 1)
# # define the model
# model = GaussianProcessRegressor()
# # fit the model
# model.fit(X, y)
# # plot before hand
# #plot(X, y, model)
# resi = []
# resx = []
# resy = []
# resf = []
# resm = []
# maxx = -1000000
# # perform the optimization process


# # plot all samples and the final surrogate function
# #plot(X, y, model)
# # best result


# #pyplot.scatter(resi, resx)
# #pyplot.plot(resi, resy)
# resy = asarray(resy)
# resf = asarray(resf)
# resm = asarray(resm)
# print(resm)
# resy = resy.reshape(len(resy), 1)
# resf = resf.reshape(len(resf), 1)
# resm = resm.reshape(len(resm), 1)

# #pyplot.plot(resi, np.array(resf) * -1, "tab:blue")
# #pyplot.plot(resi, np.array(resy) * -1, "tab:orange", label="surrogate model prediction")
# #pyplot.plot(resi, np.array(resm) * -1, "tab:green", label="current best minimum")
# # pyplot.legend()
# # pyplot.show()
# #print('Best Result: x=%.3f, y=%.3f' % (X[ix], y[ix]))
