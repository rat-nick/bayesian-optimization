from math import nan, isnan
from numpy import vstack, hstack
from numpy import argmax
from numpy import asarray
from scipy.stats import norm
from warnings import catch_warnings
from warnings import simplefilter

desired_value = 1


class BayesOptimizer:
    def __init__(
        self,
        op,
        model,
        max_iterations=1000,
        epsilon=0.01,
    ):
        self.op = op
        self.model = model
        
        self.epsilon = epsilon
        
        self.max_iterations = max_iterations

        self.__best_found = nan
        self.__i = 0

    def optimize(self, debug=False, *args, **kwargs):
        """Method for opitmizing the objective function"""
        # sample the search space
        X = self.op.search_space.sample_multiple(kwargs["start_sample_size"])
        # print(X)
        y = asarray([self.op.function(x) for x in X])
        # print(y)
        self.__i = 0
        while self.__i < self.max_iterations:
            # select next point to sample
            x = self.__opt_acquisition(X, kwargs['sample_size'], y)
            # evaluate the point with the fuction
            actual = self.op.function(x)
            if isnan(self.__best_found) or self.__best_found < actual:
                self.__best_found = actual
            # print("next sample is with cost", x, actual)
            # summarize the finding
            est, _ = self.__surrogate([x])
            if debug:
                print(x + [actual])
                #print("Surrogate estimate", est)
                #print("Actual value ", actual)
                
            if (not isnan(self.op.optimal_value)) and abs(
                actual - self.op.optimal_value
            ) < self.epsilon:
                #p rint("Converged at ", self.__i, " iteration")
                return self.__i
            # add the data to the dataset
            X = vstack((X, [x]))
            y = hstack((y, actual))
            # update the model
            self.model.fit(X, y)
            self.__i += 1

        print("Finished at ", self.__i, " iterations")
        print("Best result is ", self.__best_found)
        print("Global optima is ", self.op.optimal_value)
        return self.__i

    def __acquisition(self, X, Xsamples):
        # calculate the best surrogate score found so far
        yhat, _ = self.__surrogate(X)
        best = max(yhat)
        # calculate mean and stdev via surrogate function
        mu, std = self.__surrogate(Xsamples)
        # mu = mu[:, 0]
        # calculate the probability of improvement
        probs = norm.cdf((mu - best) / (std + 1e-9))
        return probs

    def __opt_acquisition(self, X, num_samples, y):
        Xsamples = self.op.search_space.sample_multiple(num_samples)
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

    @staticmethod
    def test_hyperparameters(op_problem, max_iterations, hp_array, avg_of=30):
        start_smp_sz = hp_array[0]
        smp_sz = hp_array[1]

        target = BayesOptimizer(
            op=op_problem,
            max_iterations=max_iterations,
            starting_sample_size=start_smp_sz,
            sample_size=smp_sz,
        )

        return target.optimize()
