from math import sin
from math import pi
import numpy as np
from numpy import arange
from numpy import vstack
from numpy import argmax
from numpy import asarray
from numpy.random import normal
from random import uniform
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
from matplotlib import pyplot
from benchmark import benchmark_func
from scipy.stats import kde
import seaborn as sns
extent_min = -5.12
extent_max = 5.12
 
def sample(extent_min, extent_max, num_samples):
	return asarray([uniform(extent_min, extent_max) for i in range(1, num_samples)])
 
# objective function
# def objective(x, noise=0.1):
# 	noise = normal(loc=0, scale=noise)
# 	return (x**2 * sin(5 * pi * x)**6.0) + noise

 
# surrogate or approximation for the objective function
def surrogate(model, X):
	# catch any warning generated when making a prediction
	with catch_warnings():
		# ignore generated warnings
		simplefilter("ignore")
		return model.predict(X, return_std=True)
 
# probability of improvement acquisition function
def acquisition(X, Xsamples, model):
	# calculate the best surrogate score found so far
	yhat, _ = surrogate(model, X)
	best = max(yhat)
	# calculate mean and stdev via surrogate function
	mu, std = surrogate(model, Xsamples)
	mu = mu[:, 0]
	# calculate the probability of improvement
	probs = norm.cdf((mu - best) / (std+1E-9))
	return probs
 
# optimize the acquisition function
def opt_acquisition(X, y, model):
	# random search, generate random samples
	Xsamples = sample(extent_min, extent_max, 100)
	Xsamples = Xsamples.reshape(len(Xsamples), 1)
	# calculate the acquisition function for each sample
	scores = acquisition(X, Xsamples, model)
	# locate the index of the largest scores
	ix = argmax(scores)
	return Xsamples[ix, 0]
 
# plot real observations vs surrogate function
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
 
# sample the domain sparsely with noise
X = sample(extent_min, extent_max, 100)
y = asarray([benchmark_func([x]) for x in X])
# reshape into rows and cols
X = X.reshape(len(X), 1)
y = y.reshape(len(y), 1)
# define the model
model = GaussianProcessRegressor()
# fit the model
model.fit(X, y)
# plot before hand
plot(X, y, model)
resi = []
resx = []
resy = []
resf = []
resm = []
maxx = -1000000
# perform the optimization process
for i in range(100):
	# select the next point to sample
	x = opt_acquisition(X, y, model)
	# sample the point
	actual = benchmark_func([x])
	# summarize the finding
	est, _ = surrogate(model, [[x]])
	print('>x=%.3f, f()=%3f, actual=%.3f' % (x, est, actual))
	if actual > maxx :
		maxx = actual
	resx.append(x)
	resi.append(i)
	resm.append(maxx)
	resy.append(est)
	resf.append(actual)
 	# add the data to the dataset
	X = vstack((X, [[x]]))
	y = vstack((y, [[actual]]))
	# update the model
	model.fit(X, y)
 
# plot all samples and the final surrogate function
plot(X, y, model)
# best result
ix = argmax(y)

#pyplot.scatter(resi, resx)
#pyplot.plot(resi, resy)
resy = asarray(resy)
resf = asarray(resf)
resm = asarray(resm)
print(resm)
resy = resy.reshape(len(resy), 1)
resf = resf.reshape(len(resf), 1)
resm = resm.reshape(len(resm), 1)

#pyplot.plot(resi, np.array(resf) * -1, "tab:blue")
pyplot.plot(resi, np.array(resy) * -1, "tab:orange", label="surrogate model prediction")
pyplot.plot(resi, np.array(resm) * -1, "tab:green", label="current best minimum")
pyplot.legend()
pyplot.show()
print('Best Result: x=%.3f, y=%.3f' % (X[ix], y[ix]))