from benchmark import six_bipolar, trap5, three_deceptive, one_max
from bayes_optimizer import BayesOptimizer
from search_space import SearchSpace, Axis
from random import choice
from sklearn.gaussian_process import GaussianProcessRegressor

problem_size = 15

optimizer = BayesOptimizer(
    SearchSpace([Axis(choice, [0, 1]) for i in range(problem_size)]),
    GaussianProcessRegressor(),
    100,
    one_max,
    optimal_value=problem_size
)

optimizer.optimize()
