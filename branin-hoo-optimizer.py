from optimization_problem import OptimizationProblem
from benchmark import branin
from bayes_optimizer import BayesOptimizer
from search_space import SearchSpace, Axis
import random
from sklearn.gaussian_process import GaussianProcessRegressor

branin_problem = OptimizationProblem(
    SearchSpace([Axis(random.uniform, -5, 10), Axis(random.uniform, 0, 15)]),
    function=branin,
    optimal_value=-0.397887,
)
sum = 0
for i in range(20):
    optimizer = BayesOptimizer(
        branin_problem,
        GaussianProcessRegressor(),
        starting_sample_size=500,
        sample_size=500,
        epsilon=0.001,
        max_iterations=2000,
    )
    sum += optimizer.optimize(debug=False)

print("\n\n\n\nAverage iterations is ", sum / 10)
