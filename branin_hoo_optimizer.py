from optimization_problem import OptimizationProblem
from benchmark import branin
from bayes_optimizer import BayesOptimizer
from search_space import SearchSpace, Parameter
import random
from sklearn.gaussian_process import GaussianProcessRegressor

branin_problem = OptimizationProblem(
    SearchSpace(
        [
            Parameter("", random.uniform, a=-5, b=10),
            Parameter("", random.uniform, a=0, b=15),
        ]
    ),
    function=branin,
    optimal_value=-0.397887,
)

sum = 0

optimizer = BayesOptimizer(
    branin_problem,
    GaussianProcessRegressor(),
    epsilon=0.0001,
    max_iterations=3000
)
for i in range(1):
    sum += optimizer.optimize(
        debug=True,
        start_sample_size=200,
        sample_size=1000,
    )

print("\n\n\n\nAverage iterations is ", sum)
