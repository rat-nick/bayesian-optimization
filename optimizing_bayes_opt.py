from optimization_problem import OptimizationProblem
from bayes_optimizer import BayesOptimizer
from search_space import SearchSpace, Parameter
import random
from sklearn.gaussian_process import GaussianProcessRegressor
from benchmark import branin
from numpy import average
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

branin_optimizer = BayesOptimizer(
    branin_problem, GaussianProcessRegressor(), max_iterations=1000, epsilon=0.01
)


def objective(array, *args, **kwargs):
    return average([branin_optimizer.optimize(start_sample_size=array[0], sample_size=array[1]) for i in range(20)]) * -1


bayes_optimization_problem = OptimizationProblem(
    SearchSpace(
        [
            # Parameter("target", random.choice, [branin_optimizer]),
            Parameter("", random.choice, [x for x in range(1, 200)]),
            Parameter("", random.choice, [x * 20 for x in range(1, 50)]),
            # Parameter("population", random.choice, [30]),
        ]
    ),
    function=objective,
)

bayes_optimizer = BayesOptimizer(
    bayes_optimization_problem, GaussianProcessRegressor(), max_iterations=100
)

print(
    bayes_optimizer.optimize(
        start_sample_size=1, sample_size=1000, hyperopt=True, point=True, debug=True
    )
)
