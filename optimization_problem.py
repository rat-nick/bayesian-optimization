from math import nan


class OptimizationProblem:

    def __init__(self, search_space, function, optimal_value=nan):
        self.search_space = search_space
        self.function = function
        self.optimal_value = optimal_value

    def evaluate(self):
        return self.function(self.search_space.sample())
