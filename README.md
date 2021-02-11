# bayesian-optimization
School project for the subject of heurstic methods of optimization. Implementation of bayesian optimization algorithm (BOA) using Gausian process regressors as surogate models.

## Usage

```shell
  $ python3 bayes-optimizer.py
```

### Changing benchmark functions
In the file `benchmark.py` change the function that is called in the benchmark function.

### Changing the search space
In the file `bayes-optimizer.py` change the extent_min and extent_max variables.

## Reading the graphs
This graph shows the distribution of the sampled points in the search space **prior** to any evidence.
![alt text](https://github.com/ratinac-nikola/bayesian-optimization/blob/main/graphs/rastrigin/prior.png?raw=true)

This graph shows the distribution of the sampled points in the search space **posterior** to gathered evidence via search space sampling.
![alt text](https://github.com/ratinac-nikola/bayesian-optimization/blob/main/graphs/rastrigin/posterior.png?raw=true)

This graph shows the performance of the optimization on a given "*black box*" function.
![alt text](https://github.com/ratinac-nikola/bayesian-optimization/blob/main/graphs/rastrigin/results.png?raw=true)
