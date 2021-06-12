class SearchSpace:
    """
    The search space class containing the domain definitions
    used for sampling points using a random distribution
    """

    def __init__(self, axes):
        self.axes = axes

    def sample(self):
        d = {}
        p = []
        for a in self.axes:
            if(a.parameter_name == ""):
                p.append(a.sample())
            else:
                d[a.parameter_name] = a.sample()            
        if p != [] and d != {}:
            return p, d
        if d != {}:
            return d
        if p != []:
            return p

    def sample_multiple(self, num):
        return [self.sample() for i in range(num)]


class Parameter:
    """
    Class to represent an axis of the search space
    be it continous or discrete
    """

    def __init__(self, parameter_name, distribution, *args, **kwargs):
        self.distribution = distribution
        self.parameter_name = parameter_name
        self.pos_constraints = args
        self.kw_constraints = kwargs
        # print(kwargs)

    def sample(self):
        return self.distribution(
            *(self.pos_constraints),
            **(self.kw_constraints),
        )
