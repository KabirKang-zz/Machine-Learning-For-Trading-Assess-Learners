import numpy as np


class BagLearner(object):
    def __init__(self, learner, kwargs, bags, boost, verbose):
        self.verbose = verbose
        self.learners = []
        self.num_bags = bags

        for i in range(bags):
            self.learners.append(learner(**kwargs))

    def author(self):
        return "kkang68"

    def add_evidence(self, data_x, data_y):
        for learner in self.learners:
            bag = np.random.choice(range(data_x.shape[0]), data_x.shape[0], replace=True)
            learner.add_evidence(data_x[bag], data_y[bag])

    def query(self, points):
        query_results = []
        for learner in self.learners:
            query_results.append(learner.query(points))
        return np.mean(query_results, axis=0)


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
