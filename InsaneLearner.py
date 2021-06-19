import numpy as np
import BagLearner as bl
import LinRegLearner as lrl
class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.verbose=verbose
    def author(self):
        return "kkang68"
    def add_evidence(self, data_x, data_y):
        self.learners = [bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20, boost=False, verbose=False) for i in range(20)]
        for learner in self.learners:
            learner.add_evidence(data_x, data_y)
    def query(self, points):
        return np.mean([learner.query(points) for learner in self.learners], axis=0)
if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")  		  	   		   	 			  		 			 	 	 		 		 	
