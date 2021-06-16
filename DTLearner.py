""""""  		  	   		   	 			  		 			 	 	 		 		 	
"""  		  	   		   	 			  		 			 	 	 		 		 	
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		   	 			  		 			 	 	 		 		 	
                                                                                    
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 			  		 			 	 	 		 		 	
Atlanta, Georgia 30332  		  	   		   	 			  		 			 	 	 		 		 	
All Rights Reserved  		  	   		   	 			  		 			 	 	 		 		 	
                                                                                    
Template code for CS 4646/7646  		  	   		   	 			  		 			 	 	 		 		 	
                                                                                    
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 			  		 			 	 	 		 		 	
works, including solutions to the projects assigned in this course. Students  		  	   		   	 			  		 			 	 	 		 		 	
and other users of this template code are advised not to share it with others  		  	   		   	 			  		 			 	 	 		 		 	
or to make it available on publicly viewable websites including repositories  		  	   		   	 			  		 			 	 	 		 		 	
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 			  		 			 	 	 		 		 	
or edited.  		  	   		   	 			  		 			 	 	 		 		 	
                                                                                    
We do grant permission to share solutions privately with non-students such  		  	   		   	 			  		 			 	 	 		 		 	
as potential employers. However, sharing with other current or future  		  	   		   	 			  		 			 	 	 		 		 	
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 			  		 			 	 	 		 		 	
GT honor code violation.  		  	   		   	 			  		 			 	 	 		 		 	
                                                                                    
-----do not edit anything above this line---  		  	   		   	 			  		 			 	 	 		 		 	
"""  		  	   		   	 			  		 			 	 	 		 		 	

import numpy as np  		  	   		   	 			  		 			 	 	 		 		 	


class DTLearner(object):
    """  		  	   		   	 			  		 			 	 	 		 		 	
    This is a Linear Regression Learner. It is implemented correctly.  		  	   		   	 			  		 			 	 	 		 		 	

    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		   	 			  		 			 	 	 		 		 	
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		   	 			  		 			 	 	 		 		 	
    :type verbose: bool  		  	   		   	 			  		 			 	 	 		 		 	
    """  		  	   		   	 			  		 			 	 	 		 		 	
    def __init__(self, leaf_size=1, verbose=False):
        """  		  	   		   	 			  		 			 	 	 		 		 	
        Constructor method  		  	   		   	 			  		 			 	 	 		 		 	
        """  		  	   		   	 			  		 			 	 	 		 		 	
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None

    def author(self):  		  	   		   	 			  		 			 	 	 		 		 	
        """  		  	   		   	 			  		 			 	 	 		 		 	
        :return: The GT username of the student  		  	   		   	 			  		 			 	 	 		 		 	
        :rtype: str  		  	   		   	 			  		 			 	 	 		 		 	
        """  		  	   		   	 			  		 			 	 	 		 		 	
        return "kkang68"

    def get_split_feature(self, data_x, data_y):
        print('get split')
        num_features = data_x.shape[1]
        max_corr = 0
        split_index = 0
        for i in range(0, num_features):
            feature = data_x[:, i]
            print(f'i {i}')
            print(f'x {feature.shape}')
            print(f'y {data_y[:].shape}')
            corr = abs(np.corrcoef(feature, data_y[:], rowvar=False)[0, 1])
            print(f'corr {corr}')
            if corr > max_corr:
                split_index = i
                max_corr = corr
        return split_index

    def build_tree(self, data_x, data_y):
        # if data.shape[0]: return [leaf, data.y, NA, NA]
        # if all data.y same: return [leaf, data.y, NA, NA]
        if data_x.shape[0] <= self.leaf_size or len(np.unique(data_y)) == 1:
            return np.asarray([np.nan, data_y, np.nan, np.nan])
        else:
            # determine best feature i to split on (feature with highest correlation with y)
            feature_i = self.get_split_feature(data_x, data_y)
            split_val = np.median(data_x[:, feature_i])
            left_tree = self.build_tree(data_x[data_x[:, feature_i] <= split_val], data_y[data_x[:, feature_i] <= split_val])
            right_tree = self.build_tree(data_x[data_x[:, feature_i] > split_val], data_y[data_x[:, feature_i] > split_val])
            root = np.asarray([feature_i, split_val, 1, left_tree.shape[0]+1])
            return np.concatenate((root, left_tree, right_tree))

    def add_evidence(self, data_x, data_y):  		  	   		   	 			  		 			 	 	 		 		 	
        """  		  	   		   	 			  		 			 	 	 		 		 	
        Add training data to learner  		  	   		   	 			  		 			 	 	 		 		 	
                                                                                    
        :param data_x: A set of feature values used to train the learner  		  	   		   	 			  		 			 	 	 		 		 	
        :type data_x: numpy.ndarray  		  	   		   	 			  		 			 	 	 		 		 	
        :param data_y: The value we are attempting to predict given the X data  		  	   		   	 			  		 			 	 	 		 		 	
        :type data_y: numpy.ndarray  		  	   		   	 			  		 			 	 	 		 		 	
        """
        self.tree = self.build_tree(data_x, data_y)
        print(self.tree)
        # slap on 1s column so linear regression finds a constant term
        # new_data_x = np.ones([data_x.shape[0], data_x.shape[1] + 1])
        # new_data_x[:, 0 : data_x.shape[1]] = data_x
        #
        # build and save the model
        # self.model_coefs, residuals, rank, s = np.linalg.lstsq(
        #     new_data_x, data_y, rcond=None
        # )


    def query(self, points):  		  	   		   	 			  		 			 	 	 		 		 	
        """  		  	   		   	 			  		 			 	 	 		 		 	
        Estimate a set of test points given the model we built.  		  	   		   	 			  		 			 	 	 		 		 	
                                                                                    
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		   	 			  		 			 	 	 		 		 	
        :type points: numpy.ndarray  		  	   		   	 			  		 			 	 	 		 		 	
        :return: The predicted result of the input data according to the trained model  		  	   		   	 			  		 			 	 	 		 		 	
        :rtype: numpy.ndarray  		  	   		   	 			  		 			 	 	 		 		 	
        """  		  	   		   	 			  		 			 	 	 		 		 	
        return (self.model_coefs[:-1] * points).sum(axis=1) + self.model_coefs[  		  	   		   	 			  		 			 	 	 		 		 	
            -1  		  	   		   	 			  		 			 	 	 		 		 	
        ]  		  	   		   	 			  		 			 	 	 		 		 	


if __name__ == "__main__":  		  	   		   	 			  		 			 	 	 		 		 	
    print("the secret clue is 'zzyzx'")  		  	   		   	 			  		 			 	 	 		 		 	
