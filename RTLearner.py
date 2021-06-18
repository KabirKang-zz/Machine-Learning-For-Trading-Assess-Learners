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
from random import randrange

class RTLearner(object):
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
        return randrange(data_x.shape[1])

    def build_tree(self, data_x, data_y):
        if data_x.shape[0] <= self.leaf_size:
            return np.array([[np.nan, np.median(data_y), np.nan, np.nan]], dtype=object)
        elif len(np.unique(data_y)) == 1:
            return np.array([[np.nan, data_y[0], np.nan, np.nan]], dtype=object)
        else:
            feature_i = self.get_split_feature(data_x, data_y)
            split_val = np.median(data_x[:, feature_i])
            left = data_x[:, feature_i] <= split_val
            if np.all(np.isclose(left, left[0])):
                return np.array([[np.nan, np.mean(data_y), np.nan, np.nan]], dtype=object)
            left_tree = self.build_tree(data_x[data_x[:, feature_i] <= split_val],
                                        data_y[data_x[:, feature_i] <= split_val])
            right_tree = self.build_tree(data_x[data_x[:, feature_i] > split_val],
                                         data_y[data_x[:, feature_i] > split_val])
            root = np.array([[feature_i, split_val, 1, left_tree.shape[0] + 1]], dtype=object)
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

    def query(self, data_x):
        """
            Estimate a set of test points given the model we built.

            :param data_x: A numpy array with each row corresponding to a specific query.
            :type data_x: numpy.ndarray
            :return: The predicted result of the input data according to the trained model
            :rtype: numpy.ndarray
            """
        result = np.empty([data_x.shape[0]])
        for i in range(0, data_x.shape[0]):
            curr = 0
            x_row = data_x[i]
            while not np.isnan(self.tree[curr][0]):
                curr_node = self.tree[curr]
                feature = x_row[int(curr_node[0])]
                split_val = curr_node[1]
                if feature <= split_val:
                    curr = curr + int(curr_node[2])
                else:
                    curr = curr + int(curr_node[3])
            result[i] = self.tree[curr][1]
        return result

if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
