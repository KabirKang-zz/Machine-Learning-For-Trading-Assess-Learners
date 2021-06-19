import numpy as np
from random import randrange


class RTLearner(object):
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None

    def author(self):
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
        self.tree = self.build_tree(data_x, data_y)

    def query(self, points):
        result = np.empty([points.shape[0]])
        for i in range(0, points.shape[0]):
            curr = 0
            x_row = points[i]
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
