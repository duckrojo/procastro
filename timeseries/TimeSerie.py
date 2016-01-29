__author__ = 'fran'

import scipy as sp
import warnings

class TimeSerie(object):

    def __init__(self, data, errors, ids):
        self.channels = data  # [[target1_im1, target1_im2, ...], [target2_im1, target2_im2, ...]]
        self.errors = errors
        # [[error_target1_im1, error_target1_im2, ...], [error_target2_im1, error_target2_im2, ...]]
        self.group = [1] + [0 for i in range(len(data)-1)]
        # Default grouping: 1st coordinate is 1 group, all other objects are another group
        self.ids = ids  # Dictionary for names?

        self.channels.append([])  # Group 1 operation result; is overwritten every time a new op is defined
        self.errors.append([])

        self.channels.append([])  # Group 2 operation result; is overwritten every time a new op is defined
        self.errors.append([])

    def __getitem__(self, item):
        # This is so I can do ts[0]/ts[2] and it works directly with the channels!
        try:
            return self.channels[item]
        except TypeError:
            return self.channels[self.ids.index(item)]

    def group1(self):
        return [self.channels[i] for i in range(len(self.channels) - 2) if self.group[i]]

    def group2(self):
        return [self.channels[i] for i in range(len(self.channels) - 2) if not self.group[i]]

    def set_group(self, new_group):  # Receives pe [0 1 0 0 1 0] and that is used to define 2 groups
        self.group = new_group

    def errors_group1(self):
        return [self.errors[i] for i in range(len(self.errors) - 2) if self.group[i]]

    def errors_group2(self):
        return [self.errors[i] for i in range(len(self.errors) - 2) if not self.group[i]]

    def mean(self, group_id):
        if group_id > 2:
            warnings.warn("group_id must be 1 or 2 only. Group 2 will be used as default.")
            group = self.group2()
            g_errors = self.errors_group2()
        elif group_id == 1:
            group = self.group1()
            g_errors = self.errors_group1()
        else:
            group = self.group2()
            g_errors = self.errors_group2()

        self.channels[-group_id] = sp.mean(group, axis=0)
        err = sp.zeros((1, len(g_errors[0])))
        for i in range(len(g_errors)):
            err += sp.divide(g_errors[i]/group[i])**2
        self.errors[-group_id] = sp.sqrt(err)

        return self.channels[-group_id]

    def median(self, group_id):
        if group_id > 2:
            warnings.warn("group_id must be 1 or 2 only. Group 2 will be used as default.")
            group = self.group2()
            g_errors = self.errors_group2()
        elif group_id == 1:
            group = self.group1()
            g_errors = self.errors_group1()
        else:
            group = self.group2()
            g_errors = self.errors_group2()

        self.channels[-group_id] = sp.median(group, axis=0)
        err = sp.zeros((1, len(g_errors[0])))
        for i in range(len(g_errors)):
            err += sp.divide(g_errors[i]/group[i])**2
        self.errors[-group_id] = sp.sqrt(err)

        return self.channels[-group_id]
