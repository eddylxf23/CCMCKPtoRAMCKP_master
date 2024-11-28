

import numpy as np

class Item:

    def __init__(self, item_id, cost, value, mean, variance):
        self.id = item_id  # identical id
        self.cost = cost
        self.value = value
        self.mean = mean
        self.variance = variance

        self.weight = None
        self.utility = None
        self.increment = None


    def set_weight(self,weight):
        self.weight = weight

    def set_utility(self,utility):
        self.utility = utility

    def set_increment(self,increment):
        self.increment = increment
