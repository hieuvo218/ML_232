import bisect
import collections
import collections.abc
import functools
import heapq
import operator
import os.path
import random
from itertools import chain, combinations
from statistics import mean

import numpy as np

def manhattan_distance(X1, X2):
	return sum(abs(i-j) for (i,j) in zip(X1, X2))

def euclidean_distance(X1, X2):
	return np.sqrt(sum((i-j)**2 for (i,j) in zip(X1, X2)))

def unique(seq):
    """Remove duplicate elements from seq. Assumes hashable elements."""
    return list(set(seq))

def product(numbers):
    """Return the product of the numbers, e.g. product([2, 3, 10]) == 60"""
    result = 1
    for x in numbers:
        result *= x
    return result