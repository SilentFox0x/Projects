import numpy as np
from typing import List
from collections import Counter


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html

def entropy(words: List[str]):
    counter = Counter(words)
    num = len(words)
    distribution = []
    for _, value in counter.items():
        distribution.append(value / num)
    pk = np.array(distribution)
    H = -np.sum(pk * np.log(pk))
    return H