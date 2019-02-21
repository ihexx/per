import random
import numpy as np
from SumTree import SumTree
import warnings


class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_anneal_step=0.001, epsilon=0.01):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.a = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_anneal_step
        self.e = epsilon

    def _get_priority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []
        uninitialized_samples = 0

        self.beta = np.min([1. - self.e, self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)

            if data == 0:  # Pulled an invalid sample (uninitialized)
                uninitialized_samples += 1
                continue

            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        if uninitialized_samples > 0:
            warnings.warn('Pulled {} uninitialized samples'.format(uninitialized_samples))

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
