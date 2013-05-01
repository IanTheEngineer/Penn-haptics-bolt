import numpy as np
from sklearn.utils.fixes import unique
import random

def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(xrange(n), r))
    return tuple(pool[i] for i in indices)

class SafeLeavePLabelOut(object):
    def __init__(self, labels, p, max_iters, train_Y, max_cache_hits = 100,
                 indices=True):
        self.labels = labels
        self.unique_labels = unique(self.labels)
        self.n_unique_labels = self.unique_labels.size
        self.p = p
        self.indices = indices
        self.train_Y = train_Y
        self.max_iters = max_iters
        self.max_cache_hits = max_cache_hits
   
    def __iter__(self):
        
        # We make a copy here to avoid side-effects during iteration
        labels = np.array(self.labels, copy=True)
        unique_labels = unique(labels)
        curr_iter = 0
        idx_cache = set()
        num_cache_hits = 0
        max_cache_hits = self.max_cache_hits
        
        #comb = combinations(range(self.n_unique_labels), self.p)
    
        while curr_iter < self.max_iters and num_cache_hits < max_cache_hits:
            
            idx = random_combination(range(self.n_unique_labels),
                                              self.p)
            if idx in idx_cache:
                num_cache_hits += 1
                if num_cache_hits >= max_cache_hits:
                    print "WARNING LeavePLabelOut: number of consecutive cache hits too high, bailing out after %d samples" % curr_iter
                continue
            else:
                num_cache_hits = 0
            idx_cache.add(idx)
            
            idx = np.array(idx)
            
            test_index = np.zeros(labels.size, dtype=np.bool)
            idx = np.array(idx)
            for l in unique_labels[idx]:
                test_index[labels == l] = True
            train_index = np.logical_not(test_index)
            if self.indices:
                ind = np.arange(labels.size)
                train_index = ind[train_index]
                test_index = ind[test_index]
            
            if len(unique(self.train_Y[train_index])) == 1:
                #prevent test sets with only one class
                continue
            
            curr_iter += 1
            yield train_index, test_index
    
    def __repr__(self):
        return '%s.%s(labels=%s, p=%s)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.labels,
            self.p,
        )
    
    def __len__(self):
        return self.max_iters