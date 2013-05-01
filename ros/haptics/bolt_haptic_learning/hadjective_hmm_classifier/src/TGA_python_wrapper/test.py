""" Simple example of usage for the global_align module
"""

import numpy as np
import global_align as ga

def test(d=10, T1=100, T2=120):
  np.random.seed(1)
  # generate two random time series of dimension d and duration T1, resp. T2
  seq1 = np.random.rand(T1, d)
  seq2 = np.random.rand(T2, d)
  # define the sigma parameter
  sigma = 0.5*(T1+T2)/2*np.sqrt((T1+T2)/2)
  # compute the global alignment kernel value for different triangular parameters
  diff_t = np.abs(T1-T2)
  Ts = sorted(set([0, 1, diff_t/2, diff_t, diff_t+1, diff_t*3/2, diff_t*2]))
  for triangular in Ts:
    val = ga.tga_dissimilarity(seq1, seq2, sigma, triangular)
    kval = np.exp(-val)
    if 0 < triangular <= diff_t:
      # for 0 < triangular <= diff_t, exp(-tga_d) == 0
      assert kval == 0
    print "T=%d \t exp(-tga_d)=%0.5f" % (triangular, kval)

