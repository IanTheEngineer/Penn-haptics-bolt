"""
Wrapper around the Global Alignment kernel code from M. Cuturi

Original code: http://www.iip.ist.i.kyoto-u.ac.jp/member/cuturi/GA.html

Written by Adrien Gaidon - INRIA - 2011
http://lear.inrialpes.fr/people/gaidon/

LICENSE: cf. logGAK.c
"""

import numpy as np
cimport numpy as np

cdef extern from "logGAK.c" nogil:
  double logGAK(double *seq1 , double *seq2, int nX, int nY, int dimvect, double sigma, int triangular)

def tga_dissimilarity(np.ndarray[np.double_t,ndim=2] seq1, np.ndarray[np.double_t,ndim=2] seq2, double sigma, int triangular):
  """ Compute the Triangular Global Alignment (TGA) dissimilarity score

  What is computed is minus the log of the normalized global alignment kernel
  evaluation between the two time series, in order to use in a RBF kernel as
  exp(-gamma*mlnk)

  PARAMETERS:
    seq1: T1 x d multivariate (dimension d) time series of duration T1
      two-dimensional C-contiguous (ie. read line by line) numpy array of doubles,

    seq2: T2 x d multivariate (dimension d) time series of duration T2

    sigma: double, bandwitdh of the inner distance kernel
      good practice: {0.1, 0.5, 1, 2, 5, 10} * median(dist(x, y)) * sqrt(median(length(x)))

    triangular: int, parameter to restrict the paths (closer to the diagonal) used by the kernel
      good practice: {0.25, 0.5} * median(length(x))
      Notes:
        * 0: use all paths
        * 1: measuring alignment of (same duration) series, ie
          kernel value is 0 for different durations
        * higher = more restricted thus faster
        * kernel value is also 0 for series with difference in duration > triangular-1

  RETURN:
    mlnk: double,
      minus the normalized log-kernel
      (logGAK(seq1,seq1)+logGAK(seq2,seq2))/2 - logGAK(seq1,seq2)

  """
  T1 = seq1.shape[0]
  T2 = seq2.shape[0]
  d  = seq1.shape[1]
  _d = seq2.shape[1]
  # check preconditions
  assert d == _d, "Invalid series: dimension mismatch (%d != %d)" % (d, _d)
  assert seq1.flags['C_CONTIGUOUS'] and seq2.flags['C_CONTIGUOUS'], "Invalid series: not C-contiguous"
  assert sigma > 0, "Invalid bandwidth sigma (%f)" % sigma
  assert triangular >= 0, "Invalid triangular parameter (%f)" % sigma
  # compute the global alignment kernel value
  ga12 = logGAK(<double*> seq1.data, <double*> seq2.data, <int> T1, <int> T2, <int> d, sigma, triangular)
  # compute the normalization factor
  ga11 = logGAK(<double*> seq1.data, <double*> seq1.data, <int> T1, <int> T1, <int> d, sigma, triangular)
  ga22 = logGAK(<double*> seq2.data, <double*> seq2.data, <int> T2, <int> T2, <int> d, sigma, triangular)
  nf = 0.5*(ga11+ga22)
  # return minus the normalized logarithm of the Global Alignment Kernel
  mlnk = nf - ga12
  return mlnk

