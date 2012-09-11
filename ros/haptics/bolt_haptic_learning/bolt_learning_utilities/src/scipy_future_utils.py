from math import ceil, log
import numpy as np
from numpy.fft import irfft
from scipy.special import sinc
from scipy.signal.signaltools import get_window
from scipy.signal import lfilter


#from scipy.signal._arraytools import axis_slice, axis_reverse, odd_ext, even_ext, const_ext

__all__ = ['firwin']


def firwin(numtaps, cutoff, width=None, window='hamming', pass_zero=True,
                                                        scale=True, nyq=1.0):
    """
    FIR filter design using the window method.

    This function computes the coefficients of a finite impulse response
    filter.  The filter will have linear phase; it will be Type I if
    `numtaps` is odd and Type II if `numtaps` is even.

    Type II filters always have zero response at the Nyquist rate, so a
    ValueError exception is raised if firwin is called with `numtaps` even and
    having a passband whose right end is at the Nyquist rate.

    Parameters
    ----------
    numtaps : int
        Length of the filter (number of coefficients, i.e. the filter
        order + 1).  `numtaps` must be even if a passband includes the
        Nyquist frequency.

    cutoff : float or 1D array_like
        Cutoff frequency of filter (expressed in the same units as `nyq`)
        OR an array of cutoff frequencies (that is, band edges). In the
        latter case, the frequencies in `cutoff` should be positive and
        monotonically increasing between 0 and `nyq`.  The values 0 and
        `nyq` must not be included in `cutoff`.

    width : float or None
        If `width` is not None, then assume it is the approximate width
        of the transition region (expressed in the same units as `nyq`)
        for use in Kaiser FIR filter design.  In this case, the `window`
        argument is ignored.
    window : string or tuple of string and parameter values
        Desired window to use. See `scipy.signal.get_window` for a list
        of windows and required parameters.

    pass_zero : bool
        If True, the gain at the frequency 0 (i.e. the "DC gain") is 1.
        Otherwise the DC gain is 0.

    scale : bool
        Set to True to scale the coefficients so that the frequency
        response is exactly unity at a certain frequency.
        That frequency is either:

            - 0 (DC) if the first passband starts at 0 (i.e. pass_zero
              is True);
            - `nyq` (the Nyquist rate) if the first passband ends at
              `nyq` (i.e the filter is a single band highpass filter);
              center of first passband otherwise.

    nyq : float
        Nyquist frequency.  Each frequency in `cutoff` must be between 0
        and `nyq`.

    Returns
    -------
    h : 1-D ndarray
        Coefficients of length `numtaps` FIR filter.

    Raises
    ------
    ValueError
        If any value in `cutoff` is less than or equal to 0 or greater
        than or equal to `nyq`, if the values in `cutoff` are not strictly
        monotonically increasing, or if `numtaps` is even but a passband
        includes the Nyquist frequency.

    Examples
    --------

    Low-pass from 0 to f::
    >>> firwin(numtaps, f)

    Use a specific window function::

    >>> firwin(numtaps, f, window='nuttall')

    High-pass ('stop' from 0 to f)::

    >>> firwin(numtaps, f, pass_zero=False)

    Band-pass::

    >>> firwin(numtaps, [f1, f2], pass_zero=False)

    Band-stop::

    >>> firwin(numtaps, [f1, f2])

    Multi-band (passbands are [0, f1], [f2, f3] and [f4, 1])::

    >>>firwin(numtaps, [f1, f2, f3, f4])

    Multi-band (passbands are [f1, f2] and [f3,f4])::

    >>> firwin(numtaps, [f1, f2, f3, f4], pass_zero=False)

    See also
    --------
    scipy.signal.firwin2

    """

    # The major enhancements to this function added in November 2010 were
    # developed by Tom Krauss (see ticket #902).

    cutoff = np.atleast_1d(cutoff) / float(nyq)

    # Check for invalid input.
    if cutoff.ndim > 1:
        raise ValueError("The cutoff argument must be at most "
                         "one-dimensional.")
    if cutoff.size == 0:
        raise ValueError("At least one cutoff frequency must be given.")
    if cutoff.min() <= 0 or cutoff.max() >= 1:
        raise ValueError("Invalid cutoff frequency: frequencies must be "
                         "greater than 0 and less than nyq.")
    if np.any(np.diff(cutoff) <= 0):
        raise ValueError("Invalid cutoff frequencies: the frequencies "
                         "must be strictly increasing.")

    if width is not None:
        # A width was given.  Find the beta parameter of the Kaiser window
        # and set `window`.  This overrides the value of `window` passed in.
        atten = kaiser_atten(numtaps, float(width) / nyq)
        beta = kaiser_beta(atten)
        window = ('kaiser', beta)

    pass_nyquist = bool(cutoff.size & 1) ^ pass_zero
    if pass_nyquist and numtaps % 2 == 0:
        raise ValueError("A filter with an even number of coefficients must "
                            "have zero response at the Nyquist rate.")

    # Insert 0 and/or 1 at the ends of cutoff so that the length of cutoff
    # is even, and each pair in cutoff corresponds to passband.
    cutoff = np.hstack(([0.0] * pass_zero, cutoff, [1.0] * pass_nyquist))

    # `bands` is a 2D array; each row gives the left and right edges of
    # a passband.
    bands = cutoff.reshape(-1, 2)

    # Build up the coefficients.
    alpha = 0.5 * (numtaps - 1)
    m = np.arange(0, numtaps) - alpha
    h = 0
    for left, right in bands:
        h += right * sinc(right * m)
        h -= left * sinc(left * m)

    # Get and apply the window function.
    win = get_window(window, numtaps, fftbins=False)
    h *= win

    # Now handle scaling if desired.
    if scale:
        # Get the first passband.
        left, right = bands[0]
        if left == 0:
            scale_frequency = 0.0
        elif right == 1:
            scale_frequency = 1.0
        else:
            scale_frequency = 0.5 * (left + right)
        c = np.cos(np.pi * m * scale_frequency)
        s = np.sum(h * c)
        h /= s

    return h





