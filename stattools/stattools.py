#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import numpy as np
import sys
from itertools import count
from matplotlib import cm
from numpy import cumsum, digitize, random
from scipy import optimize, stats
from scipy.interpolate import interp1d
from scipy.special import erf

if sys.version_info[0] == 2:
    from itertools import izip
else:
    izip = zip
    xrange = range


def bootstrap(function, t, n_obj=0, n_samples=1000,
              full_output=False, asym_errors=False, **kwargs):
    """
    Bootstrap resampling for a given statistic

    Parameters
    ----------
      function  : function or array-like of functions
                  the function or functions to be bootstrapped. If it is a
                  list of functions, all functions must take the same number
                  of arguments
      t         : array of floats
                  the values(s) for which the function(s) will be computed.
                  Must contain all the values needed by the function(s).
      n_obj     : int
                  baseline number of objects. The default of zero corresponds
                  to len(t). If another number is given then this is
                  essentially jackknifing
      n_samples : int (default 1000)
                  number of bootstrap iterations
      asym_errors : bool (default False)
                  whether to return asymmetric errors. If True, the errors
                  will be the 16th and 84th quantiles of the bootstrap
                  distribution. Otherwise, the error is the standard deviation
                  of the distribution.
      **kwargs  : any keyword arguments accepted by *function*

    Returns
    -------
      errors    : float or array of floats
                  bootstrap errors. If stats is a list, then errors is a list
                  with the same number of objects. For each stat, will be
                  either a float (asym_errors==False) or a list of two floats
                  (asym_errors==True).

    """
    if hasattr(function, '__iter__'):
        def compute(function, tj):
            if len(tj.shape) > 1:
                s = [f(*tj) for f in function]
            else:
                s = [f(tj) for f in function]
            return np.array(s)
    else:
        def compute(function, tj):
            if len(tj.shape) > 1:
                return function(*tj)
            return function(tj)

    t = np.array(t)
    if len(t.shape) > 2:
        raise TypeError('data array can have at most 2 dimensions')

    n = t.shape[-1]
    if n_obj == 0:
      n_obj = n
    x = []
    if len(t.shape) > 1:
        x = [compute(function, np.array([ti[random.randint(0, n, n_obj)]
                                            for ti in t])) \
             for i in xrange(n_samples)]
    else:
        x = [compute(function, t[random.randint(0, n, n_obj)]) \
             for i in xrange(n_samples)]
    if asym_errors:
        def err(y):
            yo = np.median(y)
            return np.absolute(np.percentile(y, [16,84]) - yo)
        if isinstance(x[0], float):
            out = err(x)
        else:
            out = [err(xi) for xi in np.transpose(x)]
    else:
        if isinstance(x[0], float):
            out = np.std(x)
        else:
            s = [np.std(xi) for xi in np.transpose(x)]
            out = np.array(s)
    if full_output:
        return out, np.array(x)
    return out


def Cbi(x, c=6.):
    """
    Biweight Location estimator
    """
    mad = MAD(x)
    m = np.median(x)
    u = (x - m) / (c * mad)
    good = (abs(u) < 1)
    num = sum((x[good] - m) * (1 - u[good]**2) ** 2)
    den = sum((1 - u[good]**2) ** 2)
    return m + num / den


def draw(x, weights, size=None):
    """
    Draw samples from an arbitrary distribution. In general, `x` and
    `weights` would be the range and height of a histogram. Note that
    the samples are drawn from `x`, so a sparse sampling of `x` will
    result in a sparse sampling of the random draws as well.
    """
    weights /= np.sum(weights)
    if size is None:
        return x[digitize(random.random(1), cumsum(weights))][0]
    return x[digitize(random.random(size), cumsum(weights))]


def generalized_erf(x, a, b, c=1, d=0):
    """
    Error function, generalized to have arbitrary minimum, maximum,
    and transition location

    Parameters
    ----------
    a : float or array-like
        defines width of the function. `b > 1` makes this function
        transition faster, while `b < 1` makes the transition slower
    b : float or array-like
        location of the function mid point
    c : float or array-like
        function maximum, reached for `x >> c`
    d : float or array-like
        floor of the function, reached for `x << c`
    """
    return d + 0.5*c*(1 + erf(a*x-b))


def jackknife(function, t, n_remove=1, n_samples=1000,
              full_output=False, asym_errors=False, **kwargs):
    """
    Jackknife resampling for a given statistic

    Parameters
    ----------
      function  : function or array-like of functions
                  the function or functions to be jackknifed. If it is a list
                  of functions, all functions must take the same number of
                  arguments
      t         : array of floats
                  the values(s) for which the function(s) will be computed.
                  Must contain all the values needed by the function(s).
      n_remove  : int (default 1)
                  number of objects to remove for each sample
      n_samples : int (default 1000)
                  number of iterations
      asym_errors : bool (default False)
                  whether to return asymmetric errors. If True, the errors
                  will be the 16th and 84th quantiles of the jackknife
                  distribution. Otherwise, the error is the standard deviation
                  of the distribution.
      **kwargs  : any keyword arguments accepted by *function*

    Returns
    -------
      errors    : float or array of floats
                  jackknife errors. If stats is a list, then errors is a list
                  with the same number of objects. For each stat, will be
                  either a float (asym_errors==False) or a list of two floats
                  (asym_errors==True).

    """
    if hasattr(function, '__iter__'):
        def compute(function, tj, **kwargs):
            if len(tj.shape) > 1:
                s = [f(*tj, **kwargs) for f in function]
            else:
                s = [f(tj, **kwargs) for f in function]
            return np.array(s)
    else:
        def compute(function, tj, **kwargs):
            if len(tj.shape) > 1:
                return function(*tj, **kwargs)
            return function(tj, **kwargs)

    t = np.array(t)
    if len(t.shape) > 2:
        raise TypeError('t can have at most 2 dimensions')

    n = t.shape[-1]
    if n_obj == 0:
        n_obj = n
    x = []
    if len(t.shape) > 1:
        for ii in xrange(n_samples):
            j = random.randint(0, n, n-n_remove)
            t1 = np.array([ti[j] for ti in t])
            x.append(compute(function, t1, **kwargs))
    else:
        for i in xrange(n_samples):
            j = random.randint(0, n, n-n_remove)
            x.append(compute(function, t[j], **kwargs))
    if asym_errors:
        def err(y):
            yo = np.median(y)
            return np.absolute(np.percentile(y, [16,84]) - yo)
        if isinstance(x[0], float):
            out = err(x)
        else:
            out = [err(xi) for xi in np.transpose(x)]
    else:
        if isinstance(x[0], float):
            out = np.std(x)
        else:
            s = [np.std(xi) for xi in np.transpose(x)]
            out = np.array(s)
    if full_output:
        return out, np.array(x)
    return out


def MAD(x):
    n = np.absolute(x - np.median(x))
    return np.median(n)


def percentile_from_histogram(hist, centers, p):
    """
    return the p-th percentile by interpolating a histogram's CDF

    """
    cdf = np.cumsum(hist)
    cdf_thin = interp1d(centers, cdf/cdf.max(), kind='slinear')
    def root(x):
        return cdf_thin(x) - p/100.
    val = optimize.brentq(root, centers[0], centers[-1])
    return val


def rms(x, clip=3, which='both', center='median', tol=1e-3):
    """Root-mean-square value

    Typically used to calculate the noise in an image. Infinite or nan
    values are discarded, and then pixels above and below `clip` (in
    units of the standard deviation) are removed iteratively.

    Parameters
    ----------
    x : `np.array`
        array from which the rms will be calculated
    clip : `float`
        clipping threshold, in units of the standard deviation. Set to
        zero to run without clipping (will simply return the standard
        deviation)
    which : {'positive','negative','both'}
        whether to clip in the positive or negative direction, or both
    center : {'mean','median','Cbi'}
        statistic to use as the central value. 'Cbi' is the biweight
        estimate, as implemented in this module.
    tol : float
        minimum fractional difference between iterations in order to be
        considered converged
    """
    x = x[np.isfinite(x)]
    s = 10 * np.std(x)
    if clip == 0:
        return s
    m = {'median': np.median, 'mean': np.mean, 'Cbi': Cbi}
    m = m[center]
    while (np.std(x)/s - 1) > tol:
        s = np.std(x)
        mx = m(x)
        if which in ('both', 'positive'):
            x = x[x - mx < clip*s]
        if which in ('both', 'negative'):
            x = x[mx - x < clip*s]
    return np.std(x)


def Sbi(x, c=9., location='median'):
    """
    Biweight Scale estimator
    """
    n = len(x)
    mad = MAD(x)
    if location == 'median':
        m = np.median(x)
    elif location == 'biweight':
        m = Cbi(x)
    u = (x - m) / (c * mad)
    good = (abs(u) < 1)
    num = sum((x[good] - m) ** 2 * (1 - u[good]**2) ** 4)
    den = sum((1 - u[good]**2) * (1 - 5 * u[good]**2))
    return (n / (n - 1)**0.5) * num**0.5 / abs(den)


def sigmaclip(sample, clip=3, loc=np.median, scale=np.std,
              ret='sample'):
    """
    returns the resulting array, the median (or average), and the standard
    deviation of it . *ret* can be 'sample', in which case the function
    returns the resulting sample, or 'indices', in which case it returns the
    indices of the returing sample in the original one.
    """
    sample = np.array(sample)
    med = loc(sample)
    std = scale(sample)
    sample_med = sample - med
    if ret == 'sample':
        return sample[abs(sample_med) < clip * std]
    elif ret == 'indices':
        ind = np.arange(len(sample))
        return ind[abs(sample_med) < clip * std]
    return


def wstd(X, weights, xo=None, axis=None):
    """
    Compute a weighted standard deviation

    Parameters
    ----------
        X : array of data points
        weights : array of weights, same shape as `X`
        xo : array of means, one entry per dimension in `X` (optional)
        axis : axis along which to do the calculation

    """
    # if means are not given, calculate weighted means
    if xo == None:
        xo = np.sum(weights*X, axis=axis) / np.sum(weights, axis=axis)
    j = (weights > 0)
    num = np.sum(weights * (X-xo)**2, axis=axis)
    den = (np.sum(j, axis=axis)-1.) / np.sum(j, axis=axis) \
          * np.sum(weights, axis=axis)
    return (num/den)**0.5

