#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import numpy
import sys
from itertools import count
from matplotlib import cm
from numpy import cumsum, digitize, random
from scipy import optimize, stats
from scipy.interpolate import interp1d

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
            return numpy.array(s)
    else:
        def compute(function, tj):
            if len(tj.shape) > 1:
                return function(*tj)
            return function(tj)

    t = numpy.array(t)
    if len(t.shape) > 2:
        raise TypeError('data array can have at most 2 dimensions')

    n = t.shape[-1]
    if n_obj == 0:
      n_obj = n
    x = []
    if len(t.shape) > 1:
        x = [compute(function, numpy.array([ti[random.randint(0, n, n_obj)]
                                            for ti in t])) \
             for i in xrange(n_samples)]
    else:
        x = [compute(function, t[random.randint(0, n, n_obj)]) \
             for i in xrange(n_samples)]
    if asym_errors:
        def err(y):
            yo = numpy.median(y)
            return numpy.absolute(numpy.percentile(y, [16,84]) - yo)
        if isinstance(x[0], float):
            out = err(x)
        else:
            out = [err(xi) for xi in numpy.transpose(x)]
    else:
        if isinstance(x[0], float):
            out = numpy.std(x)
        else:
            s = [numpy.std(xi) for xi in numpy.transpose(x)]
            out = numpy.array(s)
    if full_output:
        return out, numpy.array(x)
    return out


def Cbi(x, c=6.):
    """
    Biweight Location estimator
    """
    mad = MAD(x)
    m = numpy.median(x)
    u = (x - m) / (c * mad)
    good = (abs(u) < 1)
    num = sum((x[good] - m) * (1 - u[good]**2) ** 2)
    den = sum((1 - u[good]**2) ** 2)
    return m + num / den


def draw(x, weights, size=None):
    """
    Draw samples from an arbitrary distribution. In general, `x` and
    `weights` would be the range and height of a histogram.

    """
    weights /= numpy.sum(weights)
    if size is None:
        return x[digitize(random.random(1), cumsum(weights))][0]
    return x[digitize(random.random(size), cumsum(weights))]


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
            return numpy.array(s)
    else:
        def compute(function, tj, **kwargs):
            if len(tj.shape) > 1:
                return function(*tj, **kwargs)
            return function(tj, **kwargs)

    t = numpy.array(t)
    if len(t.shape) > 2:
        raise TypeError('t can have at most 2 dimensions')

    n = t.shape[-1]
    if n_obj == 0:
        n_obj = n
    x = []
    if len(t.shape) > 1:
        for ii in xrange(n_samples):
            j = random.randint(0, n, n-n_remove)
            t1 = numpy.array([ti[j] for ti in t])
            x.append(compute(function, t1, **kwargs))
    else:
        for i in xrange(n_samples):
            j = random.randint(0, n, n-n_remove)
            x.append(compute(function, t[j], **kwargs))
    if asym_errors:
        def err(y):
            yo = numpy.median(y)
            return numpy.absolute(numpy.percentile(y, [16,84]) - yo)
        if isinstance(x[0], float):
            out = err(x)
        else:
            out = [err(xi) for xi in numpy.transpose(x)]
    else:
        if isinstance(x[0], float):
            out = numpy.std(x)
        else:
            s = [numpy.std(xi) for xi in numpy.transpose(x)]
            out = numpy.array(s)
    if full_output:
        return out, numpy.array(x)
    return out


def MAD(x):
    n = numpy.absolute(x - numpy.median(x))
    return numpy.median(n)


def percentile_from_histogram(hist, centers, p):
    """
    return the p-th percentile by interpolating a histogram's CDF

    """
    cdf = numpy.cumsum(hist)
    cdf_thin = interp1d(centers, cdf/cdf.max(), kind='slinear')
    def root(x):
        return cdf_thin(x) - p/100.
    val = optimize.brentq(root, centers[0], centers[-1])
    return val


def Sbi(x, c=9., location='median'):
    """
    Biweight Scale estimator
    """
    n = len(x)
    mad = MAD(x)
    if location == 'median':
        m = numpy.median(x)
    elif location == 'biweight':
        m = Cbi(x)
    u = (x - m) / (c * mad)
    good = (abs(u) < 1)
    num = sum((x[good] - m) ** 2 * (1 - u[good]**2) ** 4)
    den = sum((1 - u[good]**2) * (1 - 5 * u[good]**2))
    return (n / (n - 1)**0.5) * num**0.5 / abs(den)


def sigmaclip(sample, clip=3, loc=numpy.median, scale=numpy.std,
              ret='sample'):
    """
    returns the resulting array, the median (or average), and the standard
    deviation of it . *ret* can be 'sample', in which case the function
    returns the resulting sample, or 'indices', in which case it returns the
    indices of the returing sample in the original one.
    """
    sample = numpy.array(sample)
    med = loc(sample)
    std = scale(sample)
    sample_med = sample - med
    if ret == 'sample':
        return sample[abs(sample_med) < clip * std]
    elif ret == 'indices':
        ind = numpy.arange(len(sample))
        return ind[abs(sample_med) < clip * std]
    return


