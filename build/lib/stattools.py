#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy
from itertools import count, izip
from matplotlib import cm
from numpy import random
from scipy import optimize, stats

def bootstrap(function, t, n_obj=0, n_samples=1000, asym_errors=False):
    """
    Bootstrap resampling for a given statistic

    Parameters
    ----------
      function  : function or array-like of functions
                  the function or functions to be computed. If it is a list of
                  functions, all functions must take the same number of
                  arguments
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

    Returns
    -------
      errors    : float or array of floats
                  bootstrap errors. If stats is a list, then errors is a list
                  with the same number of objects. For each stat, will be
                  either a float (asym_errors==False) or a list of two floats
                  (asym_errors==True).

    """
    if type(function) in (list, tuple, numpy.ndarray):
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
      raise TypeError('t can have at most 2 dimensions')

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
        out = (yo - stats.scoreatpercentile(y, 16),
               stats.scoreatpercentile(y, 84) - yo)
        return out
      if type(x[0]) == float:
        return err(x)
      else:
        return [err(xi) for xi in numpy.transpose(x)]
    else:
      if type(x[0]) in (float, numpy.float64):
        return numpy.std(x)
      else:
        s = [numpy.std(xi) for xi in numpy.transpose(x)]
        return numpy.array(s)

def Cbi(z, c=6.):
    """
    Biweight Location estimator
    """
    mad = MAD(z)
    m = numpy.median(z)
    u = (z - m) / (c * mad)
    good = numpy.arange(len(u))[abs(u) < 1]
    num = sum((z[good] - m) * (1 - u[good]**2) ** 2)
    den = sum((1 - u[good]**2) ** 2)
    return m + num / den

def contour_levels(x, y=[], bins=10, levels=(0.68,0.95)):
    if len(y) > 0:
      if len(x) != len(y):
        msg = 'Invalid input for arrays; must be either 1 2d array'
        msg += ' or 2 1d arrays'
        raise ValueError(msg)
    else:
      if len(numpy.array(x).shape) != 2:
        msg = 'Invalid input for arrays; must be either 1 2d array'
        msg += ' or 2 1d arrays'
        raise ValueError(msg)
    def findlevel(lo, hist, level):
      return 1.0 * hist[hist >= lo].sum()/hist.sum() - level
    if len(x) == len(y):
      hist, xedges, yedges = numpy.histogram2d(x, y, bins=bins)
      hist = numpy.transpose(hist)
      extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])
    elif len(y) == 0:
      hist = numpy.array(x)
    lvs = [optimize.bisect(findlevel, hist.min(), hist.max(),
                           args=(hist, l)) for l in levels]
    return lvs

def corner(X, names='', labels='', bins=None, clevels=(0.68,0.95),
           colors='k', cmap=None, ls2d='-', style1d='curve',
           truths=None, background=None, bcolor='r', alpha=0.5,
           output=''):
    """
    Do a corner plot with the posterior parameters of an MCMC chain.
    Note that there are still some issues with the tick labels.

    Parameters
    ----------
      X         : array-like
                  all posterior parameters. Can also be the outputs of
                  more than one chain, given as an array of arrays of models
                  (e.g., X = [[A1, B1, C1], [A2, B2, C2]])
      names     : list of strings (optional)
                  Names for each of the chains. Will be used to show a legend
                  in the (empty) upper corner
      labels    : list of strings (optional)
                  names of the parameters
      bins      : float or array (optional)
                  anything accepted under the *bins* argument of pylab.hist,
                  or a list of these. If not given, a good binning choice
                  will be made automatically
      clevels   : list of floats between 0 and 1 (default: (0.68,0.95))
                  percentiles at which to show contours
      colors    : any argument taken by the *colors* argument of
                  pylab.contour(), or a tuple of them if more than one
                  model is to be plotted
                  color of the contours
      ls2d      : linestyle value or list of them
                  linestyle for the contours NOT YET IMPLEMENTED
      style1d   : {'hist', 'curve'} (default 'curve')
                  if hist, plot the 1d posterior as a histogram; if 'curve',
                  plot them as a curve
      truths    : {list of floats, 'medians', None} (default None)
                  reference values for each parameter. NOT YET IMPLEMENTED.
      background : {None, 'points', 'density', 'filled'} (default None)
                  If not None, then either points, a smoothed 2d histogram,
                  or filled contours are plotted beneath contours. ONLY
                  'points' AND 'density' IMPLEMENTED SO FAR.
      bcolor    : color property, consistent with *background*
                  color of the points or filled contours, or colormap of the
                  2d density background
      alpha     : float between 0 and 1 (default 0.5)
                  transparency of the points if shown
      output    : string (optional)
                  filename to save the plot. If not given, the plot is shown
                  with pylab.show()

    Returns
    -------
      None

    """
    import pylab
    if style1d == 'curve':
        from scipy import interpolate
    # automatically adjust shape
    shape = numpy.array(X).shape
    if len(shape) == 2:
        if shape[1] < shape[0]:
            X = numpy.transpose(X)
        nnames = 1
        X = (X,)
    else:
        nnames = shape[0]
    if len(X) > 1 and background == 'points':
        background = None
    if min(shape[-2:]) == 0:
        msg = 'stattools.corner: received empty array.'
        msg += ' It is possible that you set the burn-in to be longer'
        msg += ' than the chain itself!'
        raise ValueError(msg)
    ndim = min(shape[-2:])
    nsamples = max(shape[-2:])
    # check clevels
    if 1 < clevels[0] < 100:
        clevels = [cl/100. for cl in clevels]
    # arbitrary
    if bins is None:
        bins = min(nsamples/1000., 30)
    figsize = 3 * ndim
    axsize = 0.85 / ndim
    fontsize = 13 + 0.5*ndim
    if len(X) == 1:
        if type(colors) == str:
            color1d = colors
        else:
            color1d = 'k'
    else:
        if len(colors) == len(X):
            color1d = colors
        # supports up to 6 names (plot would be way overcrowded!)
        else:
            color1d = ('g', 'orange', 'c', 'm', 'b', 'y')
    pylab.figure(figsize=(figsize,figsize))
    # diagonals first
    plot_ranges = []
    for i in xrange(ndim):
        ax = pylab.axes([0.1+axsize*i, 0.95-axsize*(i+1),
                         0.95*axsize, 0.95*axsize],
                        yticks=[])
        if i < ndim-1:
            ax.set_xticklabels([])
        peak = 0
        if style1d == 'curve':
            for m, Xm in enumerate(X):
                ho, edges = numpy.histogram(Xm[i], bins=bins, normed=True)
                xo = [(edges[ii]+edges[ii-1])/2 \
                      for ii in xrange(1, len(edges))]
                xn = numpy.linspace(min(xo), max(xo), 500)
                hn = interpolate.spline(xo, ho, xn)
                #pylab.plot(xo, hist, ls='-', color=color1d[m])
                pylab.plot(xn, hn, ls='-', color=color1d[m])
                if max(hn) > peak:
                    peak = max(hn)
        else:
            for m, Xm in enumerate(X):
                n = pylab.hist(Xm[i], bins=bins, histtype='step',
                               color=color1d[m], normed=True)[0]
                if max(n) > peak:
                    peak = max(n)
        if i == ndim-1 and labels:
            if len(labels) >= ndim:
                pylab.xlabel(labels[i], fontsize=fontsize)
        # to avoid overcrowding tick labels
        tickloc = pylab.MaxNLocator(5)
        ax.xaxis.set_major_locator(tickloc)
        plot_ranges.append(ax.get_xlim())
        pylab.ylim(0, 1.1*peak)
    # lower off-diagonals
    for i in xrange(1, ndim): # vertical axes
        for j in xrange(i): # horizontal axes
            ax = pylab.axes([0.1+axsize*j, 0.95-axsize*(i+1),
                             0.95*axsize, 0.95*axsize])
            for m, Xm in enumerate(X):
                h = numpy.histogram2d(Xm[j], Xm[i], bins=bins,
                                      range=(plot_ranges[j],plot_ranges[i]))
                h = numpy.transpose(h[0])
                levels = contour_levels(Xm[j], Xm[i], bins=bins,
                                        levels=clevels)
                extent = numpy.append(plot_ranges[j], plot_ranges[i])
                if background == 'points':
                    pylab.plot(Xm[j], Xm[i], ',', color=bcolor, alpha=alpha,
                               zorder=-10)
                elif background == 'density':
                    pylab.imshow([Xm[i], Xm[j]], cmap=cm.Reds,
                                 extent=extent)
                elif background == 'filled':
                    clvs = numpy.append(clevels, 1)
                    lvs = contour_levels(Xm[j], Xm[i], bins=bins, levels=clvs)
                    for l in xrange(len(levels), 0, -1):
                        pylab.contourf(h, (lvs[l-1],lvs[l]),
                                       extent=extent, colors=bcolor[l-1])
                pylab.contour(h, levels, colors=color1d[m],
                              linestyles='solid', extent=extent, zorder=10)
            if j == 0:
                pylab.ylabel(labels[i], fontsize=fontsize)
            else:
                ax.set_yticklabels([])
            if i == ndim-1:
                pylab.xlabel(labels[j], fontsize=fontsize)
            else:
                ax.set_xticklabels([])
            # to avoid overcrowding tick labels
            xloc = pylab.MaxNLocator(5)
            ax.xaxis.set_major_locator(xloc)
            yloc = pylab.MaxNLocator(5)
            ax.yaxis.set_major_locator(yloc)
    # dummy legend axes
    if len(X) > 1 and len(names) == len(X):
        lax = pylab.axes([0.1+axsize*(ndim-1), 0.95,
                          0.95*axsize, 0.95*axsize],
                         xticks=[], yticks=[])
        lax.set_frame_on(False)
        for c, model in zip(color1d, names):
            pylab.plot([], [], ls='-', lw=2, color=c, label=model)
        lg = pylab.legend(loc='center', ncol=1, fontsize=fontsize+2)
        lg.get_frame().set_alpha(0)
    if output:
        pylab.savefig(output, format=output[-3:])
        pylab.close()
    else:
        pylab.show()
    return

def jackknife(function, t, n_remove=1, n_samples=1000,
              asym_errors=False, **kwargs):
    """
    Jackknife resampling for a given statistic

    Parameters
    ----------
      function  : function or array-like of functions
                  the function or functions to be computed. If it is a list of
                  functions, all functions must take the same number of
                  arguments
      t         : array of floats
                  the values(s) for which the function(s) will be computed.
                  Must contain all the values needed by the function(s).
      n_remove  : int (default 1)
                  number of objects to remove for each sample
      n_samples : int (default 1000)
                  number of bootstrap iterations
      asym_errors : bool (default False)
                  whether to return asymmetric errors. If True, the errors
                  will be the 16th and 84th quantiles of the bootstrap
                  distribution. Otherwise, the error is the standard deviation
                  of the distribution.
      **kwargs  : any kwargs accepted by function()

    Returns
    -------
      errors    : float or array of floats
                  bootstrap errors. If stats is a list, then errors is a list
                  with the same number of objects. For each stat, will be
                  either a float (asym_errors==False) or a list of two floats
                  (asym_errors==True).

    """
    if type(function) in (list, tuple, numpy.ndarray):
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
        out = (yo - stats.scoreatpercentile(y, 16),
               stats.scoreatpercentile(y, 84) - yo)
        return out
      if type(x[0]) == float:
        return err(x)
      else:
        return [err(xi) for xi in numpy.transpose(x)]
    else:
      if type(x[0]) == float:
        return numpy.std(x)
      else:
        s = [numpy.std(xi) for xi in numpy.transpose(x)]
        return numpy.array(s)

def MAD(z):
    n = numpy.absolute(z - numpy.median(z))
    return numpy.median(n)

def Sbi(z, c=9., location='median'):
    """
    Biweight Scale estimator
    """
    n = len(z)
    mad = MAD(z)
    if location == 'median':
        m = numpy.median(z)
    elif location == 'biweight':
        m = Cbi(z)
    u = (z - m) / (c * mad)
    good = numpy.arange(n)[abs(u) < 1]
    num = sum((z[good] - m) ** 2 * (1 - u[good]**2) ** 4)
    den = sum((1 - u[good]**2) * (1 - 5 * u[good]**2))
    return (n / numpy.sqrt(n - 1)) * numpy.sqrt(num) / abs(den)

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
