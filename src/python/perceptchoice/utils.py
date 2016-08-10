import numpy as np
from scipy import optimize


def mdm_outliers(dist):
    c=1.1926
    medians=[]
    for idx,x in enumerate(dist):
        medians.append(np.median(np.abs(x-dist)))
    mdm=c*np.median(medians)
    outliers=[]
    for idx,x in enumerate(dist):
        if np.median(np.abs(x-dist))/mdm>3:
            outliers.append(idx)
    return outliers

class _baseFunctionFit:
    """
    From psychopy
    Not needed by most users except as a superclass for developping your own functions

    You must overide the eval and inverse methods and a good idea to overide the _initialGuess
    method aswell.
    """

    def __init__(self, xx, yy, sems=1.0, guess=None, display=1,
                 expectedMin=0.5):
        self.xx = np.asarray(xx)
        self.yy = np.asarray(yy)
        self.sems = np.asarray(sems)
        self.expectedMin = expectedMin
        self.display=display
        # for holding error calculations:
        self.ssq=0
        self.rms=0
        self.chi=0
        #initialise parameters
        if guess is None:
            self.params = self._initialGuess()
        else:
            self.params = guess

        #do the calculations:
        self._doFit()

    def _doFit(self):
        #get some useful variables to help choose starting fit vals
        self.params = optimize.fmin_powell(self._getErr, self.params, (self.xx,self.yy,self.sems),disp=self.display)
        #        self.params = optimize.fmin_bfgs(self._getErr, self.params, None, (self.xx,self.yy,self.sems),disp=self.display)
        self.ssq = self._getErr(self.params, self.xx, self.yy, 1.0)
        self.chi = self._getErr(self.params, self.xx, self.yy, self.sems)
        self.rms = self.ssq/len(self.xx)

    def _initialGuess(self):
        xMin = min(self.xx); xMax = max(self.xx)
        xRange=xMax-xMin; xMean= (xMax+xMin)/2.0
        guess=[xMean, xRange/5.0]
        return guess

    def _getErr(self, params, xx,yy,sems):
        mod = self.eval(xx, params)
        err = sum((yy-mod)**2/sems)
        return err

    def eval(self, xx=None, params=None):
        """Returns fitted yy for any given xx value(s).
        Uses the original xx values (from which fit was calculated)
        if none given.

        If params is specified this will override the current model params."""
        yy=xx
        return yy

    def inverse(self, yy, params=None):
        """Returns fitted xx for any given yy value(s).

        If params is specified this will override the current model params.
        """
        #define the inverse for your function here
        xx=yy
        return xx


class FitWeibull(_baseFunctionFit):
    """Fit a Weibull function (either 2AFC or YN)
    of the form::

        y = chance + (1.0-chance)*(1-exp( -(xx/alpha)**(beta) ))

    and with inverse::

        x = alpha * (-log((1.0-y)/(1-chance)))**(1.0/beta)

    After fitting the function you can evaluate an array of x-values
    with ``fit.eval(x)``, retrieve the inverse of the function with
    ``fit.inverse(y)`` or retrieve the parameters from ``fit.params``
    (a list with ``[alpha, beta]``)"""
    def eval(self, xx=None, params=None):
        if params is None:  params=self.params #so the user can set params for this particular eval
        alpha = params[0];
        if alpha<=0: alpha=0.001
        beta = params[1]
        xx = np.asarray(xx)
        yy =  self.expectedMin + (1.0-self.expectedMin)*(1-np.exp( -(xx/alpha)**(beta) ))
        return yy
    def inverse(self, yy, params=None):
        if params is None: params=self.params #so the user can set params for this particular inv
        alpha = params[0]
        beta = params[1]
        xx = alpha * (-np.log((1.0-yy)/(1-self.expectedMin))) **(1.0/beta)
        return xx

class FitRT(_baseFunctionFit):
    """Fit a Weibull function (either 2AFC or YN)
    of the form::

        y = chance + (1.0-chance)*(1-exp( -(xx/alpha)**(beta) ))

    and with inverse::

        x = alpha * (-log((1.0-y)/(1-chance)))**(1.0/beta)

    After fitting the function you can evaluate an array of x-values
    with ``fit.eval(x)``, retrieve the inverse of the function with
    ``fit.inverse(y)`` or retrieve the parameters from ``fit.params``
    (a list with ``[alpha, beta]``)"""
    def eval(self, xx=None, params=None):
        if params is None:  params=self.params #so the user can set params for this particular eval
        a = params[0]
        k = params[1]
        tr = params[2]
        xx = np.asarray(xx)
        yy = a*np.tanh(k*xx)+tr
        return yy
    def inverse(self, yy, params=None):
        if params is None: params=self.params #so the user can set params for this particular inv
        a = params[0]
        k = params[1]
        tr = params[2]
        xx = np.arctanh((yy-tr)/a)/k
        return xx

class FitSigmoid(_baseFunctionFit):
    def eval(self, xx=None, params=None):
        if params is None:  params=self.params #so the user can set params for this particular eval
        x0 = params[0]
        k=params[1]
        xx = np.asarray(xx)
        #yy = a+1.0/(1.0+np.exp(-k*xx))
        yy =1.0/(1.0+np.exp(-k*(xx-x0)))
        return yy

    def inverse(self, yy, params=None):
        if params is None:  params=self.params #so the user can set params for this particular eval
        x0 = params[0]
        k=params[1]
        #xx = -np.log((1/(yy-a))-1)/k
        xx = -np.log((1.0/yy)-1.0)/k+x0
        return xx