import numpy as np
import scipy
from scipy.stats import ks_1samp
from scipy.stats import norm, rv_continuous, cramervonmises
from scipy.optimize import minimize

### 1. Lifschitz-Slyozov-Wagner distribution for coarsening through ripening

class lsw_gen(rv_continuous):
    "LSW distribution"
    def __init__(self, **kwargs):
        super().__init__(a=0, b=1.5-1e-10, **kwargs)

    def _pdf(self, x):
        return (4/9)*(x)**2 * (1 + x/3)**(-7/3) * (1 - 2*x/3)**(-11/3)*np.exp(2*x/(2*x - 3))

stats_lsw = lsw_gen(name='lsw')

def lsw_cdf(r, rmin, rmax, mean):
    """Calculate the cumulative distribution function from the normal distribution for a particular value of r"""
    cdf = stats_lsw.cdf(r, scale=mean)
    cdf_min = stats_lsw.cdf(rmin, scale=mean)
    cdf_max = stats_lsw.cdf(rmax, scale=mean)
    return (cdf - cdf_min) / (cdf_max - cdf_min)    

def lsw(rmin, rmax, mean, dr=0.01):    
    """Function to compute the cdf according to the normal distribution"""
    r = np.arange(rmin, rmax, dr)
    return r, lsw_cdf(r, rmin, rmax, mean)

# Helper functions to fit the LSW distribution to data

def lsw_error(params, cutoffs, data):
    """Objective function to minimize to fit the lsw distribution to data"""
    cdf = lambda x: lsw_cdf(x, *cutoffs, params)
    ks = ks_1samp(data, cdf)
    return ks.statistic

def fit_lsw(data):
    cutoffs = [np.min(data), np.max(data)]

    """Fit the LSW distribution to data using Kolmogorov-Smirnov distance as an optimizing metric"""
    optimum = minimize(
        lsw_error, 
        x0     = np.mean(data), 
        args   = (cutoffs, data), 
        bounds = [cutoffs], 
        method = 'Nelder-Mead'
    )
    fun_cdf = lambda x: lsw_cdf(x, *cutoffs, optimum.x)
    test = cramervonmises(data, fun_cdf)
    return optimum, fun_cdf, test

### 2. Normal distribution for particles having a typical size

def normal_cdf(r, rmin, rmax, mean, sigma):
    """Calculate the cumulative distribution function from the normal distribution for a particular value of r"""
    cdf = scipy.stats.norm.cdf(r,loc=mean, scale=sigma)
    cdf_min = scipy.stats.norm.cdf(rmin,loc=mean, scale=sigma)
    cdf_max = scipy.stats.norm.cdf(rmax,loc=mean, scale=sigma)
    return (cdf - cdf_min) / (cdf_max - cdf_min)    

def normal(rmin, rmax, mean, sigma, dr=0.01):    
    """Function to compute the cdf according to the normal distribution"""
    r = np.arange(rmin, rmax, dr)
    return r, normal_cdf(r, rmin, rmax, mean, sigma)

# Helper functions to fit data to a normal distribution

def normal_error(params, cutoffs, data):
    """Objective function to minimize to fit the lsw distribution to data"""
    cdf = lambda x: normal_cdf(x, *cutoffs, *params)
    ks = ks_1samp(data, cdf)
    return ks.statistic

def fit_normal(data):
    cutoffs = [np.min(data), np.max(data)]

    """Fit the LSW distribution to data using Kolmogorov-Smirnov distance as an optimizing metric"""
    optimum = minimize(
        normal_error, 
        x0     = [np.mean(data), np.std(data)], 
        args   = (cutoffs, data), 
        bounds = [cutoffs, (0.0, np.inf)], 
        method = 'Nelder-Mead'
    )
    fun_cdf = lambda x: normal_cdf(x, *cutoffs, *optimum.x)
    test = cramervonmises(data, fun_cdf)
    return optimum, fun_cdf, test

### 3. Log normal distribution

def log_normal_cdf(r, rmin, rmax, exp_log_mean, exp_log_sigma):
    """Calculate the cumulative distribution function from the normal distribution for a particular value of r"""
    cdf = scipy.stats.lognorm.cdf(r,scale=exp_log_mean, s=np.log(exp_log_sigma))
    cdf_min = scipy.stats.lognorm.cdf(rmin,scale=exp_log_mean, s=np.log(exp_log_sigma))
    cdf_max = scipy.stats.lognorm.cdf(rmax,scale=exp_log_mean, s=np.log(exp_log_sigma))
    return (cdf - cdf_min) / (cdf_max - cdf_min)    

def log_normal(rmin, rmax, exp_log_mean, exp_log_sigma, dr=0.01):    
    """Function to compute the cdf according to the normal distribution"""
    r = np.arange(rmin, rmax, dr)
    return r, log_normal_cdf(r, rmin, rmax, exp_log_mean, exp_log_sigma)

# Helper functions to fit data to a log normal distribution

def log_normal_error(params, cutoffs, data):
    """Objective function to minimize to fit the lsw distribution to data"""
    cdf = lambda x: log_normal_cdf(x, *cutoffs, *params)
    ks = ks_1samp(data, cdf)
    return ks.statistic

def fit_log_normal(data):
    cutoffs = [np.min(data), np.max(data)]
    
    """Fit the LSW distribution to data using Kolmogorov-Smirnov distance as an optimizing metric"""
    optimum = minimize(
        log_normal_error, 
        x0     = [np.exp(np.mean(np.log(data))), np.exp(np.std(np.log(data)))], 
        args   = (cutoffs, data), 
        bounds = [cutoffs, (0.0, np.inf)], 
        method = 'Nelder-Mead'
    )
    fun_cdf = lambda x: log_normal_cdf(x, *cutoffs, *optimum.x)
    test = cramervonmises(data, fun_cdf)
    return optimum, fun_cdf, test
