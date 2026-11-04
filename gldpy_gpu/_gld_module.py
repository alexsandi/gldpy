"""GLD module."""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize, special, stats

# y = exp(-700) ~ 1e-304, just above where float64 underflows.
_LOGIT_BOUND = 700.0
# Below this |lambda| the FMKL/VSL moment sums, which divide by lambda**k, lose too many digits.
_MOMENT_SERIES_MIN_LAMBDA = 0.02


def _pow_minus_one(log_u, lam):
    """u**lam - 1 from log(u). RS uses y**a - s**b = (y**a - 1) - (s**b - 1), which avoids subtracting two numbers near 1."""
    return 0.0 if lam == 0 else np.expm1(lam * log_u)


def _box_cox(log_u, lam):
    """(u**lam - 1) / lam from log(u), without cancellation as lam -> 0."""
    return log_u if lam == 0 else np.expm1(lam * log_u) / lam


def _quantile_from_logs(log_y, log_s, param, param_type):
    """Q from log(y) and log(s), where s = 1 - y, so neither tail ever forms `1 - y`."""
    p1, p2, p3, p4 = param
    with np.errstate(over="ignore"):
        if param_type == "RS":
            return p1 + (_pow_minus_one(log_y, p3) - _pow_minus_one(log_s, p4)) / p2
        if param_type == "FMKL":
            return p1 + (_box_cox(log_y, p3) - _box_cox(log_s, p4)) / p2
        # delta = 0 or 1 removes a term entirely, also where its box-cox value is infinite.
        lower = 0.0 if p3 == 1 else (1.0 - p3) * _box_cox(log_y, p4)
        upper = 0.0 if p3 == 0 else p3 * _box_cox(log_s, p4)
        return p1 + p2 * (lower - upper)


def _log_density_from_logs(log_y, log_s, param, param_type):
    """log f(y) = -log Q'(y), factoring out the larger term of Q' so nothing overflows."""
    p1, p2, p3, p4 = param
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if param_type == "VSL":
            c3, c4, e3, e4, log_scale = 1.0 - p3, p3, p4 - 1.0, p4 - 1.0, -np.log(p2)
        elif param_type == "FMKL":
            c3, c4, e3, e4, log_scale = 1.0, 1.0, p3 - 1.0, p4 - 1.0, np.log(p2)
        else:
            c3, c4, e3, e4, log_scale = p3, p4, p3 - 1.0, p4 - 1.0, np.log(abs(p2))
        a = -np.inf if c3 == 0 else np.log(abs(c3)) + (0.0 if e3 == 0 else e3 * log_y)
        b = -np.inf if c4 == 0 else np.log(abs(c4)) + (0.0 if e4 == 0 else e4 * log_s)
        m = np.maximum(a, b)
        ok = np.isfinite(m)  # at y = 0 or 1 a term is infinite and the larger one alone decides
        a, b, m0 = np.where(ok, a, 0.0), np.where(ok, b, 0.0), np.where(ok, m, 0.0)
        inner = np.log(np.abs(np.sign(c3) * np.exp(a - m0) + np.sign(c4) * np.exp(b - m0)))
        return log_scale - m - np.where(ok, inner, 0.0)


def _density_from_y(y, param, param_type):
    """f(y) = scale / |c3 y**e3 + c4 (1 - y)**e4|. A term with a zero coefficient is left out, also at y = 0 or 1."""
    p1, p2, p3, p4 = param
    if param_type == "VSL":
        c3, c4, e3, e4, scale = 1.0 - p3, p3, p4 - 1.0, p4 - 1.0, 1.0 / p2
    elif param_type == "FMKL":
        c3, c4, e3, e4, scale = 1.0, 1.0, p3 - 1.0, p4 - 1.0, p2
    else:
        c3, c4, e3, e4, scale = p3, p4, p3 - 1.0, p4 - 1.0, abs(p2)
    with np.errstate(divide="ignore", over="ignore"):
        term3 = 0.0 if c3 == 0 else c3 * y**e3
        term4 = 0.0 if c4 == 0 else c4 * (1.0 - y) ** e4
        return scale / np.abs(term3 + term4)


def _fmkl_lm_terms(l):
    """FMKL L-moment terms in closed form: (-1/(1+l) + 2/(2+l)) / l = 1/((1+l)(2+l)), and so on.

    The 1/l of the textbook sums cancels, so these stay finite and accurate at l = 0.
    """
    g2 = 1 / ((1 + l) * (2 + l))
    g3 = (l - 1) * g2 / (3 + l)
    g4 = (l - 1) * (l - 2) * g2 / ((3 + l) * (4 + l))
    return g2, g3, g4


class GLD:
    r"""Univariate Generalized Lambda Distribution class.
    
    GLD is flexible family of continuous probability distributions with wide variety of shapes.
    GLD has 4 parameters and defined by quantile function. Probability density function and cumulative distribution function
    are not available in closed form and can be calculated only with the help of numerical methods.
    
    This tool implements three different parameterization types of GLD: 'RS' (introduced by Ramberg and Schmeiser, 1974),
    'FMKL' (introduced by Freimer, Mudholkar, Kollia and Lin, 1988) and 'VSL' (introduced by van Staden and Loots, 2009).
    It provides methods for calculating different characteristics of GLD, parameter estimating, generating random variables and so on.
    
    Attributes:
    ----------
    param_type : str
        Parameterization type of Generalized Lambda Distributions, should be 'RS', 'FMKL' or 'VSL'.
        
    Notes:
    -----
        Different parameterization types of GLD are not equivalent and specify similar but deifferent distribution families,
        there is no one-to-one correspondence between their parameters.
        
        GLD of 'RS' type is characterized by quantile function :math:`Q(y)` and  density quantile function :math:`f(y)`:
            
        .. math::            
            Q(y) = \lambda_1 + \frac{y^{\lambda_3} - (1-y)^{\lambda_4}}{\lambda_2},
        
        .. math::     
            f(y) = \frac{\lambda_2}{\lambda_3 y^{\lambda_3-1} + \lambda_4 (1-y)^{\lambda_4-1}},
            
        where :math:`\lambda_1` - location parameter, :math:`\lambda_2` - inverse scale parameter,
        :math:`\lambda_3, \lambda_4` - shape parameters.
        
        GLD of 'RS' type is defined only for certain values of the shape parameters which provide 
        non-negative density function and there are a complex series of rules determining which parameters
        specify a valid statistical distribution.
        
        'FMKL' parameterization removes this restrictions. GLD of 'FMKL' type is defined for all values of 
        shape parameters and described by following quantile function :math:`Q(y)` and  density quantile function :math:`f(y)`:
        
        .. math::            
            Q(y) = \lambda_1 + \frac{(y^{\lambda_3}-1)/\lambda_3 - ((1-y)^{\lambda_4}-1)/\lambda_4}{\lambda_2},
        
        .. math::     
            f(y) = \frac{\lambda_2}{y^{\lambda_3-1} + (1-y)^{\lambda_4-1}}.       
        
        'VSL' parameterization was introduced for simple parameter estimating in closed form using L-moments. Its quantile function :math:`Q(y)` and  density quantile function :math:`f(y)` are:
         
        .. math::            
            Q(y) = \alpha + \beta \Big((1-\delta)\frac{y^\lambda - 1}{\lambda} - \delta\frac{(1-y)^\lambda - 1}{\lambda}\Big),
        
        .. math::     
            f(y) = \frac{1}{\beta ((1-\delta)y^{\lambda-1}+\delta(1-y)^{\lambda-1})},
            
        where parameters have a different designation: :math:`\alpha` - location parameter, :math:`\beta` - scale parameter, 
        :math:`\delta` - skewness parameter (should be in the interval [0,1]), :math:`\lambda` - shape parameter.
        
       
       
    References:
    ----------
    .. [1] Ramberg, J.S., & Schmeiser, B.W. 1974. An approximate method for generating asymmetric random variables. 
        Communications of the ACM, 17(2), 78–82
    .. [2] Freimer, M., Kollia, G., Mudholkar, G.S., & Lin, C.T. 1988. A study of the
        generalized Tukey lambda family. Communications in Statistics-Theory and Methods, 17, 3547–3567.
    .. [3] Van Staden, Paul J., & M.T. Loots. 2009. Method of L-moment estimation for generalized lambda distribution.
        Third Annual ASEARC Conference. Newcastle, Australia.

    """
  
    def __init__(self, param_type):
        """Create a new GLD with given parameterization type.
        
        Parameters
        ----------
        param_type : str
            Parameterization type. Should be 'RS','FMKL' or 'VSL'.
        Raises
        ------
        ValueError
            If param_type is not one of 'RS','FMKL' or 'VSL'.

        """       
        if param_type not in ['RS','FMKL','VSL']:
            raise ValueError('Unknown parameterisation \'%s\' . Use \'RS\',\'FMKL\' or \'VSL\'' %param_type)
        else:
            self.param_type = param_type

    def check_param(self,param):
        """Check if parameters specify a valid distribution with non-negative density function.
        
        Parameters
        ----------
        param : array-like
            Parameters of GLD
        Raises
        ------
        ValueError
            If number of parameters is not equal to 4.
        Returns
        -------
        bool
            True for valid parameters and False for invalid.       
            
        """        
        if len(param)!=4:
            raise ValueError('GLD has 4 parameters')            
        if not np.isfinite(param).all():
            return False
        else:
            if self.param_type == 'RS':
                r1 = (param[1]<0) and (param[2]<=-1) and (param[3]>=1)
                r2 = (param[1]<0) and (param[2]>=1) and (param[3]<=-1)
                r3 = (param[1]>0) and (param[2]>=0) and (param[3]>=0) and (param[2]!=0 or param[3]!=0)
                r4 = (param[1]<0) and (param[2]<=0) and (param[3]<=0) and (param[2]!=0 or param[3]!=0)
                r5 = (param[1]<0) and (param[2]<=0 and param[2]>=-1) and (param[3]>=1)
                r6 = (param[1]<0) and (param[2]>=1) and (param[3]>=-1 and param[3]<=0)
                if r5:
                    r5 = r5 and (1-param[2])**(1-param[2])*(param[3]-1)**(param[3]-1)/(param[3] - param[2])**(param[3]- param[2])<=-param[2]/param[3]
                if r6:
                    r6 = r6 and (1-param[3])**(1-param[3])*(param[2]-1)**(param[2]-1)/(param[2] - param[3])**(param[2]- param[3])<=-param[3]/param[2]
                return r1 or r2 or r3 or r4 or r5 or r6
            if self.param_type == 'FMKL':
                return param[1]>0  
            if self.param_type == 'VSL':
                return np.logical_and(param[1]>0, np.logical_and(param[2]>=0, param[2]<=1)) 
            
    def Q(self, y, param):
        """Calculate quantile function of GLD at `y` for given parameters.

        Parameters
        ----------
        y : array-like
            Lower tail probability, must be between 0 and 1.
        param : array-like
            Parameters of GLD.
        Raises
        ------
        ValueError
            If input parameters are not valid.
        Returns
        -------
        array-like
            Value of quantile function evaluated at `y`.
            
        """
        y = np.array(y).astype(float)
        param = np.asarray(param, dtype=float)
        if np.logical_or(y>1, y<0).any(): 
            raise ValueError('y should be in range [0,1]')
        if not self.check_param(param):
            raise ValueError('Parameters are not valid')
        with np.errstate(divide='ignore'):
            return _quantile_from_logs(np.log(y), np.log1p(-y), param, self.param_type)

    def ISF(self, s, param):
        """Calculate quantile function at upper tail probability `s`, that is Q(1 - s).

        Unlike Q(1 - s) it stays accurate for tiny `s`, where `1 - s` would round to 1.

        Parameters
        ----------
        s : array-like
            Upper tail probability, must be between 0 and 1.
        param : array-like
            Parameters of GLD.
        Raises
        ------
        ValueError
            If input parameters are not valid.
        Returns
        -------
        array-like
            Value of quantile function evaluated at `1 - s`.

        """
        s = np.array(s).astype(float)
        param = np.asarray(param, dtype=float)
        if np.logical_or(s>1, s<0).any():
            raise ValueError('s should be in range [0,1]')
        if not self.check_param(param):
            raise ValueError('Parameters are not valid')
        with np.errstate(divide='ignore'):
            return _quantile_from_logs(np.log1p(-s), np.log(s), param, self.param_type)

    def PDF_Q(self, y, param):
        """Calculate density quantile function of GLD at `y` for given parameters.
 
        Parameters
        ----------
        y : array-like
            Lower tail probability, must be between 0 and 1.
        param : array-like
            Parameters of GLD.
        Raises
        ------
        ValueError
            If input parameters are not valid.
        Returns
        -------
        array-like
            Value of density quantile function evaluated at `y`.

        """
        y = np.array(y).astype(float)
        if np.logical_or(y>1, y<0).any():
            raise ValueError('y should be in range [0,1]')
        if not self.check_param(param):
            raise ValueError('Parameters are not valid')        
        return _density_from_y(y, np.asarray(param, dtype=float), self.param_type)
       
    def _logit_cdf(self, x, param, xtol):
        """Solve Q(sigmoid(t)) = x for each element.

        Working on t = logit(y) makes `xtol` a relative tolerance on the smaller of y and 1 - y.
        Returns -inf/+inf at or beyond the support, NaN for NaN input, and saturates at
        t = +-_LOGIT_BOUND, i.e. tail probabilities below ~1e-304.
        """
        if not self.check_param(param):
            raise ValueError('Parameters are not valid')
        param = np.asarray(param, dtype=float)
        x = np.array(x, dtype=float)
        if x.ndim == 0:
            x = x.reshape(1)
        a, b = self.supp(param)
        t = np.full(x.shape, np.nan)
        t[x <= a] = -np.inf
        t[x >= b] = np.inf

        def f(t_i, x_i):
            """Auxiliary function for root finding."""
            return _quantile_from_logs(-np.logaddexp(0.0, -t_i), -np.logaddexp(0.0, t_i), param, self.param_type) - x_i

        for i in np.ndindex(x.shape):
            if np.isnan(t[i]) and not np.isnan(x[i]):
                if f(-_LOGIT_BOUND, x[i]) >= 0:
                    t[i] = -_LOGIT_BOUND
                elif f(_LOGIT_BOUND, x[i]) <= 0:
                    t[i] = _LOGIT_BOUND
                else:
                    t[i] = optimize.brentq(f, -_LOGIT_BOUND, _LOGIT_BOUND, args=(x[i],), xtol=xtol, maxiter=500)
        return t

    def CDF_num(self, x, param, xtol = 1e-08):
        """Calculate cumulative distribution function of GLD numerically at `x` for given parameters.

        Parameters
        ----------
        x : array-like
            Argument of CDF.
        param : array-like
            Parameters of GLD.
        xtol : float, optional
            Relative accuracy of the smaller tail probability. The default is 1e-08.
        Raises
        ------
        ValueError
            If input parameters are not valid.
        Returns
        -------
        array-like
            Value of cumulative distribution function evaluated at `x`, with the shape of `x`
            (a scalar gives an array of length 1).

        """
        return np.exp(-np.logaddexp(0.0, -self._logit_cdf(x, param, xtol)))

    def SF_num(self, x, param, xtol = 1e-08):
        """Calculate survival function P(X > x) numerically at `x` for given parameters.

        Use it for upper tails: `1 - CDF_num` loses everything below machine epsilon.

        Parameters
        ----------
        x : array-like
            Argument of the survival function.
        param : array-like
            Parameters of GLD.
        xtol : float, optional
            Relative accuracy of the smaller tail probability. The default is 1e-08.
        Raises
        ------
        ValueError
            If input parameters are not valid.
        Returns
        -------
        array-like
            Value of the survival function evaluated at `x`, with the shape of `x`
            (a scalar gives an array of length 1).

        """
        return np.exp(-np.logaddexp(0.0, self._logit_cdf(x, param, xtol)))

    def log_PDF_num(self, x, param, xtol = 1e-08):
        """Calculate the logarithm of the probability density function numerically at `x`.

        Parameters
        ----------
        x : array-like
            Argument of PDF.
        param : array-like
            Parameters of GLD.
        xtol : float, optional
            Relative accuracy of the smaller tail probability. The default is 1e-08.
        Raises
        ------
        ValueError
            If input parameters are not valid.
        Returns
        -------
        array-like
            Log density evaluated at `x`, -inf outside the support, with the shape of `x`
            (a scalar gives an array of length 1).

        """
        t = self._logit_cdf(x, param, xtol)
        a, b = self.supp(param)
        x = np.array(x, dtype=float).reshape(t.shape)
        ans = _log_density_from_logs(-np.logaddexp(0.0, -t), -np.logaddexp(0.0, t), np.asarray(param, dtype=float), self.param_type)
        return np.where(np.logical_or(x<a, x>b), -np.inf, ans)

    def PDF_num(self, x, param, xtol = 1e-08):
        """Calculate probability density function of GLD numerically at `x` for given parameters.

        Parameters
        ----------
        x : array-like
            Argument of PDF.
        param : array-like
            Parameters of GLD.
        xtol : float, optional
            Relative accuracy of the smaller tail probability. The default is 1e-08.
        Raises
        ------
        ValueError
            If input parameters are not valid.
        Returns
        -------
        array-like
            Value of probability density function evaluated at `x`, with the shape of `x`
            (a scalar gives an array of length 1).

        """
        return np.exp(self.log_PDF_num(x, param, xtol))

    def supp(self,param):
        """Return support of GLD for given parameters.

        Parameters
        ----------
        param : array-like
            Parameters of GLD.
        Raises
        ------
        ValueError
            If input parameters are not valid.
        Returns
        -------
        array-like
            Support of GLD.

        """
        if not self.check_param(param):
            raise ValueError('Parameters are not valid')
        return self.Q(0,param), self.Q(1,param)
    
    def rand(self, param, size = 1, random_state = None):
        """Generate random variables of GLD.

        Parameters
        ----------
        param : array-like
            Parameters of GLD.
        size : int, optional
            Number of random variables. The default is 1.
        random_state : None or int, optional
            The seed of the pseudo random number generator. The default is None.
        Returns
        -------
        array-like
            Sample of GLD random variables of given size.

        """
        generator = np.random if random_state is None else np.random.RandomState(random_state)
        alpha = generator.random_sample(size)
        return self.Q(alpha,param)        
    
    def correct_supp(self,  data, param, eps = 0.0001):
        """Correct support of GLD due to data.
        
        In certain cases some data points can be outside of finite support of GLD.
        This method corrects parameters of location and scale to fit support to data.
        It is used as a component of some parameter estimation methods.

        Parameters
        ----------
        data : array-like
            Input data.
        param : array-like
            Parameters of GLD.
        eps : float, optional
            Parameter of support fitting. Tail probability of minimum and maximum data points. The default is 0.0001.
         
        Returns
        -------
        array-like
            Corrected parameters of GLD.
            
        """
        data = data.ravel()
        param = np.array(param, dtype=float)
        q_low, q_high = self.Q(eps,param), self.Q(1-eps,param)
        A = np.min([np.min(data), q_low])
        B = np.max([np.max(data), q_high])
        # Q is affine in location and scale, so the two end conditions have an exact solution.
        k = (B - A)/(q_high - q_low)
        param[1] = param[1]*k if self.param_type == 'VSL' else param[1]/k
        param[0] = A - (q_low - param[0])*k
        return param

    def GoF_Q_metric(self,data,param):
        """Calculate Goodness-of-Fit metric based on discrepancy between empirical and theoretical quantile functions.
        
        It can be used for simple comparison of different fitted distributions.

        Parameters
        ----------
        data : array-like
            Input data.
        param : array-like
            Parameters of GLD.
         
        Returns
        -------
        float
            Mean square deviation of empirical and theoretical quantiles.

        """        
        data = data.ravel()
        return np.mean((np.sort(data) - self.Q((np.arange(len(data))+0.5)/len(data),param))**2)

    def GoF_tests(self,param, data, bins_gof = 8):
        """Perform two Goodness-of_Fit tests: Kolmogorov-Smirnov test and one-way chi-square test from scipy.stats.

        Parameters
        ----------
        param : array-like
            Parameters of GLD.
        data : array-like
            Input data.
        bins_gof : int, optional
            Number of bins for chi-square test. The default is 8.
         
        Returns
        -------
        scipy.stats.stats.KstestResult
            Result of Kolmogorov-Smirnov test including statistic and p-value.
        scipy.stats.stats.Power_divergenceResult
            Result of chi-square test including statistic and p-value.
            
        """
        def cdf(x):
            """Auxiliary function for GoF test."""            
            return self.CDF_num(x,param)
        ks = stats.kstest(data, cdf)
        # Points outside the support go to the end bins, or the counts would not sum to len(data).
        a,b = self.supp(param)
        chi2 = stats.chisquare(np.histogram(np.clip(data,a,b),self.Q(np.linspace(0, 1, bins_gof + 1),param))[0],[len(data)/bins_gof]*bins_gof )
        return ks, chi2

    def plot_cdf(self, param_list, data = None, ymin = 0.01, ymax = 0.99, n_points = 100, names = None, color_emp = 'lightgrey', colors = None):
        """Plot cumulative distribution functions of GLD.
        
        This allows to compare GLD cumulative distribution functions with different parameters.
        Also it is possible to add empirical CDF on the plot.

        Parameters
        ----------
        param_list : array-like or list of array-like
            List of GLD parameters for plotting. 
        data : array-like, optional
            If not None empirical CDF estimated by data will be added to the plot.  The default is None.
        ymin : float, optional
            Minimal lower tail probability for plotting. The default is 0.01.
        ymax : float, optional
            Maximal lower tail probability for plotting. The default is 0.99.
        n_points : int, optional
            Number of points for plotting. The default is 100.
        names : list of str, optional
            Names of labels for the legend. Length of the list should be equal to the length of param_list.  
        color_emp : str, optional
            Line color of empirical CDF. It's ignored if data is None. The default is 'lightgrey'.
        colors : list of str, optional
            Line colors of CDFs. Length of the list should be equal to the length of param_list.
        plot_fitting(self, data, param, bins=None)
        
        """
        param_list = np.array(param_list)        
        if param_list.ndim==1:
            param_list = param_list.reshape(1,-1)
        if names is None:
            names = [str(x) for x in param_list]
        if colors is None:
            colors = [None]*len(param_list)
        plt.figure()
        plt.grid()
        if  not (data is None):
            data = data.ravel()
            plt.plot(np.sort(data), np.arange(len(data))/len(data),color = color_emp,lw = 2)
            names = np.hstack(['empirical data', names ])
        y = np.linspace(ymin,ymax,n_points)
        for i in range(param_list.shape[0]):
            param = param_list[i]
            plt.plot(self.Q(y,param), y, color = colors[i])
        plt.ylim(ymin = 0)
        plt.legend(names,bbox_to_anchor=(1.0,  1.0 ))
        plt.title('CDF')
        
    def plot_pdf(self, param_list, data = None, ymin = 0.01, ymax = 0.99, n_points = 100,  bins = None, names = None, color_emp = 'lightgrey', colors = None):
        """Plot probability density functions of GLD.
        
        This allows to compare GLD probability density functions with different parameters.
        Also it is possible to add data histogram on the plot.

        Parameters
        ----------
        param_list : array-like or list of array-like
            List of GLD parameters for plotting. 
        data : array-like, optional
            If not None empirical CDF estimated by data will be added to the plot.  
        ymin : float, optional
            Minimal lower tail probability for plotting. The default is 0.01.
        ymax : float, optional
            Maximal lower tail probability for plotting. The default is 0.99.
        n_points : int, optional
            Number of points for plotting. The default is 100.
        bins : int, optional
            Number of bins for histogram. It's ignored if data is None. 
        names : list of str, optional
            Names of labels for the legend. Length of the list should be equal to the length of param_list.  The default is None.
        color_emp : str, optional
            Color of the histogram. It's ignored if data is None. The default is 'lightgrey'.
        colors : list of str, optional
            Line colors of PDFs. Length of the list should be equal to the length of param_list.
            
        """
        param_list = np.array(param_list)
        if param_list.ndim==1:
            param_list = param_list.reshape(1,-1)
        if names is None:
            names = [str(x) for x in param_list]
        plt.figure()
        plt.grid()
        pdf_max = 0
        if not data is None:
            data = data.ravel()
            p = plt.hist(data, bins = bins, color = color_emp, density = True)
            pdf_max = np.max(p[0])
        if colors is None:
            colors = [None]*len(param_list)
        y = np.linspace(ymin,ymax,n_points)
        
        for i in range(param_list.shape[0]):
            param = param_list[i]
            plt.plot(self.Q(y,param), self.PDF_Q(y,param),color = colors[i])
            pdf_max = np.max([pdf_max,np.max(self.PDF_Q(y,param))])
        plt.ylim(ymin = 0,ymax = pdf_max * 1.05)
        plt.legend(names,bbox_to_anchor=(1.0,  1.0 ))
        plt.title('PDF')                
        
    def plot_fitting(self,data,param, bins = None):
        """Construct plots for comparing fitted GLD with data.
        
        It allows to compare data histogram and PDF of fitted GLD on the one plot,
        empirical and theoretical CDFs on the second plot and
        theoretical and empirical quantiles plotted against each other on the third plot.

        Parameters
        ----------
        data : array-like
            Input data.
        param : array-like
            Parameters of GLD.
        bins : int, optional
            Number of bins for histogram.
            
        """
        data = data.ravel()
        fig,ax = plt.subplots(1,3,figsize = (15,3))
        ax[0].hist(data,bins = bins,density = True,color = 'skyblue')
        y = np.linspace(0.001,0.999,100)
        ax[0].plot(self.Q(y,param),self.PDF_Q(y,param),lw = 2,color = 'r')
        ax[0].set_title('PDF')
        ax[0].grid()
        ax[1].plot(np.sort(data), np.arange(len(data))/len(data))
        ax[1].plot(self.Q(y,param), y)
        ax[1].grid()
        ax[1].set_title('CDF')
        x = np.sort(data)
        y = (np.arange(len(data))+0.5)/len(data)        
        ax[2].plot(self.Q(y,param), x,'bo',ms = 3)
        m1 = np.min([x,self.Q(y,param)])
        m2 = np.max([x,self.Q(y,param)])
        ax[2].plot([m1,m2], [m1,m2],'r')
        ax[2].grid()
        ax[2].set_title('Q-Q-plot') 
    
    def __sum_Ez(self,k,p3,p4):
        """Auxiliary function for moments calculation."""        
        s = 0
        p3 = np.array(p3)
        p4 = np.array(p4)
        smallest = {'RS': np.inf, 'FMKL': min(abs(p3), abs(p4)), 'VSL': abs(p4)}[self.param_type]
        if smallest < _MOMENT_SERIES_MIN_LAMBDA:
            # The sums below divide by lambda**k. Integrate Q(y)**k directly, in both halves
            # because the integrand has a log singularity at each end.
            standard = np.array([0.0, 1.0, float(p3), float(p4)])
            def integrand(y):
                """Auxiliary function for moments calculation."""
                return _quantile_from_logs(np.log(y), np.log1p(-y), standard, self.param_type)**k
            return (integrate.quad(integrand, 0, 0.5, limit=200, epsabs=1e-11, epsrel=1e-11)[0]
                    + integrate.quad(integrand, 0.5, 1, limit=200, epsabs=1e-11, epsrel=1e-11)[0])
        if self.param_type == 'RS':
            for i in range(0,k+1):
                s+=special.binom(k,i)*(-1)**i *special.beta(p3*(k-i)+1, p4*i+1)
        if self.param_type == 'FMKL':
            for i in range(0,k+1):
                for j in range(0, k-i+1):
                    s+=(p3-p4)**i/(p3*p4)**k * special.binom(k,i)*special.binom(k-i,j)*(-1)**j*p4**(k-i-j)*p3**j*special.beta(p3*(k-i-j)+1,p4*j+1)
        if self.param_type=='VSL':
            for i in range(0,k+1):
                for j in range(0, k-i+1):
                    s+=(2*p3-1)**i/p4**k*special.binom(k,i)*special.binom(k-i,j)*(-1)**j*(1-p3)**(k-i-j)*p3**j*special.beta(p4*(k-i-j)+1,p4*j+1) 
        return s  

    def mean(self, param):
        """Calculate mean of the GLD for given parameters.

        Parameters
        ----------
        param : array-like
            Parameters of GLD.
        Raises
        ------
        ValueError
            If input parameters are not valid.
        Returns
        -------
        float
            Mean of GLD.

        """
        if not self.check_param(param):
            raise ValueError('Parameters are not valid')
        if param[2]>-1 and param[3]>-1:               
            A = self.__sum_Ez(1,param[2], param[3])
            L = 1/param[1] if self.param_type=='VSL' else param[1]
            return A/L + param[0]
        else:            
            return np.nan
            
    def var(self, param):
        """Calculate variance of the GLD for given parameters.

        Parameters
        ----------
        param : array-like
            Parameters of GLD.
        Raises
        ------
        ValueError
            If input parameters are not valid.
        Returns
        -------
        float
            Variance of GLD.

        """
        if not self.check_param(param):
            raise ValueError('Parameters are not valid')             
        if param[2]>-1/2 and param[3]>-1/2:
            A = self.__sum_Ez(1,param[2], param[3])
            B = self.__sum_Ez(2,param[2], param[3])
            L = 1/param[1] if self.param_type=='VSL' else param[1]
            return (B-A**2)/L**2
        else:
            return np.nan
        
    def std(self, param):
        """Calculate standard deviation of the GLD for given parameters.

        Parameters
        ----------
        param : array-like
            Parameters of GLD.
        Raises
        ------
        ValueError
            If input parameters are not valid.
        Returns
        -------
        float
            Standard deviation of GLD.

        """
        return np.sqrt(self.var(param))

    def skewness(self, param):    
        """Calculate skewness of the GLD for given parameters.

        Parameters
        ----------
        param : array-like
            Parameters of GLD.
        Raises
        ------
        ValueError
            If input parameters are not valid.
        Returns
        -------
        float
            Skewness of GLD.

        """
        if not self.check_param(param):
            raise ValueError('Parameters are not valid')            
        if param[2]>-1/3 and param[3]>-1/3:
            A = self.__sum_Ez(1,param[2], param[3])
            B = self.__sum_Ez(2,param[2], param[3])
            C = self.__sum_Ez(3,param[2], param[3])
            L = 1/param[1] if self.param_type=='VSL' else param[1]
            a2 = (B-A**2)/L**2
            return (C-3*A*B+2*A**3)/L**3/a2**1.5
        else:
            return np.nan
        
    def kurtosis(self, param):
        """Calculate kurtosis of the GLD for given parameters.

        Parameters
        ----------
        param : array-like
            Parameters of GLD.
        Raises
        ------
        ValueError
            If input parameters are not valid.
        Returns
        -------
        float
            Kurtosis of GLD.
            
        """
        if not self.check_param(param):
            raise ValueError('Parameters are not valid')            
        if param[2]>-1/4 and param[3]>-1/4:
            A = self.__sum_Ez(1,param[2], param[3])
            B = self.__sum_Ez(2,param[2], param[3])
            C = self.__sum_Ez(3,param[2], param[3])
            D = self.__sum_Ez(4,param[2], param[3])
            L = 1/param[1] if self.param_type=='VSL' else param[1]            
            a2 = (B-A**2)/L**2           
            return (D-4*A*C+6*A**2*B-3*A**4)/L**4/a2**2
        else:
            return np.nan
    
    def median(self,param):
        """Calculate median of the GLD for given parameters.

        Parameters
        ----------
        param : array-like
            Parameters of GLD.
        Raises
        ------
        ValueError
            If input parameters are not valid.
        Returns
        -------
        float
            Median of GLD.

        """
        return self.Q(0.5,param)    
        
    def fit_MM(self,data, initial_guess, xtol=0.0001, maxiter=None, maxfun=None , disp_optimizer=True,   disp_fit = True, bins_hist = None, test_gof = True, bins_gof = 8):
        """Fit GLD to data using method of moments.
        
        It estimates parameters of GLD by setting first four sample moments equal to their GLD counterparts.
        Resulting system of equations are solved using numerical methods for given initial guess.
        There are some restrictions of this method related to existence of moments and computational difficulties.

        Parameters
        ----------
        data : array-like
            Input data.
        initial_guess : array-like
            Initial guess for third and fourth parameters.
        xtol : float, optional
            Absolute error for optimization procedure. The default is 0.0001.
        maxiter : int, optional
            Maximum number of iterations for optimization procedure. 
        maxfun : int, optional
            Maximum number of function evaluations for optimization procedure. 
        disp_optimizer : bool, optional
            Set True to display information about optimization procedure. The default is True.
        disp_fit : bool, optional
            Set True to display information about fitting. The default is True.
        bins_hist : int, optional
            Number of bins for histogram. It's ignored if disp_fit is False.
        test_gof : bool, optional
            Set True to perform Goodness-of-Fit tests and print results. It's ignored if disp_fit is False. The default is True.
        bins_gof : int, optional
            Number of bins for chi-square test. It's ignored if test_gof is False. The default is 8.
        Raises
        ------
        ValueError
            If length of initial guess is incorrect.
        Returns
        -------
        array-like
            Fitted parameters of GLD.
            
        References:
        ----------
        .. [1] Karian, Z.A., Dudewicz, E.J. 2000. Fitting statistical distributions: the generalized
            lambda distribution and generalized bootstrap methods. Chapman and Hall/CRC.
        
        """
        initial_guess = np.array(initial_guess)
        data = data.ravel()
        def sample_moments(data):
            """Calculate first four sample moments."""            
            a1 = np.mean(data)
            a2 = np.mean((data - a1)**2)
            a3 = np.mean((data - a1)**3)/a2**1.5
            a4 = np.mean((data - a1)**4)/a2**2
            return a1,a2,a3,a4        
        def moments( param):        
            """Calculate first four GLD moments."""            
            A = self.__sum_Ez(1,param[2], param[3])
            B = self.__sum_Ez(2,param[2], param[3])
            C = self.__sum_Ez(3,param[2], param[3])
            D = self.__sum_Ez(4,param[2], param[3])
            L = 1/param[1] if self.param_type=='VSL' else param[1]
            a1 = A/L + param[0]
            a2 = (B-A**2)/L**2
            a3 = (C-3*A*B+2*A**3)/L**3/a2**1.5
            a4 = (D-4*A*C+6*A**2*B-3*A**4)/L**4/a2**2
            return a1,a2,a3,a4
        
        def fun_VSL(x):
            """Auxiliary function for optimization."""            
            if x[0]<0 or x[0] >1 or x[1]<-0.25:
                return np.inf
            A = self.__sum_Ez(1,x[0],x[1])
            B = self.__sum_Ez(2,x[0],x[1])
            C = self.__sum_Ez(3,x[0],x[1])
            D = self.__sum_Ez(4,x[0],x[1])
            return np.max([np.abs((C-3*A*B+2*A**3)/(B-A**2)**1.5 - a3),   np.abs( (D-4*A*C+6*A**2*B-3*A**4)/(B-A**2)**2 - a4)])

        def fun_RS_FMKL(x):
            """Auxiliary function for optimization."""            
            if x[0] <-0.25 or x[1]<-0.25:
                return np.inf
            A = self.__sum_Ez(1,x[0],x[1])
            B = self.__sum_Ez(2,x[0],x[1])
            C = self.__sum_Ez(3,x[0],x[1])
            D = self.__sum_Ez(4,x[0],x[1])
            return np.max([np.abs((C-3*A*B+2*A**3)/(B-A**2)**1.5 - a3),   np.abs( (D-4*A*C+6*A**2*B-3*A**4)/(B-A**2)**2 - a4)])
        fun_opt = fun_VSL if self.param_type=='VSL' else fun_RS_FMKL
        if initial_guess.ndim==0 or len(initial_guess)!=2:
                raise ValueError('Specify initial guess for two parameters')   
        a1,a2,a3,a4 = sample_moments(data)
        [p3,p4] = optimize.fmin(fun_opt,initial_guess,maxfun = maxfun,xtol=xtol, maxiter=maxiter, disp = disp_optimizer)
        A = self.__sum_Ez(1,p3,p4)
        B = self.__sum_Ez(2,p3,p4)
        C = self.__sum_Ez(3,p3,p4)
        D = self.__sum_Ez(4,p3,p4)
        p2 = (((B-A**2)/a2)**0.5)**(-1 if self.param_type=='VSL' else 1)
        p1 = a1 - A/(p2**(-1 if self.param_type=='VSL' else 1))
        param = [p1,p2,p3,p4]
        if self.param_type=='RS' and not self.check_param(param):
            p3, p4 = p4,p3
            p2 = p2* (-1)
            p1 = a1 + A/(p2)
            param = [p1,p2,p3,p4]
        if disp_fit:  
            print('')
            print('Sample moments: ', sample_moments(data))
            print('Fitted moments: ', moments(param))
            print('')
            print('Parameters: ', param)           
            if not self.check_param(param):
                print('')
                print('Parameters are not valid. Try another initial guess.')
            else:                
                if test_gof:
                    ks, chi2 = self.GoF_tests(param, data, bins_gof)
                    print('')
                    print('Goodness-of-Fit')
                    print(ks)
                    print(chi2)
                self.plot_fitting(data,param,bins = bins_hist)
        return np.array(param)

     
    def fit_PM(self,data, initial_guess, u = 0.1, xtol=0.0001, maxiter=None, maxfun=None , disp_optimizer=True,   disp_fit = True, bins_hist = None, test_gof = True, bins_gof = 8):
        """Fit GLD to data using method of percentiles.
        
        It estimates parameters of GLD by setting four percentile-based sample statistics equal to their corresponding GLD statistics.
        To calculate this statistics it's necessary to specify parameter u (number between 0 and 0.25).
        Resulting system of equations are solved using numerical methods for given initial guess.

        Parameters
        ----------
        data : array-like
            Input data.
        initial_guess : array-like
            Initial guess for third and fourth parameters if parameterization type is 'RS' or 'FMKL' 
            and for only fourth parameter if parameterization type is 'VSL'.
        u : float, optional
            Parameter for calculating percentile-based statistics. Arbitrary number between 0 and 0.25. The default is 0.1.
        xtol : float, optional
            Absolute error for optimization procedure. The default is 0.0001.
        maxiter : int, optional
            Maximum number of iterations for optimization procedure. 
        maxfun : int, optional
            Maximum number of function evaluations for optimization procedure. 
        disp_optimizer : bool, optional
            Set True to display information about optimization procedure. The default is True.
        disp_fit : bool, optional
            Set True to display information about fitting. The default is True.
        bins_hist : int, optional
            Number of bins for histogram. It's ignored if disp_fit is False.
        test_gof : bool, optional
            Set True to perform Goodness-of-Fit tests and print results. It's ignored if disp_fit is False. The default is True.
        bins_gof : int, optional
            Number of bins for chi-square test. It's ignored if test_gof is False. The default is 8.
        Raises
        ------
        ValueError
            If length of initial guess is incorrect or parameter u is out of range [0,0.25].
        Returns
        -------
        array-like
            Fitted parameters of GLD.
            
        References:
        ----------
        .. [1] Karian, Z.A., Dudewicz, E.J. 2000. Fitting statistical distributions: the generalized
            lambda distribution and generalized bootstrap methods. Chapman and Hall/CRC.

        """
        initial_guess = np.array(initial_guess)
        data = data.ravel()
        if u<0 or u>0.25:
            raise ValueError('u should be in interval [0,0.25]')
        def sample_statistics(data, u):
            """Calculate four sample percentile-based statistics."""            
            p1 = np.quantile(data, 0.5)
            p2 = np.quantile(data, 1-u) - np.quantile(data, u)
            p3 = (np.quantile(data, 0.5) - np.quantile(data, u))/(np.quantile(data, 1-u) - np.quantile(data, 0.5))
            p4 = (np.quantile(data, 0.75) - np.quantile(data, 0.25))/p2
            return p1,p2,p3,p4
        a1,a2,a3,a4 = sample_statistics(data,u)
        if self.param_type=='RS':
            def theor_statistics(param,u):
                """Calculate four GLD percentile-based statistics."""                
                [l1,l2,l3,l4] = param
                p1 = l1+(0.5**l3 - 0.5**l4)/l2
                p2 = ((1-u)**l3 - u**l4 - u**l3+(1-u)**l4)/l2
                p3 = (0.5**l3 - 0.5**l4 - u**l3 +(1-u)**l4)/((1-u)**l3 - u**l4 - 0.5**l3 +0.5**l4)
                p4 = (0.75**l3 - 0.25**l4 - 0.25**l3 +0.75**l4)/((1-u)**l3-u**l4 - u**l3+(1-u)**l4)
                return p1,p2,p3,p4
            def fun_opt(x):
                """Auxiliary function for optimization."""                
                l3 = x[0]
                l4 = x[1]
                return np.max([np.abs((0.75**l3 - 0.25**l4 - 0.25**l3 +0.75**l4)/((1-u)**l3-u**l4 - u**l3+(1-u)**l4) - a4),
                               np.abs((0.5**l3 - 0.5**l4 - u**l3 +(1-u)**l4)/((1-u)**l3 - u**l4 - 0.5**l3 +0.5**l4) - a3)])
            if initial_guess.ndim==0 or len(initial_guess)!=2:
                raise ValueError('Specify initial guess for two parameters')  
            [l3,l4] = optimize.fmin(fun_opt,initial_guess,maxfun = maxfun,xtol=xtol, maxiter=maxiter, disp = disp_optimizer)
            l2 = 1/a2*((1-u)**l3-u**l3 + (1-u)**l4 - u**l4)
            l1 = a1 - 1/l2*(0.5**l3  - 0.5**l4)
            param = np.array([l1,l2,l3,l4]).ravel()
            theor_stat = theor_statistics(param,u)    
        if self.param_type == 'FMKL':
            def h(c,l):
                """(c**l - 1)/l, finite at l = 0."""
                return _box_cox(np.log(c),l)
            def ratios(l3,l4,u):
                """Third and fourth percentile-based statistics, which do not depend on location and scale."""
                p3 = (h(0.5,l3) - h(u,l3) + h(1-u,l4) - h(0.5,l4)) / (h(1-u,l3) - h(0.5,l3) + h(0.5,l4) - h(u,l4))
                p4 = (h(0.75,l3) - h(0.25,l3) + h(0.75,l4) - h(0.25,l4)) / (h(1-u,l3) - h(u,l3) + h(1-u,l4) - h(u,l4))
                return p3,p4
            def theor_statistics(param,u):
                """Calculate four GLD percentile-based statistics."""                
                [l1,l2,l3,l4] = param
                p1 = l1+(h(0.5,l3) - h(0.5,l4))/l2
                p2 = (h(1-u,l3) - h(u,l3) + h(1-u,l4) - h(u,l4))/l2
                p3,p4 = ratios(l3,l4,u)
                return p1,p2,p3,p4
            
            def fun_opt(x):                
                """Auxiliary function for optimization."""
                p3,p4 = ratios(x[0],x[1],u)
                return np.max([np.abs(p4 - a4), np.abs(p3 - a3)])
            if initial_guess.ndim==0 or len(initial_guess)!=2:
                raise ValueError('Specify initial guess for two parameters')   
            [l3,l4] = optimize.fmin(fun_opt,initial_guess,maxfun = maxfun,xtol=xtol, maxiter=maxiter, disp = disp_optimizer)
            l2 = 1/a2*(h(1-u,l3) - h(u,l3) + h(1-u,l4) - h(u,l4))
            l1 = a1 - 1/l2*(h(0.5,l3) - h(0.5,l4))
            param = np.array([l1,l2,l3,l4]).ravel()
            theor_stat = theor_statistics(param,u)   
        if self.param_type == 'VSL':
            def h(c,l):
                """(c**l - 1)/l, finite at l = 0."""
                return _box_cox(np.log(c),l)
            def theor_statistics(param,u):
                """Calculate four GLD percentile-based statistics."""                
                [a,b,d,l] = param
                p1 = a+b*h(0.5,l)*(1-2*d)
                p2 = b*(h(1-u,l) - h(u,l))
                p3 = ((1-d)*(h(0.5,l) - h(u,l))+d*(h(1-u,l) - h(0.5,l)))/((1-d)*(h(1-u,l) - h(0.5,l))+d*(h(0.5,l) - h(u,l)))
                p4 = (h(0.75,l) - h(0.25,l))/(h(1-u,l) - h(u,l))
                return p1,p2,p3,p4
            def fun_opt(x):
                """Auxiliary function for optimization."""                
                return np.abs((h(0.75,x[0]) - h(0.25,x[0]))/(h(1-u,x[0]) - h(u,x[0])) - a4)
            if initial_guess.ndim!=0 and len(initial_guess)!=1:
                raise ValueError('Specify initial guess for one parameter')
            l = optimize.fmin(fun_opt,np.atleast_1d(initial_guess),maxfun = maxfun,xtol=xtol, maxiter=maxiter, disp = disp_optimizer)[0]
            d = (a3*(h(1-u,l) - h(0.5,l)) - h(0.5,l) + h(u,l))/(a3+1)/(h(1-u,l) - 2*h(0.5,l) + h(u,l))
            d = np.max([0,np.min([1,d])])
            b = a2/(h(1-u,l) - h(u,l))
            a = a1 - b*h(0.5,l)*(1-2*d)
            param = np.array([a,b,d,l]).ravel()
            theor_stat = theor_statistics(param,u)
        if disp_fit:  
            print('')
            print('Sample statistics: ', sample_statistics(data,u))
            print('Fitted statistics: ', theor_stat)
            print('')
            print('Parameters: ', param)
            if not self.check_param(param):
                print('')
                print('Parameters are not valid. Try another initial guess.')
            else:
                if test_gof:
                    ks, chi2 = self.GoF_tests(param, data, bins_gof)
                    print('')
                    print('Goodness-of-Fit')
                    print(ks)
                    print(chi2)
                self.plot_fitting(data,param,bins = bins_hist)
        return np.array(param)
  
    def fit_LMM(self,data, initial_guess = None,  xtol=0.0001, maxiter=None, maxfun=None , disp_optimizer=True,   disp_fit = True, bins_hist = None, test_gof = True, bins_gof = 8):
        """Fit GLD to data using method of L-moments.
        
        It estimates parameters of GLD by equating four sample L-moments and L-moments ratios and their GLD counterparts.
        L-moments are linear combinations of order statistics analogous to conventional moments.
        
        Resulting system of equations for 'RS' and 'FMKL' parameterizations are solved using numerical methods for given initial guess.
        
        For 'VSL' parameterization there is exact analytical solution of the equations.
        In general case there are two different sets of parameters which give the same values of L-moments.
        The method returns solution which is closest to initial guess.
        If initial_guess is None the best solution is chosen using GLD.GoF_Q_metric.  

        Parameters
        ----------
        data : array-like
            Input data.
        initial_guess : array-like
            Initial guess for third and fourth parameters. It's ignored for 'VSL' parameterization type.
        xtol : float, optional
            Absolute error for optimization procedure. The default is 0.0001. It's ignored for 'VSL' parameterization type.
        maxiter : int, optional
            Maximum number of iterations for optimization procedure. It's ignored for 'VSL' parameterization type.
        maxfun : int, optional
            Maximum number of function evaluations for optimization procedure. It's ignored for 'VSL' parameterization type.
        disp_optimizer : bool, optional
            Set True to display information about optimization procedure. The default is True. It's ignored for 'VSL' parameterization type.
        disp_fit : bool, optional
            Set True to display information about fitting. The default is True.
        bins_hist : int, optional
            Number of bins for histogram. It's ignored if disp_fit is False.
        test_gof : bool, optional
            Set True to perform Goodness-of-Fit tests and print results. It's ignored if disp_fit is False. The default is True.
        bins_gof : int, optional
            Number of bins for chi-square test. It's ignored if test_gof is False. The default is 8.
        Raises
        ------
        ValueError
            If length of initial guess is incorrect.
        Returns
        -------
        array-like
            Fitted parameters of GLD.
            
        References:
        ----------
        .. [1] Karvanen, J. and Nuutinen, A. 2008. Characterizing the generalized lambda distribution by L-moments.
            Computational Statistics & Data Analysis, 52(4):1971–1983.
        .. [2] Van Staden, Paul J., & M.T. Loots. 2009. Method of L-moment estimation for generalized lambda distribution.
            Third Annual ASEARC Conference. Newcastle, Australia.
       
            
        """        
        if not initial_guess is None:
            initial_guess = np.array(initial_guess)
        data = data.ravel()
        def sample_lm(data):
            """Calculate four sample L-moments and L-moment ratios."""            
            x = np.sort(data)
            n = len(data)
            l1 = np.mean(x)
            l2 = np.sum(np.array([2*i-n - 1 for i in range(1,n+1)])*x)/2/special.binom(n,2) 
            l3 = np.sum(np.array([special.binom(i-1,2) - 2*(i-1)*(n-i)+special.binom(n-i,2) for i in range(1,n+1)])*x)/3/special.binom(n,3) 
            l4 = np.sum(np.array([special.binom(i-1,3) - 3*special.binom(i-1,2)*(n-i)+3*(i-1)*special.binom(n-i,2)-special.binom(n-i,3) for i in range(1,n+1)])*x)/4/special.binom(n,4)
            return l1,l2,l3/l2,l4/l2
        a1,a2,a3,a4 = sample_lm(data)
        def lm(param):
            """Calculate four GLD L-moments and L-moment ratios."""            
            def lr(r,param):
                """Auxiliary function for L-moments calculation."""                
                if self.param_type=='VSL':
                    [a,b,d,l] = param
                    s = 0
                    for k in range(r):
                        s+=(-1)**(r-k-1)*special.binom(r-1,k)*special.binom(r+k-1,k)*((1-d-(-1)**(r-1)*d)/l/(l+k+1))
                    if r==1:
                        s = s*b+a+b*(2*d-1)/l
                    else:
                        s = s*b
                    return s
                if self.param_type=='RS':
                    [l1,l2,l3,l4] = param
                    s = 0
                    for k in range(r):
                        s+=(-1)**(r-k-1)*special.binom(r-1,k)*special.binom(r+k-1,k)*(1/(l3+k+1) - (-1)**(r-1)/(l4+k+1))
                    if r==1:
                        s = s/l2+l1
                    else:
                        s = s/l2
                    return s
                if self.param_type=='FMKL':
                    [l1,l2,l3,l4] = param
                    if r==1:
                        return l1 + (1/(1+l4) - 1/(1+l3))/l2
                    terms3, terms4 = _fmkl_lm_terms(l3), _fmkl_lm_terms(l4)
                    return (terms3[r-2] - (-1)**(r-1)*terms4[r-2])/l2
            l1 = lr(1,param)
            l2 = lr(2,param)
            l3 = lr(3,param)
            l4 = lr(4,param)
            return l1,l2,l3/l2, l4/l2
        if self.param_type=='RS':
            def fun_opt(x):
                """Auxiliary function for optimization."""                
                [l3, l4] = x
                L2 = -1/(1+l3)+2/(2+l3)-1/(1+l4)+2/(2+l4)
                return np.max([np.abs((1/(l3+1) - 6/(2+l3) + 6/(3+l3) - 1/(l4+1) + 6/(2+l4) - 6/(3+l4))/L2 - a3),
                               np.abs((-1/(1+l3) + 12/(2+l3) - 30/(3+l3) + 20/(4+l3)-1/(1+l4) + 12/(2+l4) - 30/(3+l4) + 20/(4+l4))/L2 - a4)])
            if initial_guess is None or initial_guess.ndim==0 or len(initial_guess)!=2:
                raise ValueError('Specify initial guess for two parameters')  
            [l3,l4] = optimize.fmin(fun_opt,initial_guess,maxfun = maxfun,xtol=xtol, maxiter=maxiter, disp = disp_optimizer)
            l2 = (-1/(1+l3)+2/(2+l3)-1/(1+l4)+2/(2+l4))/a2
            l1 = a1 + 1/l2*(1/(1+l4) - 1/(1+l3))
            param = np.array([l1,l2,l3,l4]).ravel()
        if self.param_type == 'FMKL':
            def fun_opt(x):
                """Auxiliary function for optimization."""                
                g2_3, g3_3, g4_3 = _fmkl_lm_terms(x[0])
                g2_4, g3_4, g4_4 = _fmkl_lm_terms(x[1])
                L2 = g2_3 + g2_4
                return np.max([np.abs((g3_3 - g3_4)/L2 - a3), np.abs((g4_3 + g4_4)/L2 - a4)])
            if initial_guess is None or initial_guess.ndim==0 or len(initial_guess)!=2:
                raise ValueError('Specify initial guess for two parameters')  
            [l3,l4] = optimize.fmin(fun_opt,initial_guess,maxfun = maxfun,xtol=xtol, maxiter=maxiter, disp = disp_optimizer)
            l2 = (_fmkl_lm_terms(l3)[0] + _fmkl_lm_terms(l4)[0])/a2
            l1 = a1 + 1/l2*(1/(1+l3) - 1/(1+l4))
            param = np.array([l1,l2,l3,l4]).ravel()
        if self.param_type == 'VSL':
            
            if a4**2+98*a4 +1 <0:
                a4 = (-98+(98**2 - 4)**0.5)/2+10**(-10)
            p4 = np.array([(3+7*a4 + np.sqrt(a4**2+98*a4 +1))/(2*(1-a4)), (3+7*a4 - np.sqrt(a4**2+98*a4 +1))/(2*(1-a4))])
            p3 = 0.5*(1-a3*(p4+3)/(p4-1))
            p3[p4==1] = 0.5
            p3[p3<0] = 0
            p3[p3>1] = 1
            p2 = a2*(p4+1)*(p4+2)
            p1 = a1+p2*(1-2*p3)/(p4+1)
            param1 = [p1[0], p2[0],p3[0],p4[0]]
            param2 = [p1[1], p2[1],p3[1],p4[1]]
            if initial_guess is None:
                best = [self.check_param(param1)*1,self.check_param(param2)*1]
                if np.sum(best)==2:
                    GoF = [self.GoF_Q_metric(data,param1),self.GoF_Q_metric(data,param2)]
                    best = (GoF == np.min(GoF))*1
                param = np.array([param1,param2][np.argmax(best)]).ravel()
            else:
                if initial_guess.ndim!=0 and len(initial_guess)!=1:
                    raise ValueError('Specify initial guess for one parameter')
                if np.abs(initial_guess - param1[3]) <= np.abs(initial_guess - param2[3]):
                    param = np.array(param1)
                else:
                    param = np.array(param2)
        if disp_fit:  
            print('')
            print('Sample L-moments: ', sample_lm(data))
            print('Fitted L-moments: ', lm(param))
            print('')
            print('Parameters: ', param)
            if not self.check_param(param):
                print('')
                print('Parameters are not valid. Try another initial guess.')
            else:
                if test_gof:
                    ks, chi2 = self.GoF_tests(param, data, bins_gof)
                    print('')
                    print('Goodness-of-Fit')
                    print(ks)
                    print(chi2)
                self.plot_fitting(data,param,bins = bins_hist)
        return np.array(param)
    
    def grid_search(self, data, fun_min, grid_min = -3, grid_max = 3, n_grid = 10):
        """Find parameters of GLD by grid search procedure.
        
        It does grid search for third and fourth parameters. First two parameters are calculated by fitting
        support to data. It returns parameters with minimum value of `fun_min`.
            
        Parameters
        ----------
        data : array-like
            Input data.
        fun_min : function
            Function of parameters to minimize for choosing the best parameters. For example, negative log-likelihood function.
        grid_min : float, optional
            Minimum value of shape parameters for the grid. The default is -3.
        grid_max : float, optional
            Maximum value of shape parameters for the grid. The default is -3.
        n_grid : int, optional
            Number of grid points for each parameter. The default is 10.

        Returns
        -------
        array-like
            Parameters of GLD.

        """
        eps = 0.01
        def fun_opt_supp(x):
            """Auxiliary function for estimation of first two parameters by fitting support to data."""            
            A = np.min(data)
            B = np.max(data)
            par = np.hstack([x,param[2:]])
            if not self.check_param(par):
                return np.inf
            return np.max([np.abs(self.Q(eps,par) - A),    np.abs(self.Q(1-eps,par) - B)])
    
        if self.param_type == 'VSL':
            p3_list = np.linspace(0,1,n_grid)
            p4_list = np.linspace(grid_min,grid_max,n_grid)
        else:
            p3_list = np.linspace(grid_min,grid_max,n_grid)
            p4_list = np.linspace(grid_min,grid_max,n_grid)
            
        res = np.zeros((n_grid, n_grid))
        for i in range(n_grid):
            for j in range(n_grid):
                param = [np.mean(data),1,p3_list[i], p4_list[j]]
                if self.param_type == 'RS' and not self.check_param(param):
                    param[1] = -1
                x = optimize.fmin(fun_opt_supp,param[:2], disp=False, xtol = 10**(-8))
                param[:2] = x
                res[i,j] = fun_min(param)
        ind = np.unravel_index(np.argmin(res, axis=None), res.shape)
        p3,p4 = p3_list[ind[0]], p4_list[ind[1]]
        param = np.hstack([np.mean(data),1,p3,p4])
        if self.param_type == 'RS' and not self.check_param(param):
            param[1] = -1
        x = optimize.fmin(fun_opt_supp,param[:2], disp=False, xtol = 10**(-8))
        return np.hstack([x,p3,p4])
    
    def fit_MPS(self,data, initial_guess = None, method = 'grid', u = 0.1,grid_min = -3, grid_max = 3, n_grid = 10, xtol=0.0001, maxiter=None, maxfun=None , disp_optimizer=True,   disp_fit = True, bins_hist = None, test_gof = True, bins_gof = 8):
        """Fit GLD to data using method of maximum product of spacing.
        
        It estimates parameters of GLD by maximization of the geometric mean of spacings in the data,
        which are the differences between the values of the cumulative distribution function at neighbouring data points.
        
        This consists of two steps. The first step is finding initial values of parameters for maximization procedure
        using method of moments, method of percentiles, method of L-moments or grid search procedure.
        The second step is maximization of the geometric mean of spacings using numerical methods.
        The optimization procedure is quite difficult and requires some time (especially for large samples).

        Parameters
        ----------
        data : array-like
            Input data.
        initial_guess : array-like, optional
            Initial guess for the first step. Length of initial_guess depends on the method used at the first step.
            It's ignored if method is 'grid'.
        method : str, optional
            Method used for finding initial parameters at the first step. Should be 'MM' for method of moments, 
            'PM' for method of percentiles, 'LMM' for method of L-moments or 'grid' for grid search procedure.  
            The default is 'grid'.
        u : float, optional
            Parameter for calculating percentile-based statistics for method of percentiles. 
            Arbitrary number between 0 and 0.25. The default is 0.1. It's ignored if method is not 'PM'.
        grid_min : float, optional
            Minimum value of shape parameters for the grid search. The default is -3. It's ignored if method is not 'grid'.
        grid_max : float, optional
            Maximum value of shape parameters for the grid search. The default is -3. It's ignored if method is not 'grid'.
        n_grid : int, optional
            Number of grid points for the grid search. The default is 10. It's ignored if method is not 'grid'.
        xtol : float, optional
            Absolute error for optimization procedure. The default is 0.0001.
        maxiter : int, optional
            Maximum number of iterations for optimization procedure. 
        maxfun : int, optional
            Maximum number of function evaluations for optimization procedure. 
        disp_optimizer : bool, optional
            Set True to display information about optimization procedure. The default is True.
        disp_fit : bool, optional
            Set True to display information about fitting. The default is True.
        bins_hist : int, optional
            Number of bins for histogram. It's ignored if disp_fit is False.
        test_gof : bool, optional
            Set True to perform Goodness-of-Fit tests and print results. It's ignored if disp_fit is False. The default is True.
        bins_gof : int, optional
            Number of bins for chi-square test. It's ignored if test_gof is False. The default is 8.
        Raises
        ------
        ValueError
            If input parameters are incorrect or parameters of GLD from the first step are not valid.
        Returns
        -------
        array-like
            Fitted parameters of GLD.
            
         References:
        ----------
        .. [1] Cheng, R.C.H., & Amin, N.A.K. 1983. Estimating parameters in continuous univariate
            distributions with a shifted origin. Journal of the Royal Statistical Society: Series B
            (Methodological), 45(3), 394–403.

        .. [2] Ranneby, B. 1984. The maximum spacing method. an estimation method related to the
            maximum likelihood method. Scandinavian Journal of Statistics, 93–112.
            
        .. [3] Chalabi, Y., Scott, D.J., & Wuertz, D. 2012. Flexible distribution modeling with the
            generalized lambda distribution.


        """
        data = np.sort(data.ravel())
        unique, counts = np.unique(data, return_counts=True)
        delta = np.min(np.diff(unique))/2
        ind = np.nonzero(counts>1)[0]
        ind1 = np.nonzero(np.isin(data, unique[ind]))[0]
        # A fixed seed, so that fitting the same data twice gives the same result.
        data[ind1] = data[ind1] + stats.norm.rvs(0,delta/3,len(ind1),random_state=np.random.RandomState(0))
        def S(param):
            """Spacing function for optimization."""            
            if not self.check_param(param):
                return np.inf
            # All n + 1 spacings, with F = 0 before the first point and F = 1 after the last.
            # Without the two end spacings a point outside the support is not penalized.
            sorted_data = np.sort(data)
            u = self.CDF_num(sorted_data,param)
            spacings = np.hstack([u[0], np.diff(u), self.SF_num(sorted_data[-1],param)])
            with np.errstate(divide='ignore'):
                return -np.mean(np.log(np.abs(spacings)))
        if method not in ['MM','LMM','PM','grid']:
                raise ValueError('Unknown method \'%s\' . Use \'MM\',\'LMM\' , \'PM\' or \'grid\'' %method)
        if method=='MM':
            param1 = self.fit_MM(data, initial_guess, xtol=xtol, maxiter=maxiter, maxfun=maxfun , disp_optimizer=False,   disp_fit = False, test_gof = False)
        if method=='PM':
            param1 = self.fit_PM(data, initial_guess, u = u, xtol=xtol, maxiter=maxiter, maxfun=maxfun , disp_optimizer=False,   disp_fit = False, test_gof = False)
        if method=='LMM':
            param1 = self.fit_LMM(data, initial_guess, xtol=xtol, maxiter=maxiter, maxfun=maxfun , disp_optimizer=False,   disp_fit = False, test_gof = False)
        if method=='grid':
            param1 = self.grid_search(data, fun_min = S, grid_min = grid_min, grid_max = grid_max, n_grid = n_grid)
        if not self.check_param(param1):
            raise ValueError('Parameters are not valid. Try another initial guess.')
        if np.min(data)<self.supp(param1)[0] or np.max(data)>self.supp(param1)[1]:
            param1 = self.correct_supp(data, param1)
        param = optimize.fmin(S,param1,maxfun = maxfun,xtol=xtol, maxiter=maxiter, disp = disp_optimizer)
        if disp_fit:  
            print('')
            print('Initial point for Maximum Product of Spacing Method: ', param1)
            print('Estimated by ', method)
            print('')
            print('Initial negative logarithm of mean spacing: ', S(param1))
            print('Optimized negative logarithm of mean spacing: ', S(param))
            print('')
            print('Parameters: ', param)
            if test_gof:
                ks, chi2 = self.GoF_tests(param, data, bins_gof)
                print('')
                print('Goodness-of-Fit')
                print(ks)
                print(chi2)
            self.plot_fitting(data,param,bins = bins_hist)
        return np.array(param)
        
    def fit_ML(self,data, initial_guess = None, method = 'grid', u = 0.1, grid_min = -3, grid_max = 3, n_grid = 10, xtol=0.0001, maxiter=None, maxfun=None , disp_optimizer=True,   disp_fit = True, bins_hist = None, test_gof = True, bins_gof = 8):
        """Fit GLD to data using method of maximum likelihood.
        
        It estimates parameters of GLD by maximizing a likelihood function.
        
        This consists of two steps. The first step is finding initial values of parameters for maximization procedure
        using method of moments, method of percentiles, method of L-moments or grid search procedure.
        The second step is maximization of likelihood function using numerical methods.
        The optimization procedure is quite difficult and requires some time (especially for large samples).

        Parameters
        ----------
        data : array-like
            Input data.
        initial_guess : array-like, optional
            Initial guess for the first step. Length of initial_guess depends on the method used at the first step. 
            It's ignored if method is 'grid'.
        method : str, optional
            Method used for finding initial parameters at the first step. Should be 'MM' for method of moments, 
            'PM' for method of percentiles, 'LMM' for method of L-moments or 'grid' for grid search procedure.  
            The default is 'grid'.
        u : float, optional
            Parameter for calculating percentile-based statistics for method of percentiles. 
            Arbitrary number between 0 and 0.25. The default is 0.1. It's ignored if method is not 'PM'.
        grid_min : float, optional
            Minimum value of shape parameters for the grid search. The default is -3. It's ignored if method is not 'grid'.
        grid_max : float, optional
            Maximum value of shape parameters for the grid search. The default is -3. It's ignored if method is not 'grid'.
        n_grid : int, optional
            Number of grid points for the grid search. The default is 10. It's ignored if method is not 'grid'.
        xtol : float, optional
            Absolute error for optimization procedure. The default is 0.0001.
        maxiter : int, optional
            Maximum number of iterations for optimization procedure. 
        maxfun : int, optional
            Maximum number of function evaluations for optimization procedure. 
        disp_optimizer : bool, optional
            Set True to display information about optimization procedure. The default is True.
        disp_fit : bool, optional
            Set True to display information about fitting. The default is True.
        bins_hist : int, optional
            Number of bins for histogram. It's ignored if disp_fit is False.
        test_gof : bool, optional
            Set True to perform Goodness-of-Fit tests and print results. It's ignored if disp_fit is False. The default is True.
        bins_gof : int, optional
            Number of bins for chi-square test. It's ignored if test_gof is False. The default is 8.
        Raises
        ------
        ValueError
            If input parameters are incorrect or parameters of GLD from the first step are not valid.
        Returns
        -------
        array-like
            Fitted parameters of GLD.
            
         References:
        ----------
        .. [1] Su, S. 2007. Numerical maximum log likelihood estimation for generalized lambda distributions. 
            Computational Statistics & Data Analysis, 51(8), 3983–3998.
            
        """
        data = data.ravel()
        def lnL(param):
            """Likelihood function for optimization."""            
            if not self.check_param(param):
                return np.inf
            return -np.sum(np.log(self.PDF_num(data,param)))
        if method not in ['MM','LMM','PM','grid']:
                raise ValueError('Unknown method \'%s\' . Use \'MM\',\'LMM\', \'PM\' or \'grid\'' %method)
        if method=='MM':
            param1 = self.fit_MM(data, initial_guess, xtol=xtol, maxiter=maxiter, maxfun=maxfun , disp_optimizer=False,   disp_fit = False, test_gof = False)
        if method=='PM':
            param1 = self.fit_PM(data, initial_guess, u = u, xtol=xtol, maxiter=maxiter, maxfun=maxfun , disp_optimizer=False,   disp_fit = False, test_gof = False)
        if method=='LMM':
            param1 = self.fit_LMM(data, initial_guess, xtol=xtol, maxiter=maxiter, maxfun=maxfun , disp_optimizer=False,   disp_fit = False, test_gof = False)
        if method=='grid':
            param1 = self.grid_search(data, fun_min = lnL, grid_min = grid_min, grid_max = grid_max, n_grid = n_grid)
        
        if not self.check_param(param1):
            raise ValueError('Parameters are not valid. Try another initial guess.')
        if np.min(data)<self.supp(param1)[0] or np.max(data)>self.supp(param1)[1]:
            param1 = self.correct_supp(data, param1)
        
        param = optimize.fmin(lnL,param1,maxfun = maxfun,xtol=xtol, maxiter=maxiter, disp = disp_optimizer)
        if disp_fit:  
            print('')
            print('Initial point for Maximum Likilehood Method: ', param1)
            print('Estimated by ', method)
            print('')
            print('Initial negative log-likelihood function: ', lnL(param1))
            print('Optimized negative log-likelihood function: : ', lnL(param))
            print('')
            print('Parameters: ', param)
            if test_gof:
                ks, chi2 = self.GoF_tests(param, data, bins_gof)
                print('')
                print('Goodness-of-Fit')
                print(ks)
                print(chi2)
            self.plot_fitting(data,param,bins = bins_hist)
        return np.array(param)
    
    
    def fit_starship(self,data, initial_guess = None, method = 'grid', u = 0.1,grid_min = -3, grid_max = 3, n_grid = 10, xtol=0.0001, maxiter=None, maxfun=None , disp_optimizer=True,   disp_fit = True, bins_hist = None, test_gof = True, bins_gof = 8):
        """Fit GLD to data using starship method.
        
        It estimates parameters of GLD by transformation data to uniform distribution (using numerical calculation of GLD 
        cumulative distribution function) and optimization goodness-of-fit measure (Andersod-Darling statistic is used).
        
        This consists of two steps. The first step is finding initial values of parameters for optimization procedure
        using method of moments, method of percentiles, method of L-moments or grid search procedure.
        The second step is optimization of Anderson-Darling statistic for transformed data.
        The optimization procedure is quite difficult and requires some time (especially for large samples).

        Parameters
        ----------
        data : array-like
            Input data.
        initial_guess : array-like, optional
            Initial guess for the first step. Length of initial_guess depends on the method used at the first step.
            It's ignored if method is 'grid'.
        method : str, optional
            Method used for finding initial parameters at the first step. Should be 'MM' for method of moments, 
            'PM' for method of percentiles, 'LMM' for method of L-moments or 'grid' for grid search procedure.
             The default is 'grid'.
        u : float, optional
            Parameter for calculating percentile-based statistics for method of percentiles. 
            Arbitrary number between 0 and 0.25. The default is 0.1. It's ignored if method is not 'PM'.
        grid_min : float, optional
            Minimum value of shape parameters for the grid search. The default is -3. It's ignored if method is not 'grid'.
        grid_max : float, optional
            Maximum value of shape parameters for the grid search. The default is -3. It's ignored if method is not 'grid'.
        n_grid : int, optional
            Number of grid points for the grid search. The default is 10. It's ignored if method is not 'grid'.
        xtol : float, optional
            Absolute error for optimization procedure. The default is 0.0001.
        maxiter : int, optional
            Maximum number of iterations for optimization procedure. 
        maxfun : int, optional
            Maximum number of function evaluations for optimization procedure. 
        disp_optimizer : bool, optional
            Set True to display information about optimization procedure. The default is True.
        disp_fit : bool, optional
            Set True to display information about fitting. The default is True.
        bins_hist : int, optional
            Number of bins for histogram. It's ignored if disp_fit is False.
        test_gof : bool, optional
            Set True to perform Goodness-of-Fit tests and print results. It's ignored if disp_fit is False. The default is True.
        bins_gof : int, optional
            Number of bins for chi-square test. It's ignored if test_gof is False. The default is 8.
        Raises
        ------
        ValueError
            If input parameters are incorrect or parameters of GLD from the first step are not valid.
        Returns
        -------
        array-like
            Fitted parameters of GLD.
            
         References:
        ----------
         .. [1] King, R. A. R., and MacGillivray, H. L. 1999. "A Starship Estimation Method for
            the Generalized Lambda Distributions," Australian and New Zealand Journal of Statistics, 41, 353–374.
            
        """
        data = np.sort(data.ravel())
        def fun_opt(param):
            """AD-statistic for optimization."""            
            if not self.check_param(param):
                return np.inf
            u = self.CDF_num(data, param)
            sf = self.SF_num(data, param)
            return -len(data) - 1/len(data)*np.sum((np.arange(1,len(data)+1)*2 - 1)*(np.log(u) + np.log(sf[::-1])))
        if method not in ['MM','LMM','PM','grid']:
                raise ValueError('Unknown method \'%s\' . Use \'MM\',\'LMM\' , \'PM\' or \'grid\'' %method)
        if method=='MM':
            param1 = self.fit_MM(data, initial_guess, xtol=xtol, maxiter=maxiter, maxfun=maxfun , disp_optimizer=False,   disp_fit = False, test_gof = False)
        if method=='PM':
            param1 = self.fit_PM(data, initial_guess, u = u, xtol=xtol, maxiter=maxiter, maxfun=maxfun , disp_optimizer=False,   disp_fit = False, test_gof = False)
        if method=='LMM':
            param1 = self.fit_LMM(data, initial_guess, xtol=xtol, maxiter=maxiter, maxfun=maxfun , disp_optimizer=False,   disp_fit = False, test_gof = False)
        if method=='grid':
            param1 = self.grid_search(data, fun_min = fun_opt, grid_min = grid_min, grid_max = grid_max, n_grid = n_grid)
        if not self.check_param(param1):
            raise ValueError('Parameters are not valid. Try another initial guess.')
        if np.min(data)<self.supp(param1)[0] or np.max(data)>self.supp(param1)[1]:
            param1 = self.correct_supp(data, param1)
        param = optimize.fmin(fun_opt,param1,maxfun = maxfun,xtol=xtol, maxiter=maxiter, disp = disp_optimizer)
        if disp_fit:  
            print('')
            print('Initial point for Starship Method: ', param1)
            print('Estimated by ', method)
            print('')
            print('Initial AD-statistic: ', fun_opt(param1))
            print('Optimized AD-statistic : ', fun_opt(param))
            print('')
            print('Parameters: ', param)
            if test_gof:
                ks, chi2 = self.GoF_tests(param, data, bins_gof)
                print('')
                print('Goodness-of-Fit')
                print(ks)
                print(chi2)
            self.plot_fitting(data,param,bins = bins_hist)
        return np.array(param)
    
    
    def fit_curve(self,x,y, initial_guess = None, method = 'MM', u = 0.1, N_gen = 1000, shift = False, ymin = 0.01, optimization_phase = True, random_state = None, xtol=0.0001, maxiter=None, maxfun=None,disp_optimizer=True,disp_fit = True,bins_hist = None,):
        """Fit GLD to arbitrary curve given by x and y coordinates.
        
        It models a curve as `y = c * GLD.PDF_num(x, param) + shift_val` where `param` is parameters of GLD,
        `c` is normalizing constant and `shift_val` is y-shift.
        If y-shift is zero the values of y should be non-negative.
        
        The procedure of curve fitting consists of two phases: simulation and optimization.
        
        At the simulation phase the curve is normalized to specify probability density function
        (it should be non-negative and the area under the curve should be equal to 1).
        Then it generates sample of random variables defined by this density function and fit GLD to the sample
        using one of the methods: method of moments, method of percentiles or method of L-moments.
        
        Then at the optimization phase it provides more accurate solution by minimizing mean square error.
        This procedure is quite difficult and requires some time, so optimization phase is optional.

        Parameters
        ----------
        x : array-like
            x-coordinates of the curve.
        y : array-like
            y-coordinates of the curve. 
        initial_guess : array-like, optional
            Initial guess for estimating parameters at the simulation phase. Length of initial_guess depends on the method used at the simulation phase.
        method : str, optional
            Method used for estimating parameters at the simulation phase. Should be 'MM' for method of moments, 
            'PM' for method of percentiles or 'LMM' for method of L-moments.  The default is 'MM'.
        u : float, optional
            Parameter for calculating percentile-based statistics for method of percentiles. 
            Arbitrary number between 0 and 0.25. The default is 0.1. It's ignored if method is not 'PM'.
        N_gen : int, optional
            Size of sample generated at the simulation phase. The default is 1000.
        shift : bool, optional
            Set True to fit y-shift. Set False to use zero y-shift. The default is False.
        ymin : float, optional
            Minimum value of y-coordinates after shifting. Should be positive. The default is 0.01. It's ignored if shift is False.
        optimization_phase : bool, optional
            Set True to perform optimization phase. Set False to skip optimization phase. The default is True.
        random_state : None or int, optional
            The seed of the pseudo random number generator. The default is None.
        xtol : float, optional
            Absolute error for optimization procedure. The default is 0.0001.
        maxiter : int, optional
            Maximum number of iterations for optimization procedure. 
        maxfun : int, optional
            Maximum number of function evaluations for optimization procedure. 
        disp_optimizer : bool, optional
            Set True to display information about optimization procedure. The default is True.
        disp_fit : bool, optional
            Set True to display information about fitting. The default is True.
        bins_hist : int, optional
            Number of bins for histogram. It's ignored if disp_fit is False.
         
        Raises
        ------
        ValueError
            If input parameters are incorrect or parameters of GLD from the simulation phase are not valid.
         
        Returns
        -------
        param : array-like
            Parameters of GLD.
        c : float
            Normalizing constant.
        shift_val : float
            Value of y-shift.
    
        """
        x = x.ravel()
        y = y.ravel()
        if not shift and ((y<0).any() or (y==0).all()):
            raise ValueError('y shouldn\'t be zero or contain negative values. Use \'shift = True\'')
        if shift and ymin<=0:
            raise ValueError('ymin should be positive')
        if shift:
            shift_val =  np.min(y) - ymin
        else:
            shift_val = 0
        y = y-shift_val    
        S = np.diff(x)*(y[:-1]+y[1:])/2
        C = np.sum(S)
        p = S/C
        y1 = y/C
        generator = np.random if random_state is None else np.random.RandomState(random_state)
        def gen(p):
            """Auxiliary function for generating random variables."""            
            a = generator.rand(1)
            i = np.nonzero(a<=np.cumsum(p))[0][0]
            return x[i] + 2*(a - np.sum(p[:i]))/(y1[i]+y1[i+1])
        data = np.array([gen(p) for i in range(N_gen)]).ravel()
        if method not in ['MM','LMM','PM']:
            raise ValueError('Unknown method \'%s\' . Use \'MM\',\'LMM\' or \'PM\'' %method)
        if method=='MM':
            param1 = self.fit_MM(data, initial_guess, xtol=xtol, maxiter=maxiter, maxfun=maxfun , disp_optimizer=False,   disp_fit = False, test_gof = False)
        if method=='PM':
            param1 = self.fit_PM(data, initial_guess, u = u, xtol=xtol, maxiter=maxiter, maxfun=maxfun , disp_optimizer=False,   disp_fit = False, test_gof = False)
        if method=='LMM':
            param1 = self.fit_LMM(data, initial_guess, xtol=xtol, maxiter=maxiter, maxfun=maxfun , disp_optimizer=False,   disp_fit = False, test_gof = False)
        if not self.check_param(param1):
            raise ValueError('Parameters are not valid. Try another initial guess.')
        def fun_opt(p):
            """Auxiliary function for optimization."""            
            param = p[:4]
            c = p[4]
            if not self.check_param(param):
                return np.inf
            return np.mean((self.PDF_num(x,param)*c - y)**2)
        if optimization_phase:
            res = optimize.fmin(fun_opt,np.hstack([param1,C]),xtol = xtol, maxiter = maxiter,maxfun = maxfun, disp = disp_optimizer)
            param = res[:4]
            c = res[4]
        else:
            param = param1
            c = C
        if disp_fit:  
            print('')
            print('MSE: ',fun_opt(np.hstack([param,c]) ))
            print('')
            print('Parameters: ', param)
            print('C: ', c)
            if shift:
                print('shift: ', shift_val)
            fig, ax = plt.subplots(1,2,figsize = (12,3.5))
            ax[0].grid()
            ax[0].hist(data,color = 'skyblue',density = True, bins = bins_hist)
            p = np.linspace(0.01,0.99,100)
            ax[0].plot(self.Q(p,param1), self.PDF_Q(p,param1),'r')
            ax[0].set_title('Simulation phase')
            y_fit = self.PDF_num(x,param)*c + shift_val
            y = y + shift_val
            delta = np.max([0.05*np.abs(np.mean(y)), 10**(-5)])
            ax[1].grid()
            ax[1].plot(x,y,'b')      
            ax[1].plot(x,y_fit,'r')
            ax[1].set_ylim(np.min([y,y_fit]) - delta,np.max([y,y_fit])+delta)
            ax[1].legend(['data', 'GLD'])
            ax[1].set_title('Result of curve fitting')
        return param, c, shift_val