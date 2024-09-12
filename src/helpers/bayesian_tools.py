"""
File Description:
    Contains helpers to be used when performing bayesian np.inference

Most of these functions have been defined by Juan Chiachío & Lourdes Jalón,
and adapted by asanchezlc
"""
import math
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings

from matplotlib import rc, rcParams
from numpy.random import multivariate_normal
from scipy.interpolate import interp1d
from scipy.special import erf


def generate_def_interv(E, rho, R1, R2, T, M2, M3, sigma_f, w, n_joints=None,
                        n_frames=10, joint_name='supportpoints', frame_name='releaseframes'):
    """
    Function duties:
        It generates dictionary with definition intervals
    Input:
        E, rho, ...: list containing start and end interval points
    Remark:
        SAP2000 groups must be defined similarly
        It can also be used for getting the initial values of the parameters
    """
    def_interv = dict()
    if E:
        def_interv['E'] = E
    if rho:
        def_interv['rho'] = rho
    if n_joints:
        for i in range(n_joints):
            if R1:
                def_interv[f'{joint_name}_{i+1}_R1'] = R1
            if R2:
                def_interv[f'{joint_name}_{i+1}_R2'] = R2
    else:
        if R1:
            def_interv[f'{joint_name}_all_R1'] = R1
        if R2:
            def_interv[f'{joint_name}_all_R2'] = R2
    if n_frames:
        for i in range(n_frames):
            if T:
                def_interv[f'{frame_name}_{i+1}_T'] = T
            if M2:
                def_interv[f'{frame_name}_{i+1}_M2'] = M2
            if M3:
                def_interv[f'{frame_name}_{i+1}_M3'] = M3
    else:
        if T:
            def_interv[f'{frame_name}_all_T'] = T
        if M2:
            def_interv[f'{frame_name}_all_M2'] = M2
        if M3:
            def_interv[f'{frame_name}_all_M3'] = M3
    if sigma_f:
        def_interv['sigma_f'] = sigma_f
    if w:
        def_interv['w'] = w

    return def_interv


def check_ini_values(Definition_ini, Definition_interv):
    """
    Function Duties:
        Check if the initial values are within the interval
    Input:
        Definition_ini: dict with initial values
        Definition_interv: dict with interval limits
    Output:
        Definition_ini: dict in which only the initial values within interval limits are used
    """
    for key in list(Definition_ini):
        if Definition_ini[key] > Definition_interv[key][1] or Definition_ini[key] < Definition_interv[key][0]:
            Definition_ini.pop(key)
            warnings.warn(f'Initial value for {key} is out of the interval. It will not be used.')

    return Definition_ini


def initial_parameters(Definition_interv):
    """
    Input duties:
        It generates Parameters dictionary, with each element
        being a random number within parameters interval
    Input:
        Definition_interv: dictionary containing list of 2 elements
        (lower and upper interval limits)
    Output:
        Parameters: dictionary
        ME: bool flag:
            False: generated correctly
            True: not generated
    """
    Parameters = dict()
    ME = False
    for _, key in enumerate(Definition_interv):
        up_lim = Definition_interv[key][1]
        lower_lim = Definition_interv[key][0]
        Parameters[key] = np.array(
            lower_lim + np.random.rand()*(up_lim - lower_lim))

    return Parameters, ME


def proposal(Param, Param_Limits, PercentOnInterval, Testing):
    """
    Function duties:
        Generates a sample from a normal distribution (either one
        or multidimensional). Sample generated is inside the
        parameters intervals and depends on the prior value (its
        mean is the prior value).
    Input:
        Param: list; each element is the prior value of each parameter.
            Remark: it is used to build the sampling (normal) function
            from which we sample (we sample from a normal distribution
            with mean = prior sample)
        Param_Limits: dictionary; each key is a parameter and contains a list
            with its limits.
            Remark: the sample generated is always inside these limits
        PercentOnInterval: list;
            It controls the variance of the normal distribution for each of
            the parameters
            Remark: smaller PercentOnInterval -> smaller sigma -> higher
            acceptance rate
        Testing: parameters' labels
    Output:
        Theta_prop: dictionary; each key has a value, which is the proposed
            sample
    """
    Theta_prop, Mean_Var = {}, {}
    warn_dict = dict()
    ME = False

    # Check over parameter intervals (added by aslc)
    for i, key in enumerate(Param_Limits):
        delta_interv = Param_Limits[key][1] - Param_Limits[key][0]
        if delta_interv == 0 and PercentOnInterval[i] > 0:
            PercentOnInterval[i] = 0
            message = f'Variance of {key} is set to 0 as there is not interval amplitude'
            warn_dict[key] = message

    # Loop for each parameter
    for i, key in enumerate(Testing):
        # Mean=prior_value; Variance: controlled by PercentOnInterval (scaled by interval amplitude)
        Mean_Var[key] = [np.array(Param[i], ndmin=1), np.array(
            (PercentOnInterval[i]*abs(Param_Limits[key][1]-Param_Limits[key][0]))**2, ndmin=1)]

        # Variable initialization for entering in loop
        Theta_prop[key] = -np.inf*np.ones(np.size(Param[i]))
        cnt = 0

        # Sample generation (normal dist. with Mean=prior_value; Variance: controlled by PercentOnInterval)
        while ((Theta_prop[key]-Param_Limits[key][0])*(Param_Limits[key][1]-Theta_prop[key]) < 0).any() and cnt < 1000:
            Theta_prop[key] = multivariate_normal(
                Mean_Var[key][0], np.diag(Mean_Var[key][1]), 1)[0]
            cnt += 1
            # While loop ensures that the sample is within the parameter interval

        # If we are not able to generate a sample within the interval limits
        if cnt == 1000:
            Theta_prop = {}
            ME = True
            warn_dict[key] = f'Unable to generate {key} within interval limits'
            break

    return Theta_prop, ME, warn_dict


def truncated_normal(x, mu, sigma, a, b):
    """
    Input:
        x: value
        mu, sigma: mean and std. deviation
        a, b: interval limits
    Return:
        y = probab(x | mu, sigma, a, b)
    """
    if x < a or x > b:
        y = 0
    else:
        chi = (x - mu) / sigma
        x1 = (b - mu) / sigma
        x2 = (a - mu) / sigma
        phi_1 = 1/2 * (1 + erf(x1/np.sqrt(2)))
        phi_2 = 1/2 * (1 + erf(x2/np.sqrt(2)))

        y = 1/sigma * (1/np.sqrt(2*np.pi)) * \
            np.exp(-1/2*chi**2) / (phi_1 - phi_2)

    return y


def fast_interp2d(x0, y0, x1, y1, xi, yi,
                  z_x0_y0, z_x0_y1, z_x1_y0, z_x1_y1):
    """
    Function duties:
        Linear interpolation for a 2d function.
    Remark:
        It is defined because scipy is computationally
        inefficient
    Input:
        xi, yi: desired coordinates were computing z
        x0, y0: point coordinates with known z values
        z_x0_y0, z_x0_y1, z_x1_y0, z_x1_y1: known values of z
            at the interval extremes
    """
    delta_x = (xi - x0)/(x1-x0)
    delta_y = (yi - y0)/(y1-y0)
    delta_f_x0 = (z_x0_y1 - z_x0_y0)
    delta_f_x1 = (z_x1_y1 - z_x1_y0)

    z_x0_yi = z_x0_y0 + delta_y * delta_f_x0
    z_x1_yi = z_x1_y0 + delta_y * delta_f_x1
    z_xi_yi = z_x0_yi + delta_x * (z_x1_yi - z_x0_yi)

    return z_xi_yi


def Log_likelihood(Parameters, frequencies_fem, frequencies_oma,
                   Psi_fem, Psi_oma, ME):
    """
    Function Duties:
        Computes the log_likelihood for frequencies and modeshapes
    Parameters:
        Psi_fem: FEM modeshapes (already matched!)
        Psi_oma: REAL modeshapes (already matched!)
        frequencies_fem: FEM frequencies (already matched!)
        frequencies_oma: REAL frequencies (already matched!)
        ME: boolean flag
    Output:
        Log-likelihood of Psi_fem and frequencies_fem given Psi_oma and frequencies_oma
    References: This function follows the approach of:
        Sub-structure Coupling for Dynamic Analysis: Application to Complex
        Simulation-Based Problems Involving Uncertainty (Hector Jensen & Costas Papadimitriou)
            [Chapter 7: Bayesian Finite Element Model Updating]
    """
    if 'sigma_f' in list(Parameters):
        sigma_f = Parameters['sigma_f']

    if 'w' in list(Parameters):
        w = Parameters['w']
        use_modeshapes = True
    else:
        use_modeshapes = False

    if ME:
        LN_LIK = -np.inf

    if use_modeshapes:  # REVISAR LA FORMULACIÓN
        # A) Scale Psi
        beta = np.diag(Psi_fem.T @ Psi_oma) / np.linalg.norm(Psi_fem, axis=0)**2
        Psi_scaled = Psi_fem @ np.diag(beta)

        # B) LN_LIK computation
        m = len(frequencies_fem)
        N = np.shape(Psi_oma)[0]
        log_denominator = m*N*np.log(w)/2-(m+N*m)*np.log(sigma_f)
        J1 = 2*np.pi*np.sum((frequencies_fem - frequencies_oma)**2/frequencies_oma)
        J2 = np.sum(np.linalg.norm(Psi_scaled - Psi_oma, axis=0)
                    ** 2/np.linalg.norm(Psi_oma, axis=0)**2)
        LN_LIK = log_denominator-0.5*(J1+w*J2)/sigma_f**2
    else:
        m = len(frequencies_fem)
        w_fem = 2*np.pi*frequencies_fem
        w_oma = 2*np.pi*np.array(frequencies_oma)
        # log_denominator = -m*(np.log(sigma_f) + np.log(2*np.pi)/2) - np.sum(np.log(w_oma))
        log_denominator = -m*(np.log(sigma_f))  # the other variables are always the same
        J1 = np.sum((w_fem - w_oma)**2/w_oma**2)
        LN_LIK = log_denominator - 0.5*J1/sigma_f**2

    return LN_LIK, ME


def Log_likelihood_HR(Parameters, model_modal, ME,
                      variables_method, method=2, number_modes=None, p=2):
    """
    Function duties:
        Computes log(likelihood) of measured data given the
        stochastic model defined by:
            Normal distribution for frequencies
            Different pdf options (conditioned by
            method) for MAC. method=2 is recommended)
    Input:
        Parameters: Model parameters that produce the system response
            given in model_modal
        model_modal: dictionary containing modal information for each mode.
            Remark: modal information w.r.t. experimental mode is stored
                within 'comparison' item.
        variables_method: variables required for calculating LN_LIK_MAC
        ME: boolean flag
        number_modes: number of modes to be considered in the analysis
        p=2 indicates that we are assuming Gaussian LIK
    Output:
        LN_LIK: log(Likelihood) of the measured data given the model
            parameters (considering normal distribution for p=2)
        ME: boolean flag
    Some parameters for understanding the computation:
        ct: constant multiplying exponential term of the Normal
            distribution (p**(1-1/p)/(2*math.gamma(1/p)) = 1/sqrt(2*pi))
            for p=2
        LN_LIK_freq: Likelihood for a Normal distribution (if p=2)
            ct*np.power(sigma, -N) = (1/sqrt(2*pi*sigma))^N (if p=2)
    """
    if number_modes is None:
        number_modes = len(model_modal)
    if ME:
        LN_LIK = -np.inf
    else:
        delta_f = [model_modal[mode]['comparison']['delta_f']
                   for mode in list(model_modal)[0:number_modes]]
        mac = [model_modal[mode]['comparison']['MAC']
               for mode in list(model_modal)[0:number_modes]]
        LN_LIK = 0
        sigma_f, sigma_m = Parameters['sigma_f'], Parameters['sigma_m']
        for i in range(number_modes):
            lp, N = (delta_f[i])**p, 1
            ct = np.power(p**(1-1/p)/(2*math.gamma(1/p)), N)
            LN_LIK_freq = np.log(ct*np.power(sigma_f, -N)) + \
                (-1/(p*np.power(sigma_f, p))*lp)
            """
            2 changes have been made: (1) LN_LIK_freq and (2) LN_LIK_MAC
            (1) LN_LIK_freq expression has been modified by ASLC because a Numerical approx. error
                has been detected: example:
                    np.log(np.exp(1000)) = 1000 and returns inf --> we substitute log(a*b) by
                        log(a) + log(b), and like b=e^x, log(b)=x
                Remark: former expression was:
                    LN_LIK += np.log(ct*np.power(sigma, -N) * np.exp(-1/(p*np.power(sigma, p))*lp)*lp_Mac)
            (2): A lognormal approximation for MAC with a big dispersion had been considered. Some better
                approximations are proposed
                Remark: former expression was:
                    lp_Mac = (1/(1-mac[i]))/s (s=1)                    
            """
        if method == 1:  # Likelihood from JCHR
            s = 1
            lp_Mac = (1/(1-mac[i]))/s
            LN_LIK = np.log(lp_Mac)

        elif method == 2:  # Likelihood from Truncated normal
            # Remark: check if it's worthy to define a fast_interp1d
            f_mu = interp1d(
                variables_method['sigma_m'], variables_method['mu'])
            f_var = interp1d(
                variables_method['sigma_m'], variables_method['var'])
            mu = f_mu(sigma_m)
            std_dev = np.sqrt(f_var(sigma_m))
            a, b = 0, 1  # truncated normal parameters
            lp_Mac = truncated_normal(mac[i], mu, std_dev, a, b)
            LN_LIK = np.log(lp_Mac)

        elif method == 3:  # Likelihood from approximated KDE
            x_mac = mac[i]
            # LinearNDInterpolator is comput. inefficient -> Manual linear fit
            all_sigma = [float(i) for i in list(variables_method)]
            all_sigma = np.sort(all_sigma)
            sigma_inf = all_sigma[np.where(sigma_m >= all_sigma)[0][-1]]
            sigma_sup = all_sigma[np.where(sigma_m <= all_sigma)[0][0]]
            # same in sigma_sup
            x_points = variables_method[str(sigma_inf)]['x']
            x_inf = x_points[np.where(x_mac >= x_points)[0][-1]]
            x_sup = x_points[np.where(x_mac <= x_points)[0][0]]
            z_x0_y0 = variables_method[str(
                sigma_inf)]['y'][x_points.index(x_inf)]
            z_x0_y1 = variables_method[str(
                sigma_sup)]['y'][x_points.index(x_inf)]
            z_x1_y0 = variables_method[str(
                sigma_inf)]['y'][x_points.index(x_sup)]
            z_x1_y1 = variables_method[str(
                sigma_sup)]['y'][x_points.index(x_sup)]

            lp_Mac = fast_interp2d(x_inf, sigma_inf, x_sup, sigma_sup, x_mac, float(sigma_m),
                                   z_x0_y0, z_x0_y1, z_x1_y0, z_x1_y1)
            LN_LIK = np.log(lp_Mac)

    return LN_LIK, ME


def Log_likelihood_old(Parameters, model_modal, ME,
                       number_modes=None, p=2):
    """
    Function duties:
        Computes log(likelihood) of measured data given the
        stochastic model defined by a Normal distribution (if p=2)
        for frequencies and a log-normal distribution for
        MAC values
    Input:
        Parameters: Model parameters that produce the system response
            given in model_modal
        DataResponse: list of 2 arrays;
            Each array corresponds to x and y data; y data is compared
            w.r.t. model response
        ME: boolean flag
        number_modes: number of modes to be considered in the analysis
        p=2 indicates that we are assuming Gaussian LIK
    Output:
        LN_LIK: log(Likelihood) of the measured data given the model
            parameters (considering normal distribution for p=2)
        ME: boolean flag
    Some parameters for understanding the computation:
        ct: constant multiplying exponential term of the Normal
            distribution (p**(1-1/p)/(2*math.gamma(1/p)) = 1/sqrt(2*pi))
            for p=2
        LN_LIK: Likelihood for a Normal distribution (if p=2)
            ct*np.power(sigma, -N) = (1/sqrt(2*pi*sigma))^N (if p=2) together
            with a log-normal distribution with high dispersion [see:
                "Bayesian structural parameter identification from ambient vibration
                in cultural heritage buildings: The case of the San Jerónimo monastery
                in Granada, Spain"]
    """
    if number_modes is None:
        number_modes = len(model_modal)
    if ME:
        LN_LIK = -np.inf
    else:
        delta_f = [model_modal[mode]['comparison']['delta_f']
                   for mode in list(model_modal)[0:number_modes]]
        mac = [model_modal[mode]['comparison']['MAC']
               for mode in list(model_modal)[0:number_modes]]
        LN_LIK = 0
        sigma = Parameters['sigma']
        for i in range(number_modes):
            lp, N = (delta_f[i])**p, 1
            s = Parameters['dispersion']

            ct = np.power(p**(1-1/p)/(2*math.gamma(1/p)), N)
            """
            Expression modified by aslc for 2 reasons:
                A) Numerical approx. error detected (e.g. np.log(np.exp(1000)) = 1000 and
                    returns inf, so it's better to substitute np.log(np.exp(x)) by x; so
                    change in LN_LIK)
                B) Dispersion is not very high, so exponential component of lognormal
                    likelihood should not be discarded (change in lp_Mac)
            Old expressions:
                LN_LIK += np.log(ct*np.power(sigma, -N) * np.exp(-1/(p*np.power(sigma, p))*lp)*lp_Mac)
                lp_Mac = (1/(1-mac[i]))/s (s=1)                    
            """
            lp_Mac = 1-mac[i]
            LN_LIK_Normal = np.log(ct*np.power(sigma, -N)) + \
                (-1/(p*np.power(sigma, p))*lp)
            LN_LIK_LogNormal = -1/2 * \
                np.log(2*np.pi*s**2) + np.log(lp_Mac) - \
                (1/(2*s**2)*np.log(lp_Mac)**2)
            LN_LIK += LN_LIK_Normal + LN_LIK_LogNormal
            # LN_LIK += np.log(ct*np.power(sigma, -N)) + (-1/(p*np.power(sigma, p))*lp) + np.log(lp_Mac)

    return LN_LIK, ME

