import numpy as np

def sample_wigner_invCDF(N_samples:int,
                         rng=None, seed=None):
    """
    Sample the wigner distribution using inverse CDF sampling.

    This function simply samples from the wigner distribution using inverse
    CDF sampling and is used by other functions for generating resonance level spacing.

    Parameters
    ----------
    N_samples : int
        Number of samples and/or length of sample vector.
    rng : np.random.Generator or None
        Numpy random number generator object. Default is None.
    seed : int or None
        Random number generator seed. Only used when rng is None. Default is None.

    Returns
    -------
    numpy.ndarray or float
        Array of i.i.d. samples from wigner distribution.

    Notes
    -----
    
    
    Examples
    --------
    >>> from theory import resonance_statistics
    >>> np.random.seed(7)
    >>> resonance_statistics.sample_wigner_invCDF(2,10)
    array([1.7214878 , 1.31941784])
    """
    # Random number generator:
    if rng is None:
        if seed is None:
            rng = np.random # uses np.random.seed
        else:
            rng = np.random.default_rng(seed) # generates rng from provided seed

    samples = np.sqrt(-4/np.pi*np.log(rng.uniform(low=0.0,high=1.0,size=N_samples)))
    if N_samples == 1:
        samples = np.ndarray.item(samples)
    return samples

def sample_NNE_energies(E_range, avg_level_spacing:float,
                        rng=None, seed=None):
    """
    Sample resonance energies for the ladder using inverse CDF sampling.

    Parameters
    ----------
    E_range : array-like
        The energy range for sampling.
    avg_level_spacing : float
        The mean level spacing.
    rng : np.random.Generator or None
        Numpy random number generator object. Default is None.
    seed : int or None
        Random number generator seed. Only used when rng is None. Default is None.

    Returns
    -------
    np.ndarray
        Array of resonance energies sampled from the Wigner distribution.

    Notes
    -----
    See sample_GE_energies for sampling energies from Gaussian Ensembles.
    """
    # Random number generator:
    if rng is None:
        if seed is None:
            rng = np.random # uses np.random.seed
        else:
            rng = np.random.default_rng(seed) # generates rng from provided seed
    
    E_limits = (min(E_range), max(E_range))
    num_res_est = round((E_limits[1] - E_limits[0]) / avg_level_spacing)
    num_res_tot = num_res_est + round(3.3252*np.sqrt(num_res_est)) # there will only be more resonances than this once in 1e10 samples.
    
    level_spacings = np.zeros((num_res_tot+1,))
    level_spacings[0] = avg_level_spacing * np.sqrt(2/np.pi) * np.abs(rng.normal())
    level_spacings[1:] = avg_level_spacing * sample_wigner_invCDF(num_res_tot, rng=rng)
    res_E = E_limits[0] + np.cumsum(level_spacings)
    res_E = res_E[res_E < E_limits[1]]
    return res_E

