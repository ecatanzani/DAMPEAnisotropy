from __future__ import division
import numpy as np
import healpy as hp  # need version 1.12.5 or higher
import matplotlib.pyplot as mp
from scipy.integrate import quad
from scipy.optimize import fsolve, curve_fit
from scipy.stats import chi2
from ipywidgets import FloatProgress
from IPython.display import display
import os
import pickle
import numbers


def create_dipole_map(nside, dipole, lonlat=None):
    '''
    Create an anisotropy map with given nside and
        dipole amplitude
    If the direction (lonlat) is not specified,
        the dipole is created on random
    '''
    if lonlat is None:
        cl = [0., 1.]
        sky = hp.synfast(cl, nside, verbose=False)
    else:
        if len(lonlat) != 2:
            raise ValueError('Direction should' +
                             ' be (lon, lat)')
        sky = np.zeros(hp.nside2npix(nside))
        ipix = hp.ang2pix(nside, lonlat[0],
                          lonlat[1], lonlat=True)
        sky[ipix] = 1.
        new_lon = lonlat[0] + 180.
        ipix = hp.ang2pix(nside, new_lon,
                          -lonlat[1], lonlat=True)
        sky[ipix] = -1.
        sky -= hp.remove_dipole(sky, verbose=False)
    sky -= sky.mean()
    sky /= sky.max()
    sky *= dipole
    return sky


def get_obs_map(iso_map, ani_map):
    return (ani_map + 1) * iso_map


def create_noise_map(iso_map):
    '''
    Noise STD is equal to 1. / sqrt(N_observations)
    '''
    noise = np.random.poisson(iso_map) / iso_map - 1
    return noise


def get_dipole_ampl(sky, subtract_noise=False, iso_map=None):
    '''
    Returns the reconstructed dipole amplitude
    If subtract_noise is True, use the procedure,
        described in Fermi paper (1903.02905)
        for the noise subtraction.
        Exposure map should be provided in this case
    '''
    if not subtract_noise:
        mono, dipole = hp.fit_dipole(sky)
        return np.sqrt((dipole * dipole).sum())
    else:
        if iso_map is None:
            raise ValueError("Please, specify the isotropy map")
        C_ell = hp.anafast(sky)[1]
        obs_map = get_obs_map(iso_map, sky)
        C_N = 4. * np.pi / len(iso_map)**2 * (obs_map / iso_map**2).sum()
        # C_N /= 1.35**2
        return 3 * np.sqrt((C_ell - C_N) / 4 / np.pi)


def get_dipole_dir(sky):
    '''
    Return the direction of the reconstructed dipole
        in (RA, dec) coordinated
    '''
    mono, dipole = hp.fit_dipole(sky)
    d = hp.vec2ang(dipole, lonlat=True)
    return (d[0][0], d[1][0])


def get_ang_dict(nside, radius):
    '''
    An auxiliary function, which makes
        the tophat smoothing much faster
    Usually not used alone
    When used the first time with given set of
        input parameters, creates a file
        with a dictionary, where for each sky pixel
        has a corresponding array of pixels within a
        circle of given radius.
    Returns that dictionary
    '''
    cwd = os.getcwd()
    path = cwd + '/npix_dict'
    if not os.path.exists(path):
        os.makedirs(path)
    filename = 'ang-dict_nside{}_radius{}.dict'.format(nside,
                                                       radius)
    filename_full = path + '/' + filename

    if os.path.isfile(path + '/' + filename):
        with open(filename_full, 'rb') as f:
            return pickle.load(f)
    else:
        npix = hp.nside2npix(nside)
        pix_num = np.arange(npix)
        thetas, phis = hp.pix2ang(nside, pix_num, lonlat=True)
        d = {}
        for ipix in pix_num:
            theta, phi = thetas[ipix], phis[ipix]
            angdist = hp.rotator.angdist([theta, phi],
                                         [thetas, phis], lonlat=True)
            angdist = np.degrees(angdist)
            d[ipix] = angdist <= radius
        with open(filename_full, 'wb') as f:
            pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)
        return d


def tophat_smoothing(sky, radius, progress=True):
    '''
    The top-hat smoothing is often explained as an integration in a circle.
    I believe that it is not correct.
    See some discussions in anisotropy_sensitivity.ipynb
    ---------
    Return a map integrated in tophat circles with the given radius [degrees]
    progress [def = True]
    '''
    npix = len(sky)
    if progress:
        f = FloatProgress(min=0, max=npix)
        display(f)

    nside = hp.npix2nside(npix)
    d = get_ang_dict(nside, radius)
    return_sky = np.empty(npix)
    npix_under_mask = np.empty(npix)

    for ipix in xrange(npix):
        mask = d[ipix]
        return_sky[ipix] = np.mean(sky[mask])
        npix_under_mask[ipix] = mask.sum()
        if progress:
            f.value += 1

    return_sky *= npix / npix_under_mask

    return return_sky, npix_under_mask


def tophat_smoothing_filter(smoothed_maps, npix_maps):
    '''
    smoothed_maps - an array of maps, smoothed at different angles
    npix_maps - an array of corresponding npix maps
    -------
    This method is proved to do not work properly:
    it doesn't really do the thing
    '''
    nside = hp.npix2nside(len(smoothed_maps[0]))
    arg = np.argmin(np.abs(smoothed_maps), axis=0)
    smoothed = np.empty(hp.nside2npix(nside))
    npix = np.empty(hp.nside2npix(nside))
    for i in xrange(hp.nside2npix(nside)):
        smoothed[i] = smoothed_maps[arg[i], i]
        npix[i] = npix_maps[arg[i], i]
    return smoothed, npix, arg


def find_upper_limit(sample, cl=0.95):
    '''
    Returns an upper limit with confidence level cl
    Works specifically with the distribution of
        reconstructed dipole amplitude
    '''
    nbins = 20
    limits = np.linspace(sample.min(), sample.max(), nbins + 1)
    hist = np.zeros(nbins)

    for i, (lim1, lim2) in enumerate(zip(limits[:-1], limits[1:])):
        hist[i] = ((sample > lim1) * (sample <= lim2)).sum()

    binc = limits[:-1] + (lim2 - lim1) / 2

    popt, pcov = curve_fit(_f, binc, hist,
                           p0=[hist.max(), 0., 10, 1e-4])
    ul = chi2.ppf(cl, popt[2]) * popt[3] + popt[1]
    return ul


def cut_on_thr(ani_map, iso_map, thr=1.5):
    '''
    Cut the noise on the anisotropy map
    which deviates from 0 by thr*std
    '''
    thr = thr / np.sqrt(iso_map)
    ani_map[ani_map > thr] = thr[ani_map > thr]
    thr *= -1
    ani_map[ani_map < thr] = thr[ani_map < thr]
    return ani_map


def get_sig_map(ani_map, iso_map, alpha=1, npix_map=None):
    '''
    Modified HAWC formula for the smoothed sky
    See sensitivity_formula.ipynb for explanations

    By HAWC formula, adapted for
        smoothed maps, significance is
        sqrt(N_obs * N_pix / (1 + alpha)) * deltaI
    Here ani_map is deltaI
         iso_map is N_obs
         npix_map is N_pix
    if npix_map is not specified, then it is assumed
        that the ani_map is not smoothed
    '''
    sig = np.sqrt(iso_map / (1 + alpha)) * ani_map
    if npix_map is not None:
        sig *= np.sqrt(npix_map)
    return sig


class UpperLimit:
    '''
    Interface class for the Upper Limit related things
    Logic: 1. you simulate many skys under
        the Null-hypothesis (no anisotropy)
        2. Reconstruct the dipole for all of them
        3. Fit the histogram of the reconstructed dipoles with function f
        4. Find the UL where int_0^{UL} f = 0.95
    '''
    def __init__(self, sample):
        '''
        sample = a sample of the reconstructed dipoles
        '''
        def _f(x, a, m, df, s):
            return a * chi2.pdf((x - m) / s, df)

        nbins = 20
        limits = np.linspace(sample.min(), sample.max(), nbins + 1)
        hist = np.zeros(nbins)

        for i, (lim1, lim2) in enumerate(zip(limits[:-1], limits[1:])):
            hist[i] = ((sample > lim1) * (sample <= lim2)).sum()

        binc = limits[:-1] + (lim2 - lim1) / 2

        self.popt, pcov = curve_fit(_f, binc, hist,
                                    p0=[hist.max(), 0., 10, 1e-4])

    def get_upper_limit(self, cl=0.95):
        return chi2.ppf(cl, self.popt[2]) * self.popt[3] + self.popt[1]

    def get_nhlimits(self, cl=0.68):
        '''
        Returns the limits of the distribution of expected
            results under the null hypothesis.
        '''
        lim = np.array(chi2.interval(cl, self.popt[2]))
        return lim * self.popt[3] + self.popt[1]

    @staticmethod
    def get_experimental_ul(dipole_ampl, cl=0.95, tol=1e-4):
        '''
        Return an experimental dipole upper limit
            as described in PAMELA paper 1509.06249, see formula 11
        dipole_ampl -- the measured dipole amplitude
        cl -- confidence level
        tol -- desired tolerance of the result
        Idea -- the probability to measure dh (delta hat)
            given the real d (delta) follows the P function.
            UL is the solution of the integral (11)
        '''
        def P(dh, d):
            a = 3 * np.sqrt(6 / np.pi) * dh**2 / d**3
            b = np.exp(- 3 / 2 * (dh / d)**2)
            return a * b

        d = 0 + dipole_ampl  # a clumsy way to copy a variable
        i = 0  # the result of the integration will be stored here
        step = d / 10.
        up = True
        one_m_cl = 1 - cl
        while np.abs(i - one_m_cl) > tol:
            i = quad(lambda x: P(x, d), 0, dipole_ampl)[0]
            if i > one_m_cl:
                d += step
                if up:
                    up = False
                    step /= 2.
            elif i < one_m_cl:
                d -= step
                if not up:
                    up = True
                    step /= 2.
            else:
                break
        return d
