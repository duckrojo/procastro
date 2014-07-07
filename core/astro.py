#
#
# Copyright (C) 2014 Patricio Rojo
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of version 2 of the GNU General 
# Public License as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, 
# Boston, MA  02110-1301, USA.
#
#

from __future__ import division, print_function

import astropy.constants as apc
import astropy.units as apu
import scipy as sp
import glob
import os.path as path
import scipy.interpolate as it


def blackbody(T, x, unit=None):
    """Returns blackbody at specified wavelength or frequency

    :param unit: unit for x from astropy.unit. If None then assume microns"""
    
    if not isinstance(T, apu.Quantity):
        T = T*apu.K
    if not isinstance(x, apu.Quantity):
        if unit is None:
            x = x*apu.micron
        elif isinstance (unit, apu.Unit):
            x = x*unit
        else:
            raise ValueError("Specified unit (%s) is not a valid astropy.unit" 
                             % (unit,))
    if x.cgs.unit == apu.cm:
        use_length = True
    elif x.cgs.unit == 1/apu.s:
        use_length = False
    else:
        raise ValueError("Units for x must be either length or frequency, not %s" 
                         % (x.unit,))


    h_kb_t = apc.h/apc.k_B/T

    if use_length:
        B = 2*apc.h*apc.c**2 / (x**5) / (sp.exp(h_kb_t*apc.c/x) - 1)
        B = B.to(apu.erg/apu.cm**2/apu.cm/apu.s)/apu.sr
    else:
        B = 2*apc.h*x**3 / (apc.c**2) / (sp.exp(h_kb_t*x ) - 1)
        B = B.to(apu.erg/apu.cm**2/apu.Hz/apu.s)/apu.sr

    return B



def getfilter(name, 
              filter_unit=None,
              filterdir=None,):
    """Apply filter to spectra. It can be given or blackbody
 
    :param name: filter name
    :param filter_unit: default filter unit that should be in files.  It should be an astropy quantity. Default is nanometers
    """

    if filterdir is None:
        filterdir = '/home/inv/common/standards/filters/oth/svo/'
    filters = glob.glob(filterdir+'/*.dat')
    found = []
    for f in filters:
        if name in path.basename(f):
            found.append(f)

    if len(found)!=1:
        raise ValueError(("Only one of the available filters should match '%s'." +
                          " Be more specific between:\n %s")
              % (name, '\n '.join(found)))

    if filter_unit is None:
        filter_unit = apu.AA

    fwav, fflx = sp.loadtxt(found[0], unpack=True)
    fwav *= filter_unit

    return fwav,fflx




def applyfilter(name, spectra,
                wavfreq=None, nwav_bb=100,
                filter_unit=None, output_unit=None,
                filterdir=None,
                full=False):
    """Apply filter to spectra. It can be given or blackbody
 
    :param name: filter name
    :param spectra: either flux (if iterable) or temperature (if scalar) for blackbody
    :param wavfreq: wavelength or frequency for the spectra. Only necessary when flux is given as spectra
    :param nwav_bb: Number of wavelengths when blackbody sampling
    :param filter_unit: default filter unit that should be in files.  It should be an astropy quantity. Default is nanometers
    :param filterdir: if None use getfilter's default
    :param full: if True, then return (filtered value, central weighted wavelength, equivalent width)
    """

    fwav, fflx = getfilter(name, filterdir)
    filter_unit = fwav.unit
    if output_unit is None:
        output_unit = apu.micron
    wmin,wmax = fwav[0], fwav[-1]
    us = it.UnivariateSpline(fwav, fflx, s=0.0)

    try:
        itr = iter(spectra)
        if wavfreq is None:
            raise ValueError("wavfreq needs to be specified if spectra is given")
        if not isinstance(wavfreq, apu.Quantity):
            wavfreq *= filter_unit

        if wavfreq.cgs.unit == u.cm:
            if not isinstance(spectra, apu.Quantity):
                spectra *= apu.erg/apu.cm**2/apu.cm/apu.s/apu.sr
        elif wavfreq.cgs.unit == 1/u.s:
            if not isinstance(spectra, apu.Quantity):
                spectra *= apu.erg/apu.cm**2/apu.Hz/apu.s/apu.sr
            spectra = spectra.to(apu.erg/apu.cm**2/apu.cm/apu.s/apu.sr, 
                                 equivalencies=apu.spectral_density(wavfreq))[::-1]
            wavfreq = wavfreq.to(u.nm,
                                 equivalencies=apu.spectral())[::-1]
            print("WARNING: frequency domain filtering has not been tested thoroughly!!")
        else:
            raise ValueError("Invalid unit (%s) for wavfreq. Currently supported: length and frequency" % (wavfreq.unit,))

        idx = (wavfreq>wmin)*(wavfreq<wmax)
        if len(idx)<nwav_bb/5:
            warning.warn("Too little points (%i) from given spectra inside the filter range (%s - %s)" % (len(idx), wmin, wmax))
        wavfreq = wavfreq[idx]
        spectra = spectra[idx]
    except TypeError: #spectra is scalar (temperature)
        wavfreq = sp.linspace(wmin,wmax,nwav_bb)
        spectra = blackbody(spectra, wavfreq)

    spec_flt_tot = it.UnivariateSpline(wavfreq, (spectra*us(wavfreq)).value,
                                       s=0.0).integral(wavfreq[0].value,
                                                       wavfreq[-1].value)
    wav_flt_tot = it.UnivariateSpline(wavfreq, (wavfreq*us(wavfreq)).value,
                                       s=0.0).integral(wavfreq[0].value,
                                                       wavfreq[-1].value)
    flt_tot = us.integral(wavfreq[0].value,
                          wavfreq[-1].value)
    cwav = wav_flt_tot*filter_unit/flt_tot
    dwav = flt_tot*filter_unit / 2

    if full:
        return spec_flt_tot/flt_tot, cwav.to(output_unit), dwav.to(output_unit)
    else:
        return spec_flt_tot/flt_tot


def filterconversion(target_filter, temp,
                     temp_ref=9700, **kwargs):
    """Convert from one given filter (in kwargs) to a target_filter

    :param temp_ref: Temperature of zero_point in magnitudes
    """

    if len(kwargs) != 1:
        raise ValueError("One, and only one reference filter needs to be specified as kwargs (%s)" % (kwargs,))

    orig_filter, orig_value = kwargs.items()[0]
    orig_filter = orig_filter.replace('_', '.')

    ref0 = orig_value + 2.5*sp.log(applyfilter(orig_filter, temp) /
                                   applyfilter(orig_filter, temp_ref)) / sp.log(10)

    return -2.5*sp.log(applyfilter(target_filter, temp) / 
                       applyfilter(target_filter, temp_ref))/sp.log(10) + ref0
