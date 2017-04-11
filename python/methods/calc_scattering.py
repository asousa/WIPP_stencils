import matplotlib
matplotlib.use('agg')
import numpy as np
import pandas as pd
import pickle
import gzip
from scipy import interpolate
import matplotlib.pyplot as plt
import os
import itertools
import random
import os
import time
import datetime as datetime
from raytracer_utils import read_rayfile, read_damp
from scipy.spatial import Delaunay
from scipy.integrate import nquad
from scipy import stats
import xflib
from graf_iono_absorp import total_input_power, lon2MLT, MLT2lon, input_power_scaling
import logging
import math
from calc_pitch_angle_change import calc_pitch_angle_change

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km



def calc_scattering(crossing_dir='/shared/users/asousa/WIPP/WIPP_stencils/outputs/crossings/nightside/kp0/python_data',
                    power_dir    = '/shared/users/asousa/WIPP/lightning_power_study/outputs/input_powers',
                    flash_lat=40,
                    mlt = 0,
                    max_dist=112,
                    I0=-10000,
                    f_low=200, f_hi=230,
                    itime = datetime.datetime(2010,1,1,0,0,0)):

    # Parameters
    Emin = 1.0e1
    Emax = 1.0e8
    NUM_E = 512
    SCATTERING_RES_MODES = 5
    E_BANDWIDTH = 0.3
    
    # Constants
    Hz2Rad = 2.*np.pi
    D2R = np.pi/180.
    H_IONO_BOTTOM = 1e5
    H_IONO_TOP = 1e6
    R_E = 6371e3
    M_EL = 9.1e-31
    E_EL = 5.105396765648739E5 
    MU0  = np.pi*4e-7
    EPS0 = 8.854E-12
    C    = 2.997956376932163e8




    d = os.listdir(crossing_dir)
    avail_files = [x[:-5] for x in d if x.startswith('crossing_log') and x.endswith('.pklz')]

    lat_pairs = np.array([x.split('_')[3].split('-') for x in avail_files], dtype=int)
    freq_pairs = np.array([x.split('_')[5].split('-') for x in avail_files], dtype=int)
    unique_lats  = np.unique(lat_pairs)
    unique_freqs = np.unique(freq_pairs)

    # Select only ray pairs of interest
    raydists = np.array([haversine_np(0,la, 0, flash_lat) for la in unique_lats])
    masked_lats = unique_lats[raydists <= max_dist]
    masked_freqs = unique_freqs[(unique_freqs >= f_low) & (unique_freqs <= f_hi)]

    lat_pairs = zip(masked_lats[:-1], masked_lats[1:])
    freq_pairs = zip(masked_freqs[:-1], masked_freqs[1:])

    # aight -- now do these.
    # print lat_pairs
    # print freq_pairs

    # Load the first pair in the list, to get the working system axes
    fname = os.path.join(crossing_dir,'crossing_log_lat_%d-%d_f_%d-%d.pklz'%(
        lat_pairs[0][0], lat_pairs[0][1], freq_pairs[0][0], freq_pairs[0][1]))

    with gzip.open(fname,'rb') as file:
        tmpfile = pickle.load(file)

    # L axis, time axis
    Lshells = tmpfile['Lshells']
    time = tmpfile['time']

    # Energy axis
    E_tot_arr = pow(10, np.linspace(np.log10(Emin), np.log10(Emax), NUM_E))
    v_tot_arr = C*np.sqrt(1.0 - pow( (E_EL/(E_EL + E_tot_arr)), 2.))

    # output space -- 3d array, [L-shell, energy, time]
    # Last column on time axis ~ scattering beyond our time bins
    da_N = np.zeros([len(Lshells), len(E_tot_arr), len(time) + 1])
    da_S = np.zeros_like(da_N)

    # Load the input power dictionary (pre-calculated since the integral is *s l o w*)

    pwrfile = os.path.join(power_dir, 'input_energy_%d_%d.pklz'%(flash_lat, mlt))

    with gzip.open(pwrfile,'rb') as file:
        pwr_db = pickle.load(file)

    # print pwr_db.keys()
    # Now, loop through each ray set (lat pair, freq pair), and
    # calculate the pitch-angle change at each crossing

    lon_offset = 0 

    # All the constants and parameters we need to pass inward
    params = dict()
    params['SCATTERING_RES_MODES'] = SCATTERING_RES_MODES
    params['E_BANDWIDTH'] = E_BANDWIDTH
    params['Emin'] = Emin
    params['Emax'] = Emax
    params['NUM_E'] = NUM_E
    params['E_tot_arr'] = E_tot_arr
    params['v_tot_arr'] = v_tot_arr
    params['Lshells'] = Lshells
    params['tvec'] = time



    for lat1, lat2 in lat_pairs:
        center_lat = (lat1 + lat2)/2.

        for f1, f2 in freq_pairs:
            center_freq = (f1 + f2)/2.
            pwr_key = (center_freq, center_lat)
            if pwr_key not in pwr_db:
                print "failed to load input power"
            else:

                # Get input power
                inp_pwr = pwr_db[pwr_key][lon_offset]        

                # Load crossing file
                crossing_fname = os.path.join(crossing_dir,'crossing_log_lat_%d-%d_f_%d-%d.pklz'%(
                    lat1, lat2, f1, f2))
                print "loading ", crossing_fname
                with gzip.open(crossing_fname, 'rb') as file:
                    tmp = pickle.load(file)

                crossings = tmp['fieldlines']

                calc_pitch_angle_change(inp_pwr, crossings, da_N, da_S, params)

                print np.max(da_N), np.max(da_S)
if __name__ == "__main__":
    calc_scattering()

