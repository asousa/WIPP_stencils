from __future__ import division
import commands

# from partition import partition 

import matplotlib
matplotlib.use('agg')

import numpy as np
import pandas as pd
import cPickle as pickle
import gzip

from methods.index_helpers import load_Kp
import bisect
from methods.precip_model import precip_model, build_database

from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt
import os
import itertools
import random
import os
import time
import datetime as datetime
import types
import scipy.io
import matplotlib.gridspec as gridspec


from scipy import stats
from methods.xflib import xflib

import logging
import math

#----------------------------------------------------------
# 






# Worker process:
def job(tlist):
#     print "loading %d/%d"%(f_ind, total_files)
    print "doing day ", tlist[0].date()
    pwr_map = np.zeros([len(gridlons), len(gridlats), n_bands])
    
    cur_map_total = np.zeros([180, 360])
    for filetime in tlist:
#         filetime = times_todo[t]
        filename = filetime.strftime('%m_%d_%Y_%H_%M') + '.pklz'

        
#         print filename
        with gzip.open(os.path.join(data_path, filename)) as f:
            thingy = pickle.load(f)
            intime = datetime.datetime.strptime(filename[:-5],'%m_%d_%Y_%H_%M')

            # Get current Kpmax
            Kpm_index = bisect.bisect_left(Kpmtimes, intime)
            # Kpm = min(8,Kpmax[Kpm_index])  # Interpolation is only good up to kp
            Kpm = Kpmax[Kpm_index]
            # Closest Kp for which we have a stencil
            stencil_kp = unique_kp[np.argmin(np.abs(unique_kp - Kpm))]
            # print "actual kpm:",Kpm, "key kpm:", stencil_kp

#             print Kpm
            # Load squared current map:
            cur_map = thingy['cur_map']

            cur_map_total += cur_map
            # Select day and night bins:
            mltvec = xf.lon2MLT(intime, gridlons)
            mltvec_quantized = np.zeros_like(gridlons)
            mltvec_quantized[(mltvec > 6) & (mltvec <= 18)] = 12


            # Loop through each cell in the input current map, interpolate and add
            todo = np.where(np.abs(cur_map) > 0)

            for x,y in zip(todo[0], todo[1]):
                I = cur_map[x,y] # squared input current
    #             print x, y
                
                cur_mlt = int(mltvec_quantized[y])
                cur_lat = cur_map_lats[x]
                cur_lon = cur_map_lons[y]


                key = (int(np.round(10.*stencil_kp)), np.abs(cur_lat), cur_mlt)

                if precalc_stencils.has_key(key):
                    stencil = precalc_stencils[key].swapaxes(0,1)*I

                    # If southern hemisphere, flip the stencil:
                    if cur_lat < 0:
                        stencil = np.flip(stencil, axis=1)


                    # Add to respective map:
                    # lonleft  = int(y/dlon - len(stencil_lons)*dlon - 1)
                    # lonright = int(y/dlon + len(stencil_lons)*dlon)
                    
                    # lonleft  = bisect.bisect_left(gridlons, cur_lon + stencil_lons[0])
                    # lonright = bisect.bisect_left(gridlons, cur_lon + stencil_lons[-1])

                    lonleft  = int((cur_lon + stencil_lons[0])/dlon + len(gridlons)/2)
                    lonright = int((cur_lon + stencil_lons[-1])/dlon + len(gridlons)/2) + 1

                    # print np.shape(stencil), y, cur_lon, lonleft, lonright, lonright-lonleft
                    if lonleft < 0:
                        # wrap left
                        # print "left"
                        pwr_map[0:lonright,:,:]+= stencil[np.abs(lonleft):,:,:]
                        pwr_map[(len(gridlons) - np.abs(lonleft)):,:,:] += \
                                stencil[0:np.abs(lonleft),:,:]

                    elif lonright > len(gridlons):
                        # wrap right
                        # print "right"
                        pwr_map[lonleft:len(gridlons),:,:] += stencil[0:len(gridlons) - lonleft, :,:]
                        pwr_map[0:np.abs(lonright) - len(gridlons), :,:] += stencil[len(gridlons) - lonleft,:,:]

                    else:
                        # Middle
                        # print "middle"
                        pwr_map[lonleft:lonright, :,:] += stencil

                # else:
                #     print "no key at ", key

    # Total seconds per day (or partial day, if we're missing a few entries)
    divisor = len(tlist)*datetime.timedelta(hours=3).seconds 
    outdict = dict()
    # Units are either: mErg/cm^2/ev/sec (energy)
    #                   elec/cm^2/ev/sec (counts)
    # (Averaged over day)
    outdict['flux'] = pwr_map/divisor
    # Cur_map:  Time-averaged, Io^2/sec (per cell -- not per km^2!)
    outdict['cur_map'] = cur_map_total/divisor
    
#     out_filename = 'flux_' + filetime.strftime('%m_%d_%Y_%H') + '.pklz'
    out_filename = 'flux_' + tlist[0].strftime('%Y_%m_%d') + '.pklz'

    with gzip.open(os.path.join(out_path, out_filename),'wb') as file:
        pickle.dump(outdict, file, protocol=pickle.HIGHEST_PROTOCOL)
    
# ----- end of job -----


if __name__ == '__main__':



    # ------------- Settings ----------------
    # suffix = 'AE8MAX_flux_0'
    suffix = os.getenv('suffix')
    mode = os.getenv('precip_mode')
    # mode = 'energy'
    # Load the stencils ---
    unique_kp = [0.,0.3,0.7, 1.,1.3, 1.7, 2.,2.3, 2.7, 3., 3.3, 3.7, 4., 4.3, 4.7,5.,5.3,5.7,6.,6.3,6.7,7.,7.3,7.7,8.]
    # unique_kp = [0, 2, 4, 6, 8]
    inlats = np.arange(15, 56, 1)

    dlat = 1
    dlon = 1

    n_bands = 64

    # The range of days we'll do:
    starttime = datetime.datetime(2015,1,1,0,0,0)
    stoptime =   datetime.datetime(2017,1,1,0,0,0)

    data_path = '/shared/users/asousa/WIPP/lightning_power_study/outputs/GLDstats_v8/data/'
    out_path = '/shared/users/asousa/WIPP/WIPP_stencils/outputs/seasonal_precip/%s_%s'%(mode, suffix)
    db_path  = '/shared/users/asousa/WIPP/WIPP_stencils/outputs/precip_dbs'

    # ----------------------------------------
    # ------------- Misc Setup ---------------

    xf = xflib(lib_path='/shared/users/asousa/WIPP/WIPP_stencils/python/methods/libxformd.so')

    num_cores = multiprocessing.cpu_count()
    print "Num cores: ", num_cores
    
    R_E = 6371. # Km
    R2D = 180./np.pi
    D2R = np.pi/180.
    sec_in_year = 60*60*24*365.25

    if not os.path.exists(out_path):
        os.system('mkdir -p %s'%out_path)

    d = os.listdir(data_path)

    # Load Kpmax ---
    print "loading Kp"
    Ktimes, Kp_arr = load_Kp()
    Ktimes = [k + datetime.timedelta(minutes=90) for k in Ktimes]  # 3-hour bins; the original script labeled them in the middle of the bin
    Ktimes = np.array(Ktimes)
    Kp_arr = np.array(Kp_arr)

    # Get Kpmax -- max value of Kp over the last 24 hours (8 bins):
    Kpmax = np.max([Kp_arr[0:-8],Kp_arr[1:-7],Kp_arr[2:-6], Kp_arr[3:-5],
                    Kp_arr[4:-4],Kp_arr[5:-3],Kp_arr[6:-2], Kp_arr[7:-1], Kp_arr[8:]],axis=0)
    Kpmtimes = Ktimes[8:]


    files = [x for x in d if x.endswith('.pklz')]
    # intimes = [datetime.datetime.strptime(x[:-5],'%m_%d_%Y_%H_%M') for x in d if x.endswith('.pklz')]
    dtvec =   [datetime.datetime.strptime(x[:-5],'%m_%d_%Y_%H_%M') for x in files]

    # Sort by day (~ 8 entries per day)
    daydict = dict()
    for t in dtvec:
        if not daydict.has_key(t.date()):
            daydict[t.date()] = []
        daydict[t.date()].append(t)

    print "Doing %d days"%len(daydict.keys())
    stencil_lats = np.hstack([np.arange(-80, -19,dlat),(np.arange(20,80 + dlat,dlat))])
    stencil_lons = np.linspace(-14, 14, 14*2/dlon + 1)

    logscale = False
    dlon = np.abs(stencil_lons[1] - stencil_lons[0])
    dlat = np.abs(stencil_lats[1] - stencil_lats[0])

    # ----------------------------------------
    # ----- Precalculate stencils ------------
    # ----------------------------------------


    fn = os.path.join(db_path,'nightside_%s_%db.pkl'%(suffix, n_bands))
    if os.path.exists(fn):
        print "Interpolating nightside"

        with open(fn,'rb') as file:
            db = pickle.load(file)
            
            p_night = precip_model(db, mode=mode)
            p_night.precalculate(unique_kp, inlats, stencil_lats, stencil_lons, logscale=logscale)
    else:
        print "Cannot find nightside dictionary"

    fn = os.path.join(db_path,'dayside_%s_%db.pkl'%(suffix, n_bands))
    if os.path.exists(fn):
        with open(fn,'rb') as file:
            db = pickle.load(file)
            
            p_day = precip_model(db, mode=mode)
            p_day.precalculate(unique_kp, inlats, stencil_lats, stencil_lons, logscale=logscale)
    else:
        print "cannot find dayside dictionary"
    # Move the precalculated stencils from an ndarray to a dictionary --
    # key ~ [kp, inlat], data ~ [outlats, outlons, bands]
    precalc_stencils = dict()

    print "doing the dictionary thing"
    for kp_ind, kp in enumerate(unique_kp):
        for inlat_ind, inlat in enumerate(inlats):

            key = (int(np.round(10.*kp)), inlat, 0)
            precalc_stencils[key] = p_night.precalculated['data'][kp_ind, inlat_ind, :,:].squeeze()

            key = (int(np.round(10.*kp)), inlat, 12)
            precalc_stencils[key] = p_day.precalculated['data'][kp_ind, inlat_ind, :,:].squeeze()

    print "precalc dict has %d keys"%len(precalc_stencils.keys())
    # ----------------------------------------
    # ---------- Output space ----------------
    # ----------------------------------------
    gridlons = np.arange(-180, 180, dlon)
    gridlats = stencil_lats
    cell_areas = np.abs((R_E*dlat*D2R)*(R_E*dlon*D2R)*np.cos((np.abs(gridlats) + dlat/2.0)*D2R))

    cur_map_lats = np.arange(-90, 90)
    cur_map_lons = np.arange(-180,180)



    # ----------------------------------------
    # ------------ Run it ! ------------------
    # ----------------------------------------

    # Parallel jobs (single machine, multiple core)
    print "--- the main event ---"
    ret = Parallel(n_jobs=num_cores)(delayed(job)(tlist) for tlist in daydict.values())

