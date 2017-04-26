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
from random import shuffle
# from raytracer_utils import read_rayfile, read_damp
from scipy.spatial import Delaunay
from scipy.integrate import nquad
from scipy import stats
# import xflib
# from graf_iono_absorp import total_input_power, lon2MLT, MLT2lon, input_power_scaling
import logging
import math
from methods.calc_pitch_angle_change import calc_pitch_angle_change

from methods.partition import partition
import commands
from mpi4py import MPI

import ctypes as ct
from numpy.ctypeslib import ndpointer


class EA_args(ct.Structure):
    _fields_ = [('lat',      ct.c_double),
                ('alpha_eq', ct.c_double),
                ('stixP',    ct.c_double),
                ('stixR',    ct.c_double),
                ('stixL',    ct.c_double),
                ('alpha_lc', ct.c_double),
                ('wh',       ct.c_double),
                ('ds',       ct.c_double),
                ('dv_para_ds', ct.c_double),
                ('dwh_ds',   ct.c_double),
                ('ftc_n',   ct.c_double),
                ('ftc_s',   ct.c_double)
                ]

class scattering_params(ct.Structure):
    _fields_ = [('NUM_E', ct.c_size_t),
                ('NUM_T', ct.c_size_t),
                ('dt',    ct.c_double),
                ('DE_EXP',ct.c_double),
                ('E_EXP_BOT', ct.c_double),
                ('E_EXP_TOP', ct.c_double),
                ('E_BANDWIDTH', ct.c_double),
                ('SCATTERING_RES_MODES', ct.c_int),
                ('num_lons', ct.c_int)]
                # ('v_tot_arr', ct.POINTER(ct.c_double)),

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



def calc_scattering_MPI(crossing_dir=None,
                    power_dir    = None,
                    out_dir = None,
                    flash_lat=35,
                    mlt = 0,
                    max_dist=200,
                    I0=-10000,
                    d_lon = 0.25,
                    num_lons=5,
                    f_low=200, f_hi=500,
                    L_low = 1, L_hi = 10,
                    itime = datetime.datetime(2010,1,1,0,0,0)):

    # Parameters
    Emin = 1.0e1 # 1ev
    Emax = 1.0e8 # 10Mev
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

    # -------------- set up MPI -----------------------------
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    host = commands.getoutput("hostname")
    nProcs = 1.0*comm.Get_size()

    # -------------- set up C module ------------------------
    lib_path ='/shared/users/asousa/WIPP/WIPP_stencils/c/libwipp.so'
    ct.cdll.LoadLibrary(lib_path)
    lib = ct.CDLL(lib_path)
    # The function
    calc_scattering_c = lib.calc_scattering

    # Define arguments -- equivalent to the arguments in WIPP_stencil.h
    # ()
    calc_scattering_c.restype = None
    calc_scattering_c.argtypes =    [ndpointer(ct.c_double, flags='C_CONTIGUOUS'),
                                     ct.c_size_t,
                                     ndpointer(ct.c_double, flags="C_CONTIGUOUS"),
                                     ct.Structure, ct.Structure,
                                     ndpointer(ct.c_double, flags="C_CONTIGUOUS"),
                                     ndpointer(ct.c_double, flags="C_CONTIGUOUS")]




    if rank == 0:
        # Set up output directory:
        if not os.path.exists(out_dir):
            os.system('mkdir -p %s'%out_dir)


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
        print "Getting axes"
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
        # da_N = np.zeros([len(Lshells), len(E_tot_arr), len(time) + 1])
        # da_S = np.zeros_like(da_N)

        # Load the input power dictionary (pre-calculated since the integral is *s l o w*)
        print "Loading power file"
        pwrfile = os.path.join(power_dir, 'input_energy_%d_%d.pklz'%(flash_lat, mlt))

        with gzip.open(pwrfile,'rb') as file:
            pwr_db = pickle.load(file)

        # print pwr_db.keys()
        # Now, loop through each ray set (lat pair, freq pair), and
        # calculate the pitch-angle change at each crossing

        lon_offset = 0 

        # # All the constants and parameters we need to pass inward
        # params = dict()
        # params['SCATTERING_RES_MODES'] = SCATTERING_RES_MODES
        # params['E_BANDWIDTH'] = E_BANDWIDTH
        # params['Emin'] = Emin
        # params['Emax'] = Emax
        # params['NUM_E'] = NUM_E
        # params['E_tot_arr'] = E_tot_arr
        # params['v_tot_arr'] = v_tot_arr
        # params['Lshells'] = Lshells
        # params['tvec'] = time
        # params['L_low'] = L_low
        # params['L_hi'] = L_hi
        # params['PWR_THRESHOLD'] = PWR_THRESHOLD




        # All the steps to do:
        jobs = itertools.product(lat_pairs, freq_pairs)
        jobs = [k for k in jobs]
        shuffle(jobs)

        n_workers =min(len(jobs), nProcs - 1)
        partitioned_jobs = partition(jobs, n_workers)
        # print [k for k in jobs]

        buffer_dims = [2, len(Lshells), num_lons, len(E_tot_arr), len(time) + 1]

    else:
        # params = None
        jobs = None
        partitioned_jobs = None
        pwr_db = None
        buffer_dims = None
        v_tot_arr = None
        E_tot_arr = None
        time = None


    # params = comm.bcast(params, root=0)
    jobs = comm.bcast(jobs, root=0)
    partitioned_jobs = comm.bcast(partitioned_jobs, root=0)
    pwr_db = comm.bcast(pwr_db, root=0)
    buffer_dims = comm.bcast(buffer_dims, root=0)
    v_tot_arr = comm.bcast(v_tot_arr, root=0)
    E_tot_arr = comm.bcast(E_tot_arr, root=0)
    time = comm.bcast(time, root=0)

    params = scattering_params()
    params.NUM_E = NUM_E
    params.NUM_T = len(time) + 1
    params.dt    = time[1] - time[0]
    params.DE_EXP = ((np.log10(Emax) - np.log10(Emin))/(NUM_E))
    params.E_EXP_BOT = np.log10(Emin)
    params.E_EXP_TOP = np.log10(Emax)
    # params.v_tot_arr = np.ctypeslib.as_ctypes(v_tot_arr)
    params.E_BANDWIDTH = 1
    params.SCATTERING_RES_MODES = SCATTERING_RES_MODES
    params.num_lons = num_lons



    # print "node ", rank, jobs

    recv_buffer = np.zeros(buffer_dims)
    send_buffer = np.zeros(buffer_dims)

    # Set root node up to receive:
    if rank == 0:   
        print "nProcs: ", nProcs
        print "nJobs: ", len(jobs)

        # Initialize output space
        da_N = np.zeros(buffer_dims[1:])
        da_S = np.zeros_like(da_N) 
        received_count = 0

        for i in range(len(jobs)):
            comm.Recv(recv_buffer, source=MPI.ANY_SOURCE)
            da_N += recv_buffer[0]
            da_S += recv_buffer[1]
            received_count += 1
            print "finished %d/%d"%(received_count, len(jobs))
            print np.max(da_N), np.max(da_S)
    else:
        # Queue up the workers
        if ((rank -1) < len(partitioned_jobs)):
            for lat_pair, freq_pair in partitioned_jobs[rank - 1]:
                print "process ", rank, " : ", lat_pair, freq_pair

                center_lat = (lat_pair[0] + lat_pair[1] )/2.
                center_freq = (freq_pair[0] + freq_pair[1])/2.
                pwr_key = (center_freq, center_lat)
                # Temporary band aid: Crossing detection was done with 1-deg bins,
                # input power was calculated with 0.25-deg bins.        
                # inp_pwr = np.sum(pwr_db[pwr_key][0:4])
                inp_pwrs = np.asarray(pwr_db[pwr_key])*pow(np.abs(I0)/10000., 2)  # rescale to new I0

                print np.shape(inp_pwrs)
                # inp_pwr = pwr_db[pwr_key][0]*pow(np.abs(I0)/10000., 2)  # rescale to new I0

                # Load crossings 
                crossing_fname = os.path.join(crossing_dir,'crossing_log_lat_%d-%d_f_%d-%d.pklz'%(
                    lat_pair[0], lat_pair[1], freq_pair[0], freq_pair[1]))
                with gzip.open(crossing_fname, 'rb') as file:
                    tmp = pickle.load(file)
                # crossings = tmp['fieldlines']

                # # Calc scattering
                # tmp_N, tmp_S = calc_pitch_angle_change(inp_pwr, crossings, params)

                tmp_N = np.zeros(buffer_dims[1:])
                tmp_S = np.zeros(buffer_dims[1:])
                
                print np.shape(tmp_N);
                # Loop over fieldlines
                for fl_ind, fl in enumerate(tmp['fieldlines']):
                    print "L: ", fl['L']
                    # print fl['hit_counts']
                    hitlist = np.where(fl['hit_counts'] > 0)[0]  # indexes of lats worth doing
                    # print "hits at %d of %d lats"%(len(hitlist), len(fl['hit_counts']))
                    # Loop over latitudes with crossings
                    for hi, h in enumerate(hitlist):
                        EA = EA_args()
                        EA.lat        = ct.c_double(fl['lat'][h])
                        EA.alpha_eq   = ct.c_double(fl['alpha_eq'])
                        EA.stixP      = ct.c_double(fl['stixP'][h])
                        EA.stixR      = ct.c_double(fl['stixR'][h])
                        EA.stixL      = ct.c_double(fl['stixL'][h])
                        EA.alpha_lc   = ct.c_double(fl['alpha_lc'][h])
                        EA.wh         = ct.c_double(fl['wh'][h])
                        EA.ds         = ct.c_double(fl['ds'][h])
                        EA.dv_para_ds = ct.c_double(fl['dv_para_ds'][h])
                        EA.dwh_ds     = ct.c_double(fl['dwh_ds'][h])
                        EA.ftc_n      = ct.c_double(fl['ftc_n'][h])
                        EA.ftc_s      = ct.c_double(fl['ftc_s'][h])

                        crossing_list = np.array(fl['crossings'][h])

                        # tmpN = np.zeros([len(E_tot_arr), len(time) + 1])
                        # tmpS = np.zeros([len(E_tot_arr), len(time) + 1])
                        
                        calc_scattering_c(crossing_list, np.shape(crossing_list)[0],
                                        inp_pwrs, EA, params,
                                        tmp_N[fl_ind,:,:,:], tmp_S[fl_ind,:,:,:])

                        # da_N[fl_ind,:,:] += tmpN
                        # da_S[fl_ind,:,:] += tmpS

                # Send iiiiiiit
                send_buffer[0:2] = np.array([tmp_N, tmp_S])
                comm.Send(send_buffer, 0)
        else:
            print "process ", rank, " idling"
    comm.Barrier()

    if rank == 0:
        # And done -- save it.
        print "Saving..."
        # (this version for Python, since pickle is a shit about ctypes structures)
        pyparams = dict()
        pyparams['Emin'] = Emin
        pyparams['Emax'] = Emax
        pyparams['NUM_E'] = NUM_E
        pyparams['E_tot_arr'] = E_tot_arr
        pyparams['v_tot_arr'] = v_tot_arr
        pyparams['Lshells'] = Lshells
        pyparams['tvec'] = time
        pyparams['L_low'] = L_low
        pyparams['L_hi'] = L_hi




        outfile = os.path.join(out_dir,'scattering_inlat_%d.pklz'%flash_lat)
        outdict = dict()
        outdict['da_N'] = da_N
        outdict['da_S'] = da_S
        outdict['params'] = pyparams

        with gzip.open(outfile,'wb') as file:
            pickle.dump(outdict,file)


    # for lat1, lat2 in lat_pairs:
    #     center_lat = (lat1 + lat2)/2.

    #     for f1, f2 in freq_pairs:
    #         center_freq = (f1 + f2)/2.
    #         pwr_key = (center_freq, center_lat)
    #         if pwr_key not in pwr_db:
    #             print "failed to load input power"
    #         else:

    #             # Get input power
    #             # inp_pwr = pwr_db[pwr_key][lon_offset]
    #             # Temporary band aid: Crossing detection was done with 1-deg bins,
    #             # input power was calculated with 0.25-deg bins.        
    #             inp_pwr = np.sum(pwr_db[pwr_key][0:4])        
    #             # Load crossing file
    #             crossing_fname = os.path.join(crossing_dir,'crossing_log_lat_%d-%d_f_%d-%d.pklz'%(
    #                 lat1, lat2, f1, f2))
    #             print "loading ", crossing_fname
    #             with gzip.open(crossing_fname, 'rb') as file:
    #                 tmp = pickle.load(file)

    #             crossings = tmp['fieldlines']

    #             tmp_N, tmp_S = calc_pitch_angle_change(inp_pwr, crossings, params)
    #             da_N += tmp_N
    #             da_S += tmp_S
    #             print np.max(da_N), np.max(da_S)

    # # And done -- save it.
    # print "Saving..."
    # outfile = os.path.join(out_dir,'scattering_inlat_%d.pklz'%flash_lat)
    # outdict = dict()
    # outdict['da_N'] = da_N
    # outdict['da_S'] = da_S
    # outdict['params'] = params

    # with gzip.open(outfile,'wb') as file:
    #     pickle.dump(outdict,file)

    # print "Done"


if __name__ == "__main__":
    calc_scattering_MPI(crossing_dir='/shared/users/asousa/WIPP/WIPP_stencils/outputs/crossings_ngo_psi_fixing/nightside/ngo_v2/python_data',
                    power_dir    = '/shared/users/asousa/WIPP/lightning_power_study/outputs/input_powers',
                    out_dir = '/shared/users/asousa/WIPP/WIPP_stencils/outputs/scattering/nightside/ngo_psi_fixing_blerg/',
                    flash_lat=35,
                    mlt = 0,
                    max_dist=1000,
                    I0=-100000,
                    d_lon = 0.25,
                    num_lons=40,
                    f_low=200, f_hi=30000,
                    L_low = 1, L_hi = 10,
                    itime = datetime.datetime(2010,1,1,0,0,0))


