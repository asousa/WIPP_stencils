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

class flux_params(ct.Structure):
    _fields_ = [('NUM_E', ct.c_size_t),
                ('NUM_T', ct.c_size_t),
                ('dt',    ct.c_double),
                ('DE_EXP',ct.c_double),
                ('E_EXP_BOT', ct.c_double),
                ('E_EXP_TOP', ct.c_double),
                ('alpha_dist', ct.c_int),
                ('flux_dist', ct.c_int),
                ('n_JL', ct.c_size_t),
                ('n_JE', ct.c_size_t)
                ]

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



def calc_stencil_MPI(crossing_dir=None,
                    power_dir    = None,
                    out_dir = None,
                    alpha_dist = None,
                    flux_dist = None,
                    flash_lat=None,
                    mlt = None,
                    max_dist=None,
                    I0= None,
                    d_lon = None,
                    num_lons=None,
                    f_low=None, f_hi=None,
                    itime = datetime.datetime(2010,1,1,0,0,0)):

    # Parameters
    Emin = 1.0e1 # 1ev
    Emax = 1.0e8 # 10Mev
    NUM_E = 512
    SCATTERING_RES_MODES = 5
    E_BANDWIDTH = 10 #0.3

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
    
    # Define arguments -- equivalent to the arguments in WIPP_stencil.h
    calc_scattering_c = lib.calc_scattering
    calc_scattering_c.restype = None
    calc_scattering_c.argtypes =    [ndpointer(ct.c_double, flags='C_CONTIGUOUS'),
                                     ct.c_size_t,
                                     ndpointer(ct.c_double, flags="C_CONTIGUOUS"),
                                     ct.Structure, ct.Structure,
                                     ndpointer(ct.c_double, flags="C_CONTIGUOUS"),
                                     ndpointer(ct.c_double, flags="C_CONTIGUOUS")]
    
    # Define arguments -- equivalent to the arguments in WIPP_stencil.h
    calc_flux_c = lib.calc_flux
    calc_flux_c.restype = None
    calc_flux_c.argtypes =          [ndpointer(ct.c_double,flags='C_CONTIGUOUS'),
                                    ct.Structure,
                                    ndpointer(ct.c_double,flags='C_CONTIGUOUS'),
                                    ct.c_double,
                                    ndpointer(ct.c_double,flags='C_CONTIGUOUS')]


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
    # Passing entries per fieldline now, instead of as one whole chunk.
    recv_buffer = np.zeros([2, num_lons, len(E_tot_arr), len(time) + 1])
    send_buffer = np.zeros([2, num_lons, len(E_tot_arr), len(time) + 1])
    status = MPI.Status()
    # Set root node up to receive:
    if rank == 0:   
        print "nProcs: ", nProcs
        print "nJobs: ", len(jobs)

        # Initialize output space
        da_N = np.zeros(buffer_dims[1:])
        da_S = np.zeros_like(da_N) 
        received_count = 0

        for i in range(len(jobs)*len(Lshells)):
            comm.Recv(recv_buffer, source=MPI.ANY_SOURCE, status=status)
            fl_ind = status.Get_tag()
            # print "received fl_ind ", fl_ind
            # print "Receiving: N: (", np.max(recv_buffer[0]), np.min(recv_buffer[0]), "), S: (", np.max(recv_buffer[1]), np.min(recv_buffer[1]), ")"

            da_N[fl_ind, :,:,:] += recv_buffer[0,:,:,:]
            da_S[fl_ind, :,:,:] += recv_buffer[1,:,:,:]
            received_count += 1
            print "finished %d/%d"%(received_count, len(jobs)*len(Lshells))
            print "N: (", np.max(da_N), np.min(da_N), "), S: (", np.max(da_S), np.min(da_S), ")"
    else:
        # Queue up the workers
        if ((rank -1) < len(partitioned_jobs)):
            for lat_pair, freq_pair in partitioned_jobs[rank - 1]:
                print "process ", rank, " : ", lat_pair, freq_pair

                center_lat = (lat_pair[0] + lat_pair[1] )/2.
                center_freq = (freq_pair[0] + freq_pair[1])/2.
                # Old way
                # pwr_key = (center_freq, center_lat)
                # inp_pwrs = np.asarray(pwr_db[pwr_key])*pow(np.abs(I0)/10000., 2)  # rescale to new I0

                # New way
                pwr_freq_ind = np.argmin(np.abs(pwr_db['cfreqs'] - center_freq))
                pwr_lat_ind  = np.argmin(np.abs(pwr_db['clats'] - center_lat))
                inp_pwrs = pwr_db['pwr'][pwr_freq_ind, pwr_lat_ind, 0:num_lons]*pow(I0, 2.)

                # print pwr_freq_ind, pwr_db['cfreqs'][pwr_freq_ind], center_freq
                # print pwr_lat_ind, pwr_db['clats'][pwr_lat_ind], center_lat
                
                # Temporary band aid: Crossing detection was done with 1-deg bins,
                # input power was calculated with 0.25-deg bins.        
                # inp_pwr = np.sum(pwr_db[pwr_key][0:4])
                # inp_pwrs = np.asarray(pwr_db[pwr_key])*pow(np.abs(I0)/10000., 2)  # rescale to new I0

                # print "input powers: ", inp_pwrs

                # print np.shape(inp_pwrs)
                # inp_pwr = pwr_db[pwr_key][0]*pow(np.abs(I0)/10000., 2)  # rescale to new I0
                num_lons = min(num_lons, len(inp_pwrs))

                # Load crossings 
                crossing_fname = os.path.join(crossing_dir,'crossing_log_lat_%d-%d_f_%d-%d.pklz'%(
                    lat_pair[0], lat_pair[1], freq_pair[0], freq_pair[1]))
                with gzip.open(crossing_fname, 'rb') as file:
                    tmp = pickle.load(file)
                # crossings = tmp['fieldlines']

                # # Calc scattering
                # tmp_N, tmp_S = calc_pitch_angle_change(inp_pwr, crossings, params)

                # tmp_N = np.zeros(buffer_dims[1:])
                # tmp_S = np.zeros(buffer_dims[1:])
                
                # print np.shape(tmp_N);
                # Loop over fieldlines
                for fl_ind, fl in enumerate(tmp['fieldlines']):

                    tmp_N = np.zeros(buffer_dims[2:])
                    tmp_S = np.zeros(buffer_dims[2:])

                    # print "L: ", fl['L']
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
                        # print crossing_list
                        # tmpN = np.zeros([len(E_tot_arr), len(time) + 1])
                        # tmpS = np.zeros([len(E_tot_arr), len(time) + 1])
                        
                        calc_scattering_c(crossing_list, np.shape(crossing_list)[0],
                                        inp_pwrs, EA, params,
                                        tmp_N, tmp_S)

                                        # tmp_N[fl_ind,:,:,:], tmp_S[fl_ind,:,:,:])
                    # print "Sending: N: (", np.max(tmp_N), np.min(tmp_N), "), S: (", np.max(tmp_S), np.min(tmp_S), ")"
                        # da_N[fl_ind,:,:] += tmpN
                        # da_S[fl_ind,:,:] += tmpS
                    # Send iiiiiiit
                    send_buffer[0:2] = np.array([tmp_N, tmp_S])
                    comm.Send(send_buffer, 0, tag=fl_ind)
        else:
            print "process ", rank, " idling"
    comm.Barrier()

    if rank == 0:
        # And done -- save it.
        print "Saving scattering file..."
        # (this version for Python, since pickle is a shit about ctypes structures)
        pyparams = dict()
        pyparams['Emin'] = Emin
        pyparams['Emax'] = Emax
        pyparams['NUM_E'] = NUM_E
        pyparams['E_tot_arr'] = E_tot_arr
        pyparams['v_tot_arr'] = v_tot_arr
        pyparams['Lshells'] = Lshells
        pyparams['tvec'] = time


        # Save full spectra along center longitude only
        outfile = os.path.join(out_dir,'scattering_inlat_%d.pklz'%flash_lat)
        outdict = dict()
        outdict['da_N'] = da_N[:,0,:,:] 
        outdict['da_S'] = da_S[:,0,:,:]
        outdict['params'] = pyparams

        with gzip.open(outfile,'wb') as file:
            pickle.dump(outdict,file)


        # --------------------------------------------
        # Flux calculation
        # --------------------------------------------
        flux_root = '/shared/users/asousa/WIPP/WIPP_stencils/data/'

        for fluxfile_prefix in ['AE8MIN','AE8MAX']:
            fluxfile = os.path.join(flux_root,'%s_integral_flux.dat'%fluxfile_prefix)
            print "Starting flux calculation"
            Jdata = np.loadtxt(fluxfile)

            JL = Jdata[1:,0]  # L-shells in J-file
            JE = Jdata[0,1:]  # Energies in J-file

            f_params = flux_params()
            f_params.NUM_E          = NUM_E
            f_params.NUM_T          = len(time)
            f_params.dt             = time[1] - time[0]
            f_params.DE_EXP         = np.log10(E_tot_arr[1]) - np.log10(E_tot_arr[0])
            f_params.E_EXP_BOT      = np.log10(Emin)
            f_params.E_EXP_TOP      = np.log10(Emax)
            f_params.alpha_dist     = alpha_dist
            f_params.flux_dist      = flux_dist
            f_params.n_JL           = np.shape(Jdata)[0]
            f_params.n_JE           = np.shape(Jdata)[1]
            
            pyparams['n_JL'] = np.shape(Jdata)[0]
            pyparams['n_JE'] = np.shape(Jdata)[1]
            pyparams['DE_EXP'] = np.log10(pyparams['E_tot_arr'][1]) - np.log10(pyparams['E_tot_arr'][0])
            pyparams['E_EXP_BOT'] = np.log10(pyparams['Emin'])
            pyparams['E_EXP_TOP'] = np.log10(pyparams['Emax'])
            pyparams['alpha_dist'] = alpha_dist
            pyparams['flux_dist']  = flux_dist
            pyparams['fluxfile'] = fluxfile
            pyparams['dt'] = pyparams['tvec'][1] - pyparams['tvec'][0]


            phi_N = np.zeros_like(da_N)
            phi_S = np.zeros_like(da_S)

            for L_ind, L in enumerate(pyparams['Lshells']):
                for lon_ind in range(np.shape(da_N)[1]):
                    print "Calculating at ", L, lon_ind
                    calc_flux_c(da_N[L_ind, lon_ind, :, :], f_params, Jdata, L, phi_N[L_ind, lon_ind, :, :])
                    calc_flux_c(da_S[L_ind, lon_ind, :, :], f_params, Jdata, L, phi_S[L_ind, lon_ind, :, :])

            print "saving reduced phi file..."
            outfile = os.path.join(out_dir,'phi_inlat_%d_%s.pklz'%(flash_lat, fluxfile_prefix))
            outdict = dict()
            # Summed over time:
            outdict['phi_N_sum'] = np.sum(phi_N, axis=-1)
            outdict['phi_S_sum'] = np.sum(phi_S, axis=-1)

            # full-resolution, center longitude only
            outdict['phi_N_full'] = phi_N[:,0,:,:]
            outdict['phi_S_full'] = phi_S[:,0,:,:]
            outdict['params'] = pyparams


            with gzip.open(outfile,'wb') as file:
                pickle.dump(outdict,file)



if __name__ == "__main__":

    rootdir ='/shared/users/asousa/WIPP/WIPP_stencils/'
    # crossing_dir = os.path.join(rootdir,'outputs','crossings9_quick','nightside','ngo_v2','python_data')
    crossing_dir = os.path.join(rootdir,'outputs','crossings_20f','nightside','ngo_v2','python_data')
    power_dir    = os.path.join(rootdir,'outputs','input_energies')
    out_dir      = os.path.join(rootdir,'outputs','stencils','nightside','stencil_testing')

    # fluxfile = '/shared/users/asousa/WIPP/WIPP_stencils/data/AE8MAX_integral_flux.dat'
    alpha_dist = 0  # 0: Sine / ramp  1: Square
    flux_dist = 0   # 0: fluxfile  1: Suprathermal  2: Flat

    inlats =[20, 30, 40, 50]

    for inlat in inlats:
        print "Doing flash lat at ", inlat, "deg"
        calc_stencil_MPI(
            crossing_dir = crossing_dir,
            power_dir = power_dir,
            out_dir = out_dir,
            # fluxfile = fluxfile,
            alpha_dist = alpha_dist,
            flux_dist = flux_dist,
            flash_lat=inlat,
            mlt = 0,
            max_dist=1000,
            I0=-10000,
            d_lon = 1,
            num_lons=15,
            f_low=200, f_hi=30000,
            itime = datetime.datetime(2010,1,1,0,0,0))


