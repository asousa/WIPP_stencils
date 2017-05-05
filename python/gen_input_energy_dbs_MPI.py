import matplotlib
matplotlib.use('agg')

from mpi4py import MPI
import numpy as np
import pickle
import gzip
import matplotlib.pyplot as plt

import commands
import subprocess
from random import shuffle
import os
import itertools
import random
import os
import time
from scipy.integrate import nquad

import logging
import math

from methods.partition import partition
from methods.graf_iono_absorp import lon2MLT, MLT2lon, input_power_scaling
import methods.xflib as xflib

import datetime as datetime
# Constants
Hz2Rad = 2.*np.pi
D2R = np.pi/180.
H_IONO_BOTTOM = 1e5
H_IONO_TOP = 1e6
R_E = 6371e3



flash_lats = np.arange(15,55,1)
flash_lat_inds = np.arange(0, len(flash_lats))
flash_mlt = 0

dlat = 1.0 # degrees
dlon = 0.5 # degrees
max_lon = 20
max_lat = 20

f1 = 200; f2 = 30000;
num_freqs = 33
flogs = np.linspace(np.log10(f1), np.log10(f2), num_freqs)
freqs = np.round(pow(10, flogs)/10.)*10
freq_pairs = zip(freqs[0:], freqs[1:])

I0 = -1  # so we can easily scale it afterward
itime = datetime.datetime(2010,1,1,0,0,0)


# Where to save 'em
out_base ='/shared/users/asousa/WIPP/WIPP_stencils/outputs/input_energies_0.5deg'

xf = xflib.xflib(lib_path='/shared/users/asousa/WIPP/3dWIPP/python/libxformd.so')


flash_lon = MLT2lon(itime,flash_mlt,xf)


# ------------ Start MPI -------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
host = commands.getoutput("hostname")
if "cluster" in host:
    host_num = int(host[-3:])
elif "nansen" in host:
    host_num = 0

fmt_str = '[%(levelname)s: ' + '%d/'%rank + host + '] %(message)s'
logging.basicConfig(level=logging.INFO,
                    format=fmt_str)
nProcs = 1.0*comm.Get_size()


# Center of latitude and longitude bins (offset from flash location)
# clat_offsets = np.arange(-max_lat, max_lat + dlat, dlat)
clat_offsets = np.linspace(-max_lat, max_lat, (2*max_lat/dlat) + 1)
clon_offsets = np.linspace(0, max_lon, max_lon/dlon + 1) - dlon/2.
# clon_offsets = np.arange(-dlon/2., max_lon + dlon, dlon)

clat_pairs = zip(clat_offsets[0:-1], clat_offsets[1:])
clon_pairs = zip(clon_offsets[0:-1], clon_offsets[1:])
ilats = np.arange(0, len(clat_pairs))
ilons = np.arange(0, len(clon_pairs))
ifreqs = np.arange(0, len(freq_pairs))

if rank==0:
    # make output directory
    if not os.path.exists(out_base):
        os.system('mkdir -p %s'%out_base)

    print len(ilats), len(ilons), len(ifreqs)
    jobs = [(w,x,y,z) for w,x,y,z in itertools.product(flash_lat_inds, ilats, ilons, ifreqs)]
    np.random.shuffle(jobs)
    n_workers =min(len(jobs), nProcs - 1)
    partitioned_jobs = partition(jobs, n_workers)
    print len(jobs), "total jobs"
else:
    jobs = None
    partitioned_jobs   = None

comm.Barrier()

jobs = comm.bcast(jobs, root=0)
partitioned_jobs   = comm.bcast(partitioned_jobs, root=0)


recv_buffer = np.zeros(5)
send_buffer = np.zeros(5)

# Integrator params
opts = dict()
opts['epsabs']= 1.5e-8
opts['epsrel']= 1.5e-8
opts['limit']= 10

# Set root node up to receive:
if rank == 0:
    # one db per input flash
    out_dbs = [dict() for x in flash_lats]

    for d in out_dbs:
        # power output is a 3d array, freq x lat x lon
        d['pwr'] = np.zeros([len(freq_pairs), len(clat_pairs), len(clon_pairs)])

    # Each individual cell is its own job
    for i in range(len(jobs)):
        comm.Recv(recv_buffer, source=MPI.ANY_SOURCE)
        # print recv_buffer

        # To file the returned data, we need: flash lat, ifreq, ilat, ilon
        inlat, ifreq, ilat, ilon, pwr = recv_buffer[0:5]

        out_dbs[int(inlat)]['pwr'][int(ifreq), int(ilat), int(ilon)] = pwr


else:
    # Queue up the workers
    if ((rank -1) < len(partitioned_jobs)):    
        print "process ", rank, "doing ", len(partitioned_jobs[rank-1]), "entries"    
        for flash_lat_ind, ilat, ilon, ifreq in partitioned_jobs[rank - 1]:
            flash_lat = flash_lats[flash_lat_ind]
            flash_pos_mag = [1, flash_lat, flash_lon]
            flash_pos_sm = xf.rllmag2sm(flash_pos_mag, itime)

            f1, f2 = freq_pairs[ifreq]
            lat1, lat2 = clat_pairs[ilat]
            lat1 += flash_lat; lat2 += flash_lat;
            lon1, lon2 = clon_pairs[ilon]
            lon1 += flash_lon; lon2 += flash_lon;

            f_center = (f1 + f2)/2.
            clat = (lat1 + lat2)/2.
            w1 = Hz2Rad*f1
            w2 = Hz2Rad*f2
            w   = Hz2Rad*(f1 + f2)/2.
            dw = np.abs(f1 - f2)*Hz2Rad

            # Integrand to call with integration routine
            def integrand(inlat, inlon, inw, itime, I0, flash_pos_sm_in, itime_in):
                mlt = lon2MLT(itime, inlon, xf);
                # print "lon:", inlon, "MLT:",mlt
                tmp_coords = [1, inlat, inlon];
                x_sm = xf.rllmag2sm(tmp_coords, itime_in);

                pwr = input_power_scaling(flash_pos_sm_in, x_sm, inlat, inw, I0, mlt, xf);
                return pwr*(R_E + H_IONO_TOP)*D2R*(R_E + H_IONO_TOP)*np.cos(D2R*inlat)*D2R

            ranges = [[lat1, lat2], [lon1, lon2]]
            integ = nquad(integrand, ranges, args=[w, itime, I0, flash_pos_sm, itime], opts=opts, full_output=False)
            pwr = integ[0]*dw
            # pwr = rank

            send_buffer[0:5] = [flash_lat_ind, ifreq, ilat, ilon, pwr]
            comm.Send(send_buffer, 0)
    else:
        print "process ", rank, " idling"

comm.Barrier()

if rank == 0:
    # Save 'em up
    for ind, flash_lat in enumerate(flash_lats):
        fname = os.path.join(out_base,'input_energy_%d_%d.pklz'%(flash_lat, flash_mlt))
        print "saving ", fname
        d = out_dbs[ind]
        d['flash_lat'] = flash_lat
        d['flash_mlt'] = flash_mlt
        d['clats'] = clat_offsets[0:-1] + flash_lat + dlat/2.
        d['clons'] = clon_offsets[0:-1] + flash_lon + dlon/2.
        d['cfreqs'] = (freqs[0:-1] + freqs[1:])/2.
        d['I0'] = I0

        with gzip.open(fname,'wb') as file:
            pickle.dump(d, file)
        print "finished"




