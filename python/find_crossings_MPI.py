import numpy as np
from scipy import interpolate
# import matplotlib.pyplot as plt
import os
import itertools
from methods.partition import partition
import time
import datetime as dt
import sys
from methods.index_helpers import load_TS_params
from methods.index_helpers import load_Dst
from methods.index_helpers import load_Kp
from methods.index_helpers import load_ae
# from spacepy import coordinates as coord
# from spacepy.time import Ticktock

import methods.xflib  # Fortran xform-double library (coordinate transforms)
import bisect

import gzip
import pickle
import commands
import subprocess
from mpi4py import MPI

from methods.find_crossings import find_crossings

project_root = '/shared/users/asousa/WIPP/WIPP_stencils/'

# ---------------------- Constants -------------------------
R_E = 6371.0    # km
R2D = 180./np.pi
D2R = np.pi/180.

# ------------------ Simulation params ---------------------

# Simulation time
ray_datenum = dt.datetime(2010, 01, 01, 0, 00, 00);

# ray lats:
ray_lat_spacing = 1
ray_lats = np.arange(12, 61, ray_lat_spacing)
# ray_lats = np.arange(20,25,ray_lat_spacing)

tmax = 20
dt = 0.02


ray_lat_pairs = zip(ray_lats[0:], ray_lats[1:])
# Frequencies
f1 = 200; f2 = 30000;
num_freqs = 33
flogs = np.linspace(np.log10(f1), np.log10(f2), num_freqs)
freqs = np.round(pow(10, flogs)/10.)*10
freq_pairs = zip(freqs[0:], freqs[1:])

# freq_pairs = freq_pairs[0:1]
Llims = [2,4] #[1.2,7]
Lstep = 0.5
out_Lsh = np.arange(Llims[0], Llims[1], Lstep)
out_lat = np.round(10.0*np.arccos(np.sqrt(1./out_Lsh))*R2D)/10.0

# out_lon = [0]

offset_lon_spacing = 1

dlat_fieldline = 0.25 #1     # degree spacing between EA segments
model_number = 0        # b-field model (0 = dipole, 1 = IGRF)
num_freq_steps = 0      # number of interpolating steps between 
                        # each guide frequency.
                        # 0 does 1hz increments.
damp_threshold = 0.1 # Value below which we ignore crossings

# vec_ind = 0     # Which set of default params to use for the gcpm model
# Mean parameter vals for set Kp:
Kpvec = ['ngo_v2']

nightday = 'nightside'




# ray_input_directory_root = '/shared/users/asousa/WIPP/rays/2d/%s/mode6/'%nightday
ray_input_directory_root = '/shared/users/asousa/WIPP/rays/2d/%s/'%nightday

output_directory_root    = os.path.join(project_root, "outputs", "crossings8", nightday)

# ----------------------------------------------------------

iyr = ray_datenum.year
idoy= ray_datenum.timetuple().tm_yday 
isec = (ray_datenum.second + (ray_datenum.minute)*60 + ray_datenum.hour*60*60)


# -------------- set up MPI -----------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
host = commands.getoutput("hostname")
nProcs = 1.0*comm.Get_size()

# Set up output tree
if rank==0:
    for Kp in Kpvec:
        output_directory = os.path.join(output_directory_root, Kp)    
        # if 'ngo' in Kp:
        #     output_directory = os.path.join(output_directory_root, Kp)    
        # else:
        #     output_directory = os.path.join(output_directory_root, 'kp%d'%Kp)
        if not os.path.exists(output_directory):
            # os.mkdir(output_directory)
            os.system('mkdir -p %s'%output_directory)
        
        python_dump_path = os.path.join(output_directory,'python_data')
        
        if not os.path.exists(python_dump_path):
            os.mkdir(python_dump_path)
        
        # for (f1, f2) in freq_pairs:
        #     subdir = os.path.join(output_directory, 'f_%g_%g'%(f1, f2))
        #     if not os.path.exists(subdir):
        #         os.mkdir(subdir)
        #     for olat in out_lat:
        #         subsubdir = os.path.join(subdir, 'out_%g'%olat)
        #         if not os.path.exists(subsubdir):
        #             os.mkdir(subsubdir)
else:
    pass

comm.Barrier()

if rank==0:

    tasklist = [(x,y,z) for x,y,z in itertools.product(ray_lat_pairs, freq_pairs, Kpvec)]
    np.random.shuffle(tasklist)
    chunks = partition(tasklist, nProcs)

    print len(tasklist), "total jobs"
else:
    tasklist = None
    chunks   = None

comm.Barrier()

tasklist = comm.bcast(tasklist, root=0)
chunks   = comm.bcast(chunks, root=0)

# Split frequency vector into smaller chunks, pass each chunk to a process
nTasks  = 1.0*len(tasklist)
nSteps = np.ceil(nTasks/nProcs).astype(int)


# Run each set of jobs on current node:
if (rank < len(chunks)):
    print "Process %d on host %s, doing %g jobs"%(rank, host, len(chunks[rank]))

    for job_ind, job in enumerate(chunks[rank]):
        ray_lat_1 = job[0][0]
        ray_lat_2 = job[0][1]
        f1 = job[1][0]
        f2 = job[1][1]
        Kp = job[2]

        ray_dir = os.path.join(ray_input_directory_root, Kp)

        # if 'ngo' in Kp:
        #     ray_dir = os.path.join(ray_input_directory_root, Kp)
        # else:
        #     ray_dir = os.path.join(ray_input_directory_root,'kp%d'%Kp)

        if nightday == "nightside":
            center_lon = 0.
        else:
            center_lon = 180.

        data = find_crossings(ray_dir=ray_dir,
                                tmax =tmax,
                                dt = dt,
                                lat_low = ray_lat_1,
                                lat_step_size = ray_lat_spacing,
                                f_low=f1, f_hi = f2,
                                Llims = Llims,
                                L_step = Lstep,
                                center_lon = center_lon,
                                lon_spacing = offset_lon_spacing,
                                dlat_fieldline = dlat_fieldline,
                                DAMP_THRESHOLD = damp_threshold,
                                n_sub_freqs = num_freq_steps
                                )

        output_directory = os.path.join(output_directory_root, Kp)    

        # if 'ngo' in Kp:
        #     output_directory = os.path.join(output_directory_root, Kp)    
        # else:
        #     output_directory = os.path.join(output_directory_root, 'kp%d'%Kp)
        # output_directory = os.path.join(output_directory_root, 'kp%d'%Kp)
        python_dump_path = os.path.join(output_directory,'python_data')

        outfile = os.path.join(python_dump_path,'crossing_log_lat_%d-%d_f_%d-%d.pklz'%(ray_lat_1, ray_lat_2, f1, f2))
        with gzip.open(outfile,'wb') as file:
            pickle.dump(data, file)
        
        print "Process %d on host %s, finished %d/%d"%(rank, host, job_ind+1, len(chunks[rank]))

    print "Process %d on host %s complete"%(rank, host)



