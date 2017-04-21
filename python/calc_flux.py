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
# from mpi4py import MPI

import ctypes as ct
from numpy.ctypeslib import ndpointer



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

def calc_flux(scattering_dir=None,
                out_dir = None,
                flash_lat=35,
                mlt = 0,
                alpha_dist = 0,
                flux_dist  = 0,
                fluxfile = '/shared/users/asousa/WIPP/WIPP_stencils/c/data/AE8MaxFlux_expanded.dat',
                itime = datetime.datetime(2010,1,1,0,0,0)):

    # -------------- set up C module ------------------------
    lib_path ='/shared/users/asousa/WIPP/WIPP_stencils/c/libwipp.so'
    ct.cdll.LoadLibrary(lib_path)
    lib = ct.CDLL(lib_path)
    # The function
    calc_flux_c = lib.calc_flux





    Jdata = np.loadtxt(fluxfile)

    JL = Jdata[1:,0]  # L-shells in J-file
    JE = Jdata[0,1:]  # Energies in J-file

    print JL  
    print JE 

    infile = os.path.join(scattering_dir,'scattering_inlat_%d.pklz'%flash_lat)
    

    with gzip.open(infile,'r') as file:
        indata = pickle.load(file)


    da_N = indata['da_N']
    da_S = indata['da_S']
    pyparams = indata['params']

    print pyparams.keys()






if __name__ == "__main__":
    calc_flux(scattering_dir = '/shared/users/asousa/WIPP/WIPP_stencils/outputs/scattering/nightside/ngo_v2/')






