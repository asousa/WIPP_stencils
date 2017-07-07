import matplotlib
matplotlib.use('agg')
import numpy as np
import pandas as pd
import pickle
import gzip
from scipy import interpolate
import matplotlib.pyplot as plt
import sys
import itertools
import random
import os
import time
import datetime as datetime
from random import shuffle

# The goods
from calc_stencil_MPI import calc_stencil_MPI


rootdir ='/shared/users/asousa/WIPP/WIPP_stencils/'
# crossing_dir = os.path.join(rootdir,'outputs','crossings9_quick','nightside','ngo_v2','python_data')
crossing_dir = os.path.join(rootdir,'outputs','crossings_20f','nightside','ngo_v2','python_data')
power_dir    = os.path.join(rootdir,'outputs','input_energies')
out_dir      = os.path.join(rootdir,'outputs','stencils','nightside','stencil_testing')

# fluxfile = '/shared/users/asousa/WIPP/WIPP_stencils/data/AE8MAX_integral_flux.dat'
alpha_dist = 0  # 0: Sine / ramp  1: Square
flux_dist = 0   # 0: fluxfile  1: Suprathermal  2: Flat

inlat = float(sys.argv[1])





