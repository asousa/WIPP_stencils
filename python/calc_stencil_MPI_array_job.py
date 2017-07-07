from calc_stencil_MPI import calc_stencil_MPI
import os
import datetime as datetime
import sys
# from mpi4py import MPI
# import commands


# # -------------- set up MPI -----------------------------
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# host = commands.getoutput("hostname")
# nProcs = 1.0*comm.Get_size()

# inlats = [15, 20, 25, 30, 35, 40, 45, 50, 55]

# if rank == 0:
#     index =int(os.getenv('PBS_ARRAYID'))
#     print "host:",host
#     print "nProcs",nProcs
#     print "index", index
#     print "inlat", inlats[index]



inlat = int(os.getenv('inlat'))
side =  os.getenv('niteday')
kp   = int(os.getenv('kp'))
# side = 'nightside'

rootdir ='/shared/users/asousa/WIPP/WIPP_stencils/'
crossing_dir = os.path.join(rootdir,'outputs','crossings_30f_0.2l',side,'kp%d'%kp,'python_data')
power_dir    = os.path.join(rootdir,'outputs','input_energies')
out_dir      = os.path.join(rootdir,'outputs','stencils',side,'stencil_30f_0.2L','kp%d'%kp)

alpha_dist = 0  # 0: Sine / ramp  1: Square
flux_dist = 0   # 0: fluxfile  1: Suprathermal  2: Flat

# inlats = [15, 20, 25, 30, 35, 40, 45, 50, 55]
print "in lat is:", inlat
print "side is:",side

if side=='nightside':
    mlt = 0
if side=='dayside':
    mlt= 12


# print "sub-job", index, "doing lat", inlat
calc_stencil_MPI(
    crossing_dir = crossing_dir,
    power_dir = power_dir,
    out_dir = out_dir,
    # fluxfile = fluxfile,
    alpha_dist = alpha_dist,
    # flux_dist = flux_dist,
    flash_lat=inlat,
    mlt = mlt,
    max_dist=600,
    I0=-10000,
    d_lon = 1,
    num_lons=15,
    f_low=200, f_hi=30000,
    itime = datetime.datetime(2010,1,1,0,0,0))