from calc_scattering_MPI import calc_scattering_MPI
import datetime as datetime



Ivec = [-1000, -2000, -10000, -20000, -50000, -100000, -200000, -500000]

for I0 in Ivec:
    calc_scattering_MPI(
        crossing_dir ='/shared/users/asousa/WIPP/WIPP_stencils/outputs/crossings7/nightside/ngo_v2/python_data',
        power_dir = '/shared/users/asousa/WIPP/WIPP_stencils/outputs/input_energies/',
        out_dir = '/shared/users/asousa/WIPP/WIPP_stencils/outputs/scattering/nightside/power_sweep/current_%d'%I0,
        flash_lat=35,
        mlt = 0,
        max_dist=500,
        I0=I0,
        d_lon = 1,
        num_lons=1,
        f_low=200, f_hi=30000,
        itime = datetime.datetime(2010,1,1,0,0,0))
