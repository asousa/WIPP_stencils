import matplotlib
matplotlib.use('agg')
import numpy as np
import pickle
from scipy import interpolate
import matplotlib.pyplot as plt
import os
import itertools
import random
import os
import time
import datetime
import gzip

R2D = 180./np.pi
D2R = np.pi/180.


def build_database(path):
    # Build precipitation database:
    # path='../outputs/stencils/nightside/stencil_3/'
    print "building database from", path
    ev2joule = (1.60217657)*1e-19 # Joules / ev
    joule2millierg = 10*1e10
    D2R = np.pi/180.0

    d = os.listdir(path)
    inlats = np.unique([int(x.split('_')[2]) for x in d if x.startswith('phi_inlat') and x.endswith('.pklz')])
    suffixes = np.unique([x[13:-5] for x in d if x.startswith('phi_inlat') and x.endswith('.pklz')])

    suffix = suffixes[0]

    with gzip.open(os.path.join(path,'phi_inlat_%d_%s.pklz')%(inlats[0], suffix),'rb') as file:
        data = pickle.load(file)

    print data['params'].keys()

    num_lons = np.shape(data['phi_N_sum'])[1]
    print num_lons
    dlon = 1
    lon_max = num_lons - 1
    lon_axis = np.linspace(-lon_max, lon_max, 2*num_lons - 1)
    L_axis = data['params']['Lshells']
    # lat_axis = np.arccos(np.sqrt(1.0/L_axis))*R2D

    # Energy vector, in ev
    E = data['params']['E_tot_arr']
    DE_EXP = data['params']['DE_EXP']
    E_EXP_BOT = data['params']['E_EXP_BOT']
    E_EXP_TOP = data['params']['E_EXP_TOP']
    dt = data['params']['dt']    

    #  Energy differential term dE, in kev
    DE = np.exp(np.linspace(1,len(E),len(E))*DE_EXP/np.log10(np.e))
    DE = DE*DE_EXP/np.log10(np.e)
    DE = DE*(1e-3)*np.power(10, E_EXP_BOT + DE_EXP/2.)


    # Interpolated axis:
    L_interp   = L_axis #np.linspace(L_axis[0], L_axis[-1], 50)
    lon_interp = lon_axis #np.linspace(lon_axis[0], lon_axis[-1], 50)

    # Output space:
    NN_grid = np.zeros([len(inlats), len(L_interp), len(lon_interp)])
    NS_grid = np.zeros([len(inlats), len(L_interp), len(lon_interp)])
    QN_grid = np.zeros([len(inlats), len(L_interp), len(lon_interp)])
    QS_grid = np.zeros([len(inlats), len(L_interp), len(lon_interp)])



    # --------(Do this per input latitude)

    for inlat_index, inlat in enumerate(inlats):
        with gzip.open(os.path.join(path,'phi_inlat_%d_%s.pklz')%(inlat, suffix),'rb') as file:
            print "loading", file.filename
            data = pickle.load(file)


        # Integrate over each energy bin
        QN = data['phi_N_sum']*(E*DE)[np.newaxis,np.newaxis,:]*ev2joule*joule2millierg*dt
        NN = data['phi_N_sum']*(DE)[np.newaxis,np.newaxis,:]*dt
        QS = data['phi_S_sum']*(E*DE)[np.newaxis,np.newaxis,:]*ev2joule*joule2millierg*dt
        NS = data['phi_S_sum']*(DE)[np.newaxis,np.newaxis,:]*dt

        # Sum over energy bins
        NNtotal = np.sum(NN,axis=-1) # counts/cm^2, total for flash
        QNtotal = np.sum(QN,axis=-1) # mErg/cm^2, total for flash
        NStotal = np.sum(NS,axis=-1) # counts/cm^2, total for flash
        QStotal = np.sum(QS,axis=-1) # mErg/cm^2, total for flash

        # Interpolate each card onto pleasant new axes
        def interp_stencil(D):
            D2 = np.hstack([np.fliplr(D)[:,0:-1], D])  # Flip left - right
            interp = interpolate.RegularGridInterpolator([L_axis, lon_axis],D2, bounds_error=False, fill_value=0)
            px, py = np.meshgrid(lon_interp, L_interp)
            pts = zip(py.ravel(), px.ravel())
            D_interp = np.log10(interp(pts)).reshape(len(L_interp), len(lon_interp))
            D_interp[np.isinf(D_interp)] = -100
            return D_interp

        N_N_stencil = interp_stencil(NNtotal)
        Q_N_stencil = interp_stencil(QNtotal)
        N_S_stencil = interp_stencil(NStotal)
        Q_S_stencil = interp_stencil(QStotal)

        NN_grid[inlat_index,:,:] = N_N_stencil
        QN_grid[inlat_index,:,:] = Q_N_stencil
        NS_grid[inlat_index,:,:] = N_S_stencil
        QS_grid[inlat_index,:,:] = Q_S_stencil
        
    # Result: NN_grid, QN_grid, NS_grid, QS_grid ~ [inlats x L_interp x lon_interp]

    db = dict()
    db['N_NH'] = NN_grid
    db['N_SH'] = NS_grid
    db['Q_NH'] = QN_grid
    db['Q_SH'] = QS_grid
    db['L_axis'] = L_interp
    db['lon_axis'] = lon_interp
    db['inlats'] = inlats

    return db



# Precip model:


class precip_model(object):
    def __init__(self, db, mode='counts'):

        self.db = db
        self.db['lat_axis'] = np.arccos(np.sqrt(1.0/db['L_axis']))*R2D;
        if mode=='counts':
            # Electron number flux
            n_key = 'N_NH'; s_key = 'N_SH'
        else:
            # Energy flux
            n_key = 'Q_NH'; s_key = 'Q_SH'
        self.N_interp = interpolate.RegularGridInterpolator((self.db['inlats'], self.db['lat_axis'], self.db['lon_axis']),
                                                            self.db[n_key], fill_value=0, bounds_error=False)
        self.S_interp = interpolate.RegularGridInterpolator((self.db['inlats'], self.db['lat_axis'], self.db['lon_axis']),
                                                            self.db[s_key], fill_value=0, bounds_error=False)
            
        # Initialize any other parameters we might store:
        self.precalculated = None
        self.pc_inlats = None
        self.pc_L = None
        self.pc_lon = None
        
        
    def get_precip_at(self, inlats, outlats, outlons):
        tx, ty, tz  = np.meshgrid(inlats, outlats, outlons, indexing='ij')
        keys =  np.array([np.abs(tx.ravel()),np.abs(ty.ravel()), np.abs(tz.ravel())]).T
        
        # Model is symmetric around northern / southern hemispheres (mag. dipole coordinates):
        # If in = N, out = N  --> Northern hemisphere
        #    in = N, out = S  --> Southern hemisphere
        #    in = S, out = N  --> Southern hemisphere
        #    in = S, out = S  --> Northern hemisphere
        use_southern_hemi = np.array(((tx > 0) ^ (ty > 0)).ravel())
        keys_N = keys[~use_southern_hemi,:]
        keys_S = keys[ use_southern_hemi,:]
        
        out_data = np.zeros(len(keys))

        out_data[ use_southern_hemi] = np.maximum(0, self.S_interp(keys_S))
        out_data[~use_southern_hemi] = np.maximum(0, self.N_interp(keys_N))
        return out_data.reshape(len(inlats), len(outlats),  len(outlons))


if __name__=='__main__':

    db = build_database(path='../outputs/stencils/nightside/stencil_3/')
    p = precip_model(db)


    inlats = [25, 35, 45]
    outlats = np.arange(-70, 70, 0.5)
    outlons = np.arange(-20, 21, 0.5)
    data = p.get_precip_at(inlats, outlats, outlons)

    fig, ax = plt.subplots(1,len(inlats))
    for k,v in enumerate(inlats):
        print k, v
        ax[k].pcolorfast(outlons, outlats, data[k,:,:])

    fig.savefig('dumpy.png')