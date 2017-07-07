# Precipitation model, with Kp interpolation and separated energy bands:
from __future__ import division

# import matplotlib
# matplotlib.use('agg')
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

from joblib import Parallel, delayed
import multiprocessing

R2D = 180./np.pi
D2R = np.pi/180.



def build_database(path, n_bands=1, avail_kp = None, suffix = None):

    print "building database from", path
    ev2joule = (1.60217657)*1e-19 # Joules / ev
    joule2millierg = 10*1e10
    D2R = np.pi/180.0


    # --------- Load the first file to get axes, etc ------------
    
    d = os.listdir(path)
    if avail_kp == None:
        avail_kp = [int(x[-1]) for x in d if x.startswith('kp')]
    # avail_kp = [0, 2, 4]

    in_kp = avail_kp[0]

    curpath = os.path.join(path, 'kp%d'%in_kp)
    d = os.listdir(curpath)
    inlats = np.unique([int(x.split('_')[2]) for x in d if x.startswith('phi_inlat') and x.endswith('.pklz')])
    suffixes = np.unique([x[13:-5] for x in d if x.startswith('phi_inlat') and x.endswith('.pklz')])
    print suffixes
    if suffix == None:
        suffix = suffixes[0]


    with gzip.open(os.path.join(curpath,'phi_inlat_%d_%s.pklz')%(inlats[0], suffix),'rb') as file:
        data = pickle.load(file)

    print data['params'].keys()

    num_lons = np.shape(data['phi_N_sum'])[1]
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

    # Energy band dividers
    # n_bands = 32
    E_bands = pow(10, np.linspace(E_EXP_BOT,E_EXP_TOP,n_bands + 1))
    E_divider_indexes = np.digitize(E_bands, E) - 1
    E_band_pairs = zip(E_divider_indexes[0:-1], E_divider_indexes[1:])

    print "band pairs:"
    print E_band_pairs

    # # Interpolated axis:
    # L_interp   = L_axis #np.linspace(L_axis[0], L_axis[-1], 50)
    # lon_interp = lon_axis #np.linspace(lon_axis[0], lon_axis[-1], 50)

    # Output space (not interpolated yet):

    frame_dims = np.shape(data['phi_N_sum'])[:-1]
    # print frame_dims
    NN_grid = np.zeros([len(avail_kp), len(inlats), frame_dims[0], frame_dims[1], len(E_band_pairs)])
    NS_grid = np.zeros([len(avail_kp), len(inlats), frame_dims[0], frame_dims[1], len(E_band_pairs)])
    QN_grid = np.zeros([len(avail_kp), len(inlats), frame_dims[0], frame_dims[1], len(E_band_pairs)])
    QS_grid = np.zeros([len(avail_kp), len(inlats), frame_dims[0], frame_dims[1], len(E_band_pairs)])

    print "Got axes, etc"
        # stencil_arr = np.zeros([len(inmlts), len(inlats), len(inkps), len(stencil_Lsh), len(stencil_lons)])


        #     # --------(Do this per input latitude)
    for kp_ind, kp in enumerate(avail_kp):
        curpath = os.path.join(path, 'kp%d'%kp)
        for inlat_index, inlat in enumerate(inlats):

            with gzip.open(os.path.join(curpath,'phi_inlat_%d_%s.pklz')%(inlat, suffix),'rb') as file:
                print "loading", file.filename
                data = pickle.load(file)

            # Integrate over each energy bin
            QN = data['phi_N_sum']*(E*DE)[np.newaxis,np.newaxis,:]*ev2joule*joule2millierg*dt
            NN = data['phi_N_sum']*(DE)[np.newaxis,np.newaxis,:]*dt
            QS = data['phi_S_sum']*(E*DE)[np.newaxis,np.newaxis,:]*ev2joule*joule2millierg*dt
            NS = data['phi_S_sum']*(DE)[np.newaxis,np.newaxis,:]*dt
            
            # print np.shape(QN)
            for p_ind, (p1, p2) in enumerate(E_band_pairs):
                # print p_ind, p1, p2
                
                # Sum over energy bins, normalize per frequency
                NNtotal = np.sum(NN[:,:,p1:p2],axis=-1)/(E[p2]-E[p1]) # counts/cm^2/ev, total for flash (specrtral)
                QNtotal = np.sum(QN[:,:,p1:p2],axis=-1)/(E[p2]-E[p1]) # mErg/cm^2/ev, total for flash
                NStotal = np.sum(NS[:,:,p1:p2],axis=-1)/(E[p2]-E[p1]) # counts/cm^2/ev, total for flash
                QStotal = np.sum(QS[:,:,p1:p2],axis=-1)/(E[p2]-E[p1]) # mErg/cm^2/ev, total for flash

                NN_grid[kp_ind, inlat_index,:,:, p_ind] = NNtotal # N_N_stencil
                QN_grid[kp_ind, inlat_index,:,:, p_ind] = QNtotal # Q_N_stencil
                NS_grid[kp_ind, inlat_index,:,:, p_ind] = NStotal # N_S_stencil
                QS_grid[kp_ind, inlat_index,:,:, p_ind] = QStotal # Q_S_stencil

            # Result: NN_grid, QN_grid, NS_grid, QS_grid ~ [Kp x inlats x L_interp x lon_interp x energy_band]

    db = dict()
    db['N_NH'] = NN_grid
    db['N_SH'] = NS_grid
    db['Q_NH'] = QN_grid
    db['Q_SH'] = QS_grid
    db['L_axis'] = L_axis
    db['lon_axis'] = lon_axis
    db['kp_axis'] = avail_kp
    db['inlats'] = inlats
    db['band_pairs'] = E_band_pairs
    db['E_vec'] = E

    return db




class precip_model(object):
    def __init__(self, db, mode='counts'):
        self.mode = mode
        self.db = db
        # Cast L-shell axis to latitude (dipole model)
        self.db['lat_axis'] = np.arccos(np.sqrt(1.0/db['L_axis']))*R2D;
        if mode=='counts':
            # Electron number flux
            n_key = 'N_NH'; s_key = 'N_SH'
        if mode=='energy':
            # Energy flux
            n_key = 'Q_NH'; s_key = 'Q_SH'
            
        print n_key, s_key
        print np.max(self.db[s_key]), np.min(self.db[s_key])
        band_axis = range(len(db['band_pairs']))
        lon_axis = np.unique(np.abs(self.db['lon_axis']))

        self.n_bands = len(db['band_pairs'])

        if self.n_bands == 1:
            # single band
            print "single band interpolators"
            self.N_interp = interpolate.RegularGridInterpolator((self.db['kp_axis'],self.db['inlats'], self.db['lat_axis'], 
                                                                 lon_axis),
                                                                self.db[n_key], fill_value=0, bounds_error=False)
            self.S_interp = interpolate.RegularGridInterpolator((self.db['kp_axis'],self.db['inlats'], self.db['lat_axis'], 
                                                                 lon_axis),
                                                                self.db[s_key], fill_value=0, bounds_error=False)
        else:
            print "multi band interpolators"
            # Multiple bands
            self.N_interp = interpolate.RegularGridInterpolator((self.db['kp_axis'],self.db['inlats'], self.db['lat_axis'], 
                                                                 lon_axis, band_axis),
                                                                self.db[n_key], fill_value=0, bounds_error=False)
            self.S_interp = interpolate.RegularGridInterpolator((self.db['kp_axis'],self.db['inlats'], self.db['lat_axis'], 
                                                                 lon_axis, band_axis),
                                                                self.db[s_key], fill_value=0, bounds_error=False)
            
        # Initialize any other parameters we might store:
        self.precalculated = dict()
        # self.precalculated = None
        # self.pc_inlats = None
        # self.pc_kp = None
        # self.pc_outlats = None
        # self.pc_lon = None
        # self.pc_logscale = None
        # self.pc_bands = None
        

    def get_precip_at(self, inkps, inlats, outlats, outlons, logscale = False):

        if self.n_bands == 1:

            tw, tx, ty, tz  = np.meshgrid(inkps, inlats, outlats, outlons, indexing='ij')
            keys =  np.array([np.abs(tw.ravel()), np.abs(tx.ravel()),np.abs(ty.ravel()), np.abs(tz.ravel())]).T
            
            # Model is symmetric around northern / southern hemispheres (mag. dipole coordinates):
            # If in = N, out = N  --> Northern hemisphere
            #    in = N, out = S  --> Southern hemisphere
            #    in = S, out = N  --> Southern hemisphere
            #    in = S, out = S  --> Northern hemisphere
            use_southern_hemi = np.array(((tx > 0) ^ (ty > 0)).ravel())
            keys_N = keys[~use_southern_hemi,:]
            keys_S = keys[ use_southern_hemi,:]

            out_data = np.zeros(len(keys))

            out_data[ use_southern_hemi] = self.S_interp(keys_S)
            out_data[~use_southern_hemi] = self.N_interp(keys_N)
    #         return out_data.reshape(len(inkps), len(inlats), len(outlats),  len(outlons))
            out_data = out_data.reshape(len(inkps), len(inlats), len(outlats),  len(outlons))
            if logscale:
                out_data = np.log10(out_data)
    #             out_data[np.isinf(out_data)] = -100
            od = out_data
        else:
            num_cores = multiprocessing.cpu_count()

            # More than one energy band
            od = np.zeros([len(inkps), len(inlats), len(outlats), len(outlons), self.n_bands])

            # for b in range(self.n_bands):
        #     def per_band(b):
        #         print "band",b
        #         tw, tx, ty, tz, ta  = np.meshgrid(inkps, inlats, outlats, outlons, b, indexing='ij')
        #         keys =  np.array([np.abs(tw.ravel()), np.abs(tx.ravel()),np.abs(ty.ravel()), np.abs(tz.ravel()), ta.ravel()]).T
                
        #         # Model is symmetric around northern / southern hemispheres (mag. dipole coordinates):
        #         # If in = N, out = N  --> Northern hemisphere
        #         #    in = N, out = S  --> Southern hemisphere
        #         #    in = S, out = N  --> Southern hemisphere
        #         #    in = S, out = S  --> Northern hemisphere
        #         use_southern_hemi = np.array(((tx > 0) ^ (ty > 0)).ravel())
        #         keys_N = keys[~use_southern_hemi,:]
        #         keys_S = keys[ use_southern_hemi,:]

        #         out_data = np.zeros(len(keys))

        #         out_data[ use_southern_hemi] = self.S_interp(keys_S)
        #         out_data[~use_southern_hemi] = self.N_interp(keys_N)
        # #         return out_data.reshape(len(inkps), len(inlats), len(outlats),  len(outlons))
        #         out_data = out_data.reshape(len(inkps), len(inlats), len(outlats),  len(outlons))
        #         if logscale:
        #             out_data = np.log10(out_data)
        # #             out_data[np.isinf(out_data)] = -100
        #         # od[:,:,:,:,b] = out_data
        #         return out_data

            # print "n_bands:", self.n_bands
            # print "num cores:", num_cores
            band_inds = range(self.n_bands)
            jo = Parallel(n_jobs=num_cores)(delayed(per_band)(self, inkps, inlats, outlats, outlons, logscale, b)
                         for b in band_inds)
            # print "jo is:", np.shape(jo)
            
            # Shuffle indexes -- we've paralleled along the energy axis (last axis),
            # but the data is returned from the workers along the first axis            
            for b in range(self.n_bands):
                od[:,:,:,:,b] = jo[b]
            # od = np.array(jo).swapaxes(0,-1)
            # print "od is:", np.shape(od)
            # print "inkps:", np.shape(inkps)
            # print "inlats:", np.shape(inlats)
            # print "outlats:", np.shape(outlats)
            # print "outlons:", np.shape(outlons)
        return od
    
    def precalculate(self, inkps, inlats, outlats, outlons, logscale):
        self.precalculated['data'] = self.get_precip_at(inkps, inlats, outlats, outlons, logscale=logscale)
        self.precalculated['inlats'] = inlats
        self.precalculated['kps'] = inkps
        self.precalculated['outlats'] = outlats
        self.precalculated['outlons'] = outlons
        self.precalculated['logscale'] = logscale



def per_band(obj,inkps, inlats, outlats, outlons, logscale, b):
    # Per-band helper for get_precip_at().
    # When using the easy parallel library, we can't call methods belonging
    # to an explicit object. I'm sure some CS dude is cursing me for this.
        print "band",b
        tw, tx, ty, tz, ta  = np.meshgrid(inkps, inlats, outlats, outlons, b, indexing='ij')
        keys =  np.array([np.abs(tw.ravel()), np.abs(tx.ravel()),np.abs(ty.ravel()), np.abs(tz.ravel()), ta.ravel()]).T
        
        # Model is symmetric around northern / southern hemispheres (mag. dipole coordinates):
        # If in = N, out = N  --> Northern hemisphere
        #    in = N, out = S  --> Southern hemisphere
        #    in = S, out = N  --> Southern hemisphere
        #    in = S, out = S  --> Northern hemisphere
        use_southern_hemi = np.array(((tx > 0) ^ (ty > 0)).ravel())
        keys_N = keys[~use_southern_hemi,:]
        keys_S = keys[ use_southern_hemi,:]

        out_data = np.zeros(len(keys))

        out_data[ use_southern_hemi] = obj.S_interp(keys_S)
        out_data[~use_southern_hemi] = obj.N_interp(keys_N)
#         return out_data.reshape(len(inkps), len(inlats), len(outlats),  len(outlons))
        out_data = out_data.reshape(len(inkps), len(inlats), len(outlats),  len(outlons))
        if logscale:
            out_data = np.log10(out_data)
#             out_data[np.isinf(out_data)] = -100
        # od[:,:,:,:,b] = out_data
        return out_data

if __name__=='__main__':

   # --- build db ---
    db = build_database(path='../outputs/stencils/nightside/stencil_30f_0.2L/', n_bands = 16)

    with open('../outputs/precip_dbs/nightside.pkl','wb') as file:
        pickle.dump(db, file, protocol=pickle.HIGHEST_PROTOCOL)

    # --- do things with it --
    # with open('../outputs/precip_dbs/nightside.pkl','rb') as file:
    #     db = pickle.load(file)

    p = precip_model(db, mode='counts')

    inkps = [0, 2, 4]
    inlats = np.arange(15, 56, 10) #[15, 20, 25, 35, 45, 55]
    outlats = np.arange(-70, 70, 0.5)
    outlons = np.arange(-14, 14, 0.1)

    p.precalculate(inkps, inlats, outlats, outlons, logscale=True)
    data = p.precalculated

    fig, ax = plt.subplots(1,len(inlats))
    for k,v in enumerate(inlats):
        print k, v
        ax[k].pcolorfast(outlons, outlats, data[0,k,:,:,0], vmin=0, vmax=7, cmap=plt.get_cmap('jet'))
        print "lat ", v, "max val: ", np.max(data[0,k,:,:,0])
        ax[k].set_title('%d'%v)
    print np.max(p.db['Q_SH'])

    for a in ax[1:]:
        a.set_yticks([])
    ax[0].set_ylabel('Latitude (deg)')

    fig.savefig('dumpy.png')