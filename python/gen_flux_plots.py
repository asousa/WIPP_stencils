import matplotlib
matplotlib.use('agg')
import numpy as np
import pandas as pd
import pickle
#from build_database import flux_obj
from scipy import interpolate
import matplotlib.pyplot as plt
import os
import itertools
import random
import os
import time
import datetime
import gzip
import matplotlib.gridspec as gridspec

from spacepy import coordinates as coord
from spacepy.time import Ticktock

from methods.raytracer_utils import readdump, read_rayfile, read_rayfiles
from mpl_toolkits.mplot3d import Axes3D

import methods.xflib as xflib


def plot_flux(data, L_ind, clims=[-10,4]):
    # Plot 2d slice, flux vs time vs energy

    ev2joule = (1.60217657)*1e-19 # Joules / ev
    joule2millierg = 10*1e10 


    # --------------- Latex Plot Beautification --------------------------
    fig_width = 12 
    fig_height = 8
    fig_size =  [fig_width+1,fig_height+1]
    params = {'backend': 'ps',
              'axes.labelsize': 18,
              'font.size': 18,
              'legend.fontsize': 18,
              'xtick.labelsize': 14,
              'ytick.labelsize': 14,
              'text.usetex': False,
              'figure.figsize': fig_size}
    plt.rcParams.update(params)
    # --------------- Latex Plot Beautification --------------------------
    # print data.keys()
    phi_N = data['phi_N_full']
    phi_S = data['phi_S_full']
    Lshells = data['params']['Lshells']
    tvec    = data['params']['tvec']
    evec    = data['params']['E_tot_arr']

    # print data['params'].keys()
    # print np.shape(phi_N)
    
    # print Lshells[L_ind]

    # clims = [-10, 4]
    # print np.shape(tvec), np.shape(evec)

    fig = plt.figure()
    gs = gridspec.GridSpec(2,2, width_ratios=[1, 0.02])
    
    ax0 = plt.subplot(gs[0,0])
    ax1 = plt.subplot(gs[1,0])
    cbax = plt.subplot(gs[:,-1])
    cmap = plt.get_cmap('jet')


    Ndata = np.log10(phi_N[L_ind,:,:-1])
    Ndata[np.isinf(Ndata)] = -100
    # print np.max(Ndata), np.min(Ndata)
    Sdata = np.log10(phi_S[L_ind,:,:-1])
    Sdata[np.isinf(Sdata)] = -100
    # print np.max(Sdata), np.min(Sdata)
    p0 = ax0.pcolormesh(tvec, np.log10(evec/1000.), Ndata, vmin = clims[0], vmax=clims[1], cmap = cmap)
    p1 = ax1.pcolormesh(tvec, np.log10(evec/1000.), Sdata, vmin = clims[0], vmax=clims[1], cmap = cmap)

    ax0.set_xticks([])
    ytix = ax0.get_yticks()
    yticklabels = ['$10^{%d}$'%k for k in ytix]
    ax0.set_yticklabels(yticklabels)
    ytix = ax1.get_yticks()
    yticklabels = ['$10^{%d}$'%k for k in ytix]
    ax1.set_yticklabels(yticklabels)

    ax0.set_ylabel('Northern Hemisphere\nParticle Energy (kev)')
    ax1.set_ylabel('Southern Hemisphere\nParticle Energy (kev)')
    ax1.set_xlabel('Time (sec)')

    cb = plt.colorbar(p0, cax=cbax)
    cticks = np.arange(clims[0],clims[1] + 1)
    cb.set_ticks(cticks)
    cticklabels = ['$10^{%d}$'%k for k in cticks]
    cb.set_ticklabels(cticklabels)
    cb.set_label('$\Delta \\alpha_{RMS}$ (deg)')
    cb.set_label('$\Phi$ [#/cm$^2$/keV/s]')

    # ax0.set_title('Phi: In lat = %g deg, L = %g'%(Lshells[L_ind]))
    fig.tight_layout()

    return fig, ax0, ax1


def plot_hotspot(data):
    # --------------- Latex Plot Beautification --------------------------
    fig_width = 12 
    fig_height = 8
    fig_size =  [fig_width+1,fig_height+1]
    params = {'backend': 'ps',
              'axes.labelsize': 18,
              'font.size': 18,
              'legend.fontsize': 18,
              'xtick.labelsize': 14,
              'ytick.labelsize': 14,
              'text.usetex': False,
              'figure.figsize': fig_size}
    plt.rcParams.update(params)
    # --------------- Latex Plot Beautification --------------------------

    # Hotspot maps vs longitude and L-shell:
    Qlims = [-6,0]
    Nlims = [-1,5]
    num_lons = np.shape(data['phi_N_sum'])[1]
    dlon = 1
    lon_max = num_lons - 1
    lon_axis = np.linspace(-lon_max, lon_max, 2*num_lons - 1)
    L_axis = data['params']['Lshells']

    L_interp   = np.linspace(L_axis[0], L_axis[-1], 100)
    lon_interp = np.linspace(lon_axis[0], lon_axis[-1], 100)


    # First -- integrate phi to get N (number flux) or E

    ev2joule = (1.60217657)*1e-19 # Joules / ev
    joule2millierg = 10*1e10
    # print data['params']['E_tot_arr']

    # print data['params'].keys()

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

    def interp_stencil(D):
        D2 = np.hstack([np.fliplr(D)[:,0:-1], D])  # Flip left - right
        interp = interpolate.RegularGridInterpolator([Lshells, lon_axis],D2)
        px, py = np.meshgrid(lon_interp, L_interp)
        pts = zip(py.ravel(), px.ravel())
        D_interp = np.log10(interp(pts)).reshape(len(L_interp), len(lon_interp))
        D_interp[np.isinf(D_interp)] = -100
        return D_interp

    N_N_stencil = interp_stencil(NNtotal)
    Q_N_stencil = interp_stencil(QNtotal)
    N_S_stencil = interp_stencil(NStotal)
    Q_S_stencil = interp_stencil(QStotal)


    cmap = plt.get_cmap('jet')
    fig = plt.figure()
    gs = gridspec.GridSpec(2,5, width_ratios=[1, 0.05, 0.1, 1, 0.05])
    axQN = plt.subplot(gs[0,0])
    axQS = plt.subplot(gs[1,0])
    axNN = plt.subplot(gs[0,3])
    axNS = plt.subplot(gs[1,3])
    cbQ  = plt.subplot(gs[:,1])
    cbN  = plt.subplot(gs[:,4])
    # ax.imshow(np.hstack([np.fliplr(Ntotal)[:,0:-1], Ntotal]))

    pQN = axQN.pcolorfast(lon_interp, L_interp, Q_N_stencil, vmin=Qlims[0], vmax=Qlims[1], cmap=cmap)
    pQS = axQS.pcolorfast(lon_interp, L_interp, Q_S_stencil, vmin=Qlims[0], vmax=Qlims[1], cmap=cmap)
    cQ = plt.colorbar(pQN, cax=cbQ)

    pNN = axNN.pcolorfast(lon_interp, L_interp, N_N_stencil, vmin=Nlims[0], vmax=Nlims[1],  cmap=cmap)
    pNS = axNS.pcolorfast(lon_interp, L_interp, N_S_stencil, vmin=Nlims[0], vmax=Nlims[1],  cmap=cmap)
    cN = plt.colorbar(pNN, cax=cbN)

    axQS.invert_yaxis()
    axNS.invert_yaxis()
    # Title the columns
    axQN.set_title('Energy flux')
    axNN.set_title('Number flux')
    # Hide y axes on right column:
    axNN.set_yticks([])
    axNS.set_yticks([])
    axQN.set_xticks([])
    axNN.set_xticks([])

    cticks = np.arange(Qlims[0],Qlims[1] + 1)
    cQ.set_ticks(cticks)
    cticklabels = ['$10^{%d}$'%k for k in cticks]
    cQ.set_ticklabels(cticklabels)
    cQ.set_label('Energy [mErg/cm$^2$]')

    cticks = np.arange(Nlims[0],Nlims[1] + 1)
    cN.set_ticks(cticks)
    cticklabels = ['$10^{%d}$'%k for k in cticks]
    cN.set_ticklabels(cticklabels)
    cN.set_label('Electrons [#/cm$^2$]')



    axQN.set_ylabel('Northern Hemi\n L-shell')
    axQS.set_ylabel('Southern Hemi\n L-shell')
    axQS.set_xlabel('Longitude from Flash [deg]')
    axNS.set_xlabel('Longitude from Flash [deg]')

    # fig.tight_layout()
    return fig











xf = xflib.xflib(lib_path='/shared/users/asousa/WIPP/WIPP_stencils/python/methods/libxformd.so')

R2D = 180./np.pi
D2R = np.pi/180.
#%matplotlib inline
# %matplotlib nbagg
# Autoload changes made in external editor:
# %load_ext autoreload
# %autoreload 2
# --------------- Latex Plot Beautification --------------------------
fig_width = 12 
fig_height = 8
fig_size =  [fig_width+1,fig_height+1]
params = {'backend': 'ps',
          'axes.labelsize': 18,
          'font.size': 18,
          'legend.fontsize': 18,
          'xtick.labelsize': 14,
          'ytick.labelsize': 14,
          'text.usetex': False,
          'figure.figsize': fig_size}
plt.rcParams.update(params)
# --------------- Latex Plot Beautification --------------------------




inp_root = '/shared/users/asousa/WIPP/WIPP_stencils/outputs/stencils/nightside/stencil_3/'
out_root = os.path.join(inp_root,'figures')

if not (os.path.exists(out_root)):
    os.system('mkdir -p %s'%out_root)
in_lats = [25, 35, 45, 55]
in_suffixes = ['AE8MIN_flux_0', 
               'AE8MAX_flux_0',
               'AE8MAX_flux_1',
               'AE8MAX_flux_2']


for inlat in in_lats:
    outpath = os.path.join(out_root,'in_%d'%inlat)
    if not os.path.exists(outpath):
        os.system('mkdir %s'%outpath)
    for suffix in in_suffixes:
        # Load it

        infile = os.path.join(inp_root,'phi_inlat_%g_%s.pklz'%(inlat, suffix))
        print "Loading",infile
        with gzip.open(infile,'rb') as file:
            data = pickle.load(file)
        print infile
        print out_root
        Lshells = data['params']['Lshells']
        
        # Flux vs time vs energy, center longitude slices:
        for L_ind, Lsh in enumerate(Lshells):
            print "L: ", Lsh
            fig, ax0, ax1 = plot_flux(data, L_ind)
            out_fn = os.path.join(outpath,'phi_inlat_%d_L_%2.1f_%s.png'%(inlat, Lsh,suffix))

            ax0.set_title('phi: %g $^o$, L=%g'%(inlat, Lsh))
            fig.tight_layout()
            fig.savefig(out_fn, ldpi=300)
            plt.close(fig)

        # Hotspot, L-shell vs longitude:
        print "hotspot plot"
        out_fn = os.path.join(outpath,'hotspot_%d_%s.png'%(inlat, suffix))
        fig = plot_hotspot(data)
        fig.savefig(out_fn,ldpi=300)
        plt.close(fig)






