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
from raytracer_utils import read_rayfile, read_damp
from scipy.spatial import Delaunay
from scipy.integrate import nquad
from scipy import stats
import xflib
from graf_iono_absorp import total_input_power, lon2MLT, MLT2lon, input_power_scaling
import logging
import math

from scipy.special import jn      # Bessel function of the 1st kind
from scipy.special import fresnel # Fresnel integrals

def calc_pitch_angle_change(inp_pwr, fieldlines, da_N, da_S, params):

    # Constants
    Hz2Rad = 2.*np.pi
    D2R = np.pi/180.
    H_IONO_BOTTOM = 1e5
    H_IONO_TOP = 1e6
    R_E = 6371e3
    Q_EL = 1.602e-19
    M_EL = 9.1e-31
    E_EL = 5.105396765648739E5 
    MU0  = np.pi*4e-7
    EPS0 = 8.854E-12
    C    = 2.997956376932163e8
    B0   = 3.12e-5

    # # Parameters
    SCATTERING_RES_MODES = params['SCATTERING_RES_MODES']
    E_BANDWIDTH = params['E_BANDWIDTH']
    Emin = params['Emin']
    Emax = params['Emax']
    NUM_E = params['NUM_E']
    E_tot_arr = params['E_tot_arr']
    v_tot_arr = params['v_tot_arr']
    Lshells = params['Lshells']
    tvec = params['tvec']
    PWR_THRESHOLD = 10e-30

    # fieldlines is a list of dictionaries with the following:
    #['stixR', 'stixP', 'hit_counts', 'x_unit_vect', 'vol', 'total_vol', 
    #'crossings', 'y_unit_vect', 'L', 'pos', 'mu', 'lat', 'ftc_s', 'R', 'y', 'x', 'ftc_n', 'xradius', 'stixL']
    
    # Loop over field lines
    for fl_ind, fl in enumerate(fieldlines):
        # Which segments have crossings at them?
        L = fl['L']
        print "L: ", L
        # Loop over EA segments
        lat_inds = np.where(fl['hit_counts'] > 0)[0]
        for lat_ind in lat_inds:
            lat = fl['lat'][lat_ind]

            slat = np.sin(lat*D2R)
            clat = np.cos(lat*D2R)
            slat_term = np.sqrt(1. + 3.*slat*slat)

            # Stix parameters
            stixP = fl['stixP'][lat_ind]
            stixR = fl['stixR'][lat_ind]
            stixL = fl['stixL'][lat_ind]

            # Other parameters which are a function of the EA segment only
            ftc_n = fl['ftc_n'][lat_ind]
            ftc_s = fl['ftc_s'][lat_ind]
            alpha_eq = fl['alpha_eq']
            alpha_lc = fl['alpha_lc'][lat_ind]
            calph = np.cos(alpha_lc)
            salph = np.sin(alpha_lc) 
            wh = fl['wh'][lat_ind]
            ds = fl['ds'][lat_ind]
            dv_para_ds = fl['dv_para_ds'][lat_ind]
            dwh_ds = fl['dwh_ds'][lat_ind]

            # Go through the crossings
            for row in fl['crossings'][lat_ind]:
                # print row

                t = row[0]
                f = row[1]
                pwr = row[2]*inp_pwr
                psi = D2R*row[3]
                mu =  row[4]

                if pwr > PWR_THRESHOLD:

                    spsi = np.sin(psi);
                    cpsi = np.cos(psi);
                    spsi_sq = pow(spsi,2);
                    cpsi_sq = pow(cpsi,2);
                    n_x = mu*abs(spsi);
                    n_z = mu*cpsi;
                    mu_sq = mu*mu;
                    w = 2.0*np.pi*f;
                    k = w*mu/C;
                    kx = w*n_x/C;
                    kz = w*n_z/C;
                    Y = wh / w ;

                    stixS = ( stixR + stixL ) /2.0;
                    stixD = ( stixR - stixL ) /2.0;
                    stixA = stixS + (stixP-stixS)*cpsi_sq;
                    stixB = stixP*stixS+stixR*stixL+(stixP*stixS-stixR*stixL)*cpsi_sq;
                    stixX = stixP/(stixP- mu_sq*spsi_sq);

                    rho1=((mu_sq-stixS)*mu_sq*spsi*cpsi)/(stixD*(mu_sq*spsi_sq-stixP));
                    rho2 = (mu_sq - stixS) / stixD ;

                    # (bortnik 2.28)
                    Byw_sq =   (2.0*MU0/C*pwr*stixX*stixX*rho2*rho2*mu*abs(cpsi) /
                                np.sqrt(  pow((np.tan(psi)-rho1*rho2*stixX),2) + 
                                pow( (1+rho2*rho2*stixX), 2 ) ) )

                    # RMS wave components
                    Byw = np.sqrt(Byw_sq);
                    Exw = abs(C*Byw * (stixP - n_x*n_x)/(stixP*n_z)); 
                    Eyw = abs(Exw * stixD/(stixS-mu_sq));
                    Ezw = abs(Exw *n_x*n_z / (n_x*n_x - stixP));
                    Bxw = abs(Exw *stixD*n_z /C/ (stixS - mu_sq));
                    Bzw = abs((Exw *stixD *n_x) /(C*(stixX - mu_sq)));

                    # Oblique integration quantities
                    R1 = (Exw + Eyw)/(Bxw+Byw);
                    R2 = (Exw - Eyw)/(Bxw-Byw);
                    w1 = Q_EL/(2*M_EL)*(Bxw+Byw);
                    w2 = Q_EL/(2*M_EL)*(Bxw-Byw);
                    alpha1 = w2/w1;

                    # Resonance modes to consider
                    for mres in np.arange(-SCATTERING_RES_MODES, SCATTERING_RES_MODES+1, 1):

                        t1 = w*w*kz*kz;
                        t2 = pow((mres*wh),2)-w*w;
                        t3 = kz*kz + pow((mres*wh),2)/(pow(C*np.cos(alpha_lc),2));

                        # Pick the direction. I don't know why Jacob picked negative for the 0th mode.
                        if mres==0:
                            direction = -1.*np.sign(kz)
                        else:
                            direction = np.sign(kz)*np.sign(mres)

                        v_para_res = ( direction*np.sqrt(t1 + t2*t3) - w*kz ) / t3;
                        v_tot_res = v_para_res / np.cos(alpha_lc); 
                        E_res = E_EL*( 1.0/np.sqrt( 1.0-(v_tot_res*v_tot_res/(C*C)) ) - 1.0 );
                        E_lo = pow(10, np.log10(E_res) - E_BANDWIDTH)
                        E_hi = pow(10, np.log10(E_res) + E_BANDWIDTH)
                        
                        # Start and stop indices for the range of energies to do:
                        e_starti = np.digitize(E_lo, E_tot_arr)
                        e_endi  =  np.digitize(E_hi, E_tot_arr)
                        # e_starti = np.argmin(np.abs(E_lo - E_tot_arr))
                        # e_endi   = np.argmin(np.abs(E_hi - E_tot_arr))
                        # print e_starti, e_endi

                        # energy bins to calculate at:
                        evec_inds = np.arange(e_starti,e_endi,dtype=int)

                        if len(evec_inds) > 0:
                            v_tot = direction*v_tot_arr[evec_inds]
                            v_para = v_tot*calph
                            v_perp = abs(v_tot*salph)

                            # Relativistic factor
                            gamma = 1.0/np.sqrt(1.0 - pow(v_tot/C, 2))
           
                            alpha2 = Q_EL*Ezw /(M_EL*gamma*w1*v_perp)
                            beta = kx*v_perp/wh

                            wtau_sq = (pow((-1.),(mres-1.)) * w1/gamma * 
                                        ( jn( (mres-1), beta ) - 
                                          alpha1*jn( (mres+1) , beta ) +
                                          gamma*alpha2*jn( mres , beta ) ))

                            T1 = -wtau_sq*(1 + calph*calph/(mres*Y - 1))

                            # Analytical evaluation!
                            if (abs(lat) < 1e-3):        
                                eta_dot = mres*wh/gamma - w - kz*v_para
                                dalpha_eq = np.zeros_like(eta_dot)
                                
                                eta_mask = eta_dot < 10
                                # Bortnik A.31
                                dalpha_eq[eta_mask]  = abs(T1[eta_mask] /v_para[eta_mask])*ds/np.sqrt(2)
                                # Bortnik A.30
                                dalpha_eq[~eta_mask] = abs(T1[~eta_mask]/eta_dot[~eta_mask])*np.sqrt(1-np.cos(ds*eta_dot[~eta_mask]/v_para[~eta_mask]))

                            else:
                                v_para_star = v_para - dv_para_ds*ds/2.0
                                v_para_star_sq = v_para_star*v_para_star

                                # Bortnik A.18 -- part A1
                                AA =  ( (mres/(2.0*v_para_star*gamma))*dwh_ds* 
                                     (1 + ds/(2.0*v_para_star)*dv_para_ds) - 
                                      mres/(2.0*v_para_star_sq*gamma)*wh*dv_para_ds + 
                                      w/(2.0*v_para_star_sq)*dv_para_ds )
                                
                                # Bortnik A.18 -- part A0
                                BB =  ( mres*wh/(gamma*v_para_star)
                                     - mres/(gamma*v_para_star)*dwh_ds*(ds/2.0) * (w/v_para_star)*kz )

                                Farg = (BB + 2.0*AA*ds) / np.sqrt(2.0*np.pi*abs(AA))
                                Farg0 = BB / np.sqrt(2*np.pi*abs(AA))

                                Fs,  Fc  = fresnel(Farg)
                                Fs0, Fc0 = fresnel(Farg0)

                                dFs_sq = pow(Fs - Fs0, 2)
                                dFc_sq = pow(Fc - Fc0, 2)

                                dalpha = np.sqrt( (np.pi/4.0)/abs(AA))*abs(T1/v_para)*np.sqrt(dFs_sq + dFc_sq)
                                alpha_eq_p = np.arcsin( np.sin(alpha_lc+dalpha)*pow(clat,3) / np.sqrt(slat_term) )
                                dalpha_eq  = alpha_eq_p - alpha_eq


                                if direction > 0: 
                                    # print "Norf!"
                                    iono_time = t + abs(ftc_n/v_para)
                                    tt = np.digitize(iono_time, tvec)
                                    for tind in tt:
                                        da_N[fl_ind, evec_inds, tind] += pow(dalpha_eq, 2.)
                                else:
                                    # print "Souf!" 
                                    iono_time = t + abs(ftc_s/v_para)
                                    
                                    tt = np.digitize(iono_time, tvec)
                                    # print tt
                                    # tt_inds, tt_counts = np.unique(tt, return_counts=True)
                                    # print "t_inds:", tt_inds, "t_counts:", tt_counts
                                    # print "shape(tinds)",np.shape(tt_inds), "shape(dalpha)", np.shape(dalpha_eq)
                                    # blergy = (np.outer(pow(dalpha_eq,2.),tt_counts))
                                    # print "blergy", np.shape(blergy)
                                    # print "targ", np.shape(da_S[fl_ind, evec_inds, tt_inds])
                                    for tind in tt:
                                        da_S[fl_ind, evec_inds, tind] += pow(dalpha_eq, 2.)



