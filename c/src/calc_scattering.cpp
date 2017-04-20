#include <WIPP_stencil.h>

void calc_scattering(double* crossings, size_t rows, double inp_pwr, EA_args EA,
                    scattering_params params, double* da_N, double* da_S) {

    // cout << "num_lats: " << EA.num_lats << endl;
    
    // cout << "alpha_lc: ";
    // for (int i=0; i < EA.num_lats; ++i) {
    //     cout << EA.alpha_lc[i] << " ";
    // }
    // cout << endl;
    double lat, slat, clat, L, t, f, pwr, psi, mu, stixP, stixR, stixL, latk;
    double Bxw, Byw, Bzw, Exw, Eyw, Ezw, stixD, stixS, stixA, stixB, stixX;
    double spsi, cpsi, spsi_sq, cpsi_sq, mu_sq, w, damping;
    double Y,  R1, R2, w1, w2, alpha1;
    double n_x, n_z, k, kx, kz, rho1, rho2, Byw_sq;
    double mres, T1, t1, t2, t3;
    double alpha_lc, alpha_eq, ds, wh, dwh_ds, dv_para_ds;
    double direction, v_para_res, v_tot_res, E_res, v_tot;
    int e_starti, e_endi, e_toti, timei;
    double v_para, v_perp, calph, salph, alpha2, beta, wtau_sq;
    double v_para_star, v_para_star_sq;
    double eta_dot, dalpha, dalpha_eq, alpha_eq_p, dalpha_eq_p;
    double flt_const_N, flt_const_S, flt_time;
    double AA, BB, Farg, Farg0, Fs, Fc, Fs0, Fc0, dFs_sq, dFc_sq;

    double gamma, slat_term;


    // double flt_const_N, flt_const_S, flt_time, eta_dot;

    // // int irows= int(rows);
    // // int icols = int(cols);
    // // double(*inptr)[irows][icols] = reinterpret_cast<double(*)[irows][icols]>(indata);
    // // double(*outptr)[irows][icols] = reinterpret_cast<double(*)[irows][icols]>(outdata);
    // // for (size_t i=0; i<rows; ++i) {
    // //     for (size_t j=0; j<cols; ++j) {
    // //         cout << i << ", " << j << endl;
    // //     // index = int((i-1)*cols + j);
    // //     // outdata[i*cols + j] = i*j;
    // //         (*outptr)[i][j] = (*inptr)[i][j]*i*j;
    // //         // cout << arr_ptr[i][j] << " ";
    // //     }
    // //     // cout << endl;
    // // }


    int NUM_E = params.NUM_E;
    int NUM_T = params.NUM_T;
    double DE_EXP = params.DE_EXP;
    double E_EXP_BOT = params.E_EXP_BOT;
    double TIME_STEP = params.dt;

    // cout << " C (params) ----------" <<endl;
    // cout << "NUM_E: " << NUM_E << endl;
    // cout << "NUM_T: " << NUM_T << endl;
    // cout << "DE_EXP: "<< DE_EXP << endl;
    // cout << "E_EXP_BOT " << E_EXP_BOT << endl;
    // cout << "TIME_STEP " << TIME_STEP << endl;

    // 2d input array
    double(*cross_ptr)[rows][6] = reinterpret_cast<double(*)[rows][6]>(crossings);
    // 2d output arrays
    double(*daN_ptr)[NUM_E][NUM_T] = reinterpret_cast<double(*)[NUM_E][NUM_T]>(da_N);
    double(*daS_ptr)[NUM_E][NUM_T] = reinterpret_cast<double(*)[NUM_E][NUM_T]>(da_S);

    double(*v_tot_arr_p)[NUM_E]    = reinterpret_cast<double(*)[NUM_E]>(params.v_tot_arr);

    // cout << "V array: ";
    // for (int i=0; i<NUM_E; ++i) {
    //     cout << (*v_tot_arr_p)[i] << " ";
    // }
    // cout << endl;

    alpha_lc = EA.alpha_lc;
    alpha_eq = EA.alpha_eq;
    wh       = EA.wh;
    ds       = EA.ds;
    dv_para_ds = EA.dv_para_ds;
    dwh_ds   = EA.dwh_ds;
    flt_const_N = EA.ftc_n;
    flt_const_S = EA.ftc_s;
    lat = EA.lat;

    calph = cos(alpha_lc);
    salph = sin(alpha_lc);
    stixP = EA.stixP;
    stixR = EA.stixR;
    stixL = EA.stixL;

    clat = cos(lat*D2R);
    slat = sin(lat*D2R);
    slat_term = sqrt(1. + 3.*slat*slat);
    // cout << "slat term: " << slat_term << endl;
    // cout << " C (EA) ----------------" << endl;
    // cout << "stixP "       << stixP << endl;
    // cout << "stixR "       << stixR << endl;
    // cout << "stixL "       << stixL << endl;
    // cout << "alpha_lc "    << alpha_lc   << endl;
    // cout << "wh "          << wh         << endl;
    // cout << "ds "          << ds         << endl;
    // cout << "dv_para_ds "  << dv_para_ds << endl;
    // cout << "dwh_ds "      << dwh_ds     << endl;
    // cout << "flt_const_N " << flt_const_N << endl;
    // cout << "flt_const_S " << flt_const_S << endl;
    // cout << "alpha_eq "    << alpha_eq   << endl;
    // cout << "lat "         << lat         << endl; 






    // // Loop through entries:
    for (int i=0; i<rows; ++i) {
        t       = (*cross_ptr)[i][0];
        f       = (*cross_ptr)[i][1];
        pwr     = (*cross_ptr)[i][2]*inp_pwr;
        psi     = (*cross_ptr)[i][3];
        mu      = (*cross_ptr)[i][4];
        damping = (*cross_ptr)[i][5];

        // cout << "t: " << t << " f: " << f << " pwr: " << pwr << " mu: " << mu << " psi " << psi << " damp " << damping << endl;

        spsi = sin(psi);
        cpsi = cos(psi);
        spsi_sq = pow(spsi,2);
        cpsi_sq = pow(cpsi,2);
        n_x = mu*fabs(spsi);
        n_z = mu*cpsi;
        mu_sq = mu*mu;
        w = 2.0*PI*f;
        k = w*mu/C;
        kx = w*n_x/C;
        kz = w*n_z/C;
        Y = wh / w ;

        // Stix parameters
        stixS = ( stixR + stixL ) /2.0;
        stixD = ( stixR - stixL ) /2.0;
        stixA = stixS + (stixP-stixS)*cpsi_sq;
        stixB = stixP*stixS+stixR*stixL+(stixP*stixS-stixR*stixL)*cpsi_sq;
        stixX = stixP/(stixP- mu_sq*spsi_sq);

        // Polarization ratios
        rho1=((mu_sq-stixS)*mu_sq*spsi*cpsi)/(stixD*(mu_sq*spsi_sq-stixP));
        rho2 = (mu_sq - stixS) / stixD ;

        // (bortnik 2.28)
        Byw_sq =  2.0*MU0/C*pwr*stixX*stixX*rho2*rho2*mu*fabs(cpsi)/
           sqrt(  pow((tan(psi)-rho1*rho2*stixX),2) + 
           pow( (1+rho2*rho2*stixX), 2 ) );

        // cout << "Byw_sq: " << Byw_sq << endl;

        // RMS wave components
        Byw = sqrt(Byw_sq);
        Exw = fabs(C*Byw * (stixP - n_x*n_x)/(stixP*n_z)); 
        Eyw = fabs(Exw * stixD/(stixS-mu_sq));
        Ezw = fabs(Exw *n_x*n_z / (n_x*n_x - stixP));
        Bxw = fabs(Exw *stixD*n_z /C/ (stixS - mu_sq));
        Bzw = fabs((Exw *stixD *n_x) /(C*(stixX - mu_sq)));

        // Oblique integration quantities
        R1 = (Exw + Eyw)/(Bxw+Byw);
        R2 = (Exw - Eyw)/(Bxw-Byw);
        w1 = Q_EL/(2*M_EL)*(Bxw+Byw);
        w2 = Q_EL/(2*M_EL)*(Bxw-Byw);
        alpha1 = w2/w1;

        // cout << " Byw " << Byw
        // << " Exw " << Exw
        // << " Eyw " << Eyw
        // << " Ezw " << Ezw
        // << " Bxw " << Bxw
        // << " Bzw " << Bzw << endl; 
        //begin MRES loop here
        for(mres=-SCATTERING_RES_MODES; mres <= SCATTERING_RES_MODES; mres++) {
            // get parallel resonance velocity
            t1 = w*w*kz*kz;
            t2 = pow((mres*wh),2)-w*w;
            t3 = kz*kz + pow((mres*wh),2)/(pow(C*cos(alpha_lc),2));

            if(mres==0) {
                direction = -kz/fabs(kz);
            } else {
                direction = kz/fabs(kz) * mres/fabs(mres) ;
            }

            v_para_res = ( direction*sqrt(t1 + t2*t3) - w*kz ) / t3;
            v_tot_res = v_para_res / cos(alpha_lc); 
            E_res = E_EL*( 1.0/sqrt( 1.0-(v_tot_res*v_tot_res/(C*C)) ) -1.0 );

            // if(DEBUG) {printf("t1: %g t2: %g t3: %g v_para_res: %g v_tot_res: %g E_res: %g\n",
            //                    t1,    t2,    t3,    v_para_res,    v_tot_res,    E_res);}
            
            // get starting and ending indices, +-20% energy band
            e_starti = floor((log10(E_res) - E_EXP_BOT - 0.3)/(DE_EXP));
            e_endi   =  ceil((log10(E_res) - E_EXP_BOT + 0.3)/(DE_EXP));

            if(e_endi>NUM_E) e_endi=NUM_E;
            if(e_starti>NUM_E) e_starti=NUM_E;
            if(e_endi<0) e_endi=0;
            if(e_starti<0) e_starti=0;
            

            // begin V_TOT loop here
            for(e_toti=e_starti; e_toti < e_endi; e_toti++) {

                v_tot = direction * ( (*v_tot_arr_p)[e_toti]);
                v_para = v_tot * calph;
                v_perp = fabs(v_tot * salph);
                // cout << "v_perp: " << v_perp << endl;
                gamma = 1.0 / sqrt(1 - pow((v_tot/C),2)); 
                alpha2 = Q_EL*Ezw /(M_EL*gamma*w1*v_perp);
                beta = kx*v_perp / wh ;
                // cout << "w1: " << w1 << " gamma: " << gamma << " beta: " << beta << " alpha1: " << alpha1 << " alpha2 " << alpha2 <<endl;

                wtau_sq = pow((-1),(mres-1)) * w1/gamma * 
                ( jn( (mres-1), beta ) - 
                  alpha1*jn( (mres+1) , beta ) +
                  gamma*alpha2*jn( mres , beta ) ); 
                T1 = -wtau_sq*(1+ ( (calph*calph) / (mres*Y-1) ) );
                // cout << "wtau_sq: " << wtau_sq << " calph: " << calph << " mres: " << mres << " Y: " << Y << endl;
                // Now - start analytical evaluation!!!
              
                if( fabs(lat)< 1e-3) {
                    // Near the equator we can use a simplified expression:
                    eta_dot = mres*wh/gamma - w - kz*v_para;

                    if(fabs(eta_dot)<10) {
                        // Bortnik A.31
                        dalpha_eq = fabs(T1/v_para)*ds/sqrt(2); 
                    } else {
                        // Bortnik A.30
                        dalpha_eq = fabs(T1/eta_dot)*sqrt(1-cos(ds*eta_dot/v_para)); 
                    }

                } else {  
                    
                    v_para_star = v_para - dv_para_ds*ds/2.0;
                    v_para_star_sq = v_para_star * v_para_star;

                    // Bortnik A.18 -- part A1
                    AA = (mres/(2.0*v_para_star*gamma))*dwh_ds* 
                         (1 + ds/(2.0*v_para_star)*dv_para_ds) - 
                          mres/(2.0*v_para_star_sq*gamma)*wh*dv_para_ds + 
                          w/(2.0*v_para_star_sq)*dv_para_ds ;

                    // // Bortnik A.18 -- part A0   -- THIS DOES NOT MATCH THE THESIS
                    // BB = mres/(gamma*v_para_star)*wh - 
                    //      mres/(gamma*v_para_star)*dwh_ds*(ds/2.0) -
                    //      w/v_para_star - kz;

                    // Bortnik A.18 -- part A0
                    BB =   mres*wh/(gamma*v_para_star)
                         - mres/(gamma*v_para_star)*dwh_ds*(ds/2.0) * (w/v_para_star)*kz;


                    // Evaluate Bortnik A.26 -- integration performed thru Fresnel functions
                    Farg = (BB + 2*AA*ds) / sqrt(2*PI*fabs(AA));
                    Farg0 = BB / sqrt(2*PI*fabs(AA));  
                    
                    Fresnel(Farg, &Fs, &Fc);
                    Fresnel(Farg0, &Fs0, &Fc0);
                    
                    dFs_sq = pow((Fs - Fs0),2);
                    dFc_sq = pow((Fc - Fc0),2);
                    
                    // cout << "AA " << AA << " T1 " << T1 << " v_para " << v_para << " Dfs_sq " << dFs_sq << " dFc_sq " << dFc_sq << endl; 
                    dalpha = sqrt(PI/4/fabs(AA))*fabs(T1/v_para)*sqrt(dFs_sq+dFc_sq);
                    
                    // Map the local change in pitch angle to the equivalent
                    // pitch angle at the equator:  (still using dipole model here)
                    // alpha_eq_p = asin( sin(alpha_lc+dalpha)*pow(clat,3) / 
                    //            sqrt(slat_term) );
                    // alpha_eq_p = asin( sin(alpha_lc + dalpha)*EA.Bo_ratio);
                    alpha_eq_p = asin( sin(alpha_lc+dalpha)*pow(clat,3) / sqrt(slat_term) );
                    // cout << "alpha_eq_p: " << alpha_eq_p << " alpha_eq: " << alpha_eq << endl;

                    dalpha_eq = alpha_eq_p - alpha_eq;

                    
                    // if (isnan(fabs(dalpha_eq))) {
                    //     cout << "NaN: ";

                    //     cout << "direction: " << direction << " v_to_arr[e_toti]: " << v_tot_arr[e_toti] << " "; 
                    //     printf("w1: %g gamma: %g alpha1: %g beta: %g alpha2: %g v_perp: %g v_tot: %g salph: %g e_toti: %g\n",
                    //             w1,    gamma,    alpha1,    beta,    alpha2,    v_perp,    v_tot,    salph,    e_toti);


                    //     // printf("wtau_sq: %g calph: %g mres: %g Y: %g\n",
                    //     //         wtau_sq,    calph,    mres,    Y);
                    //     // printf("AA: %g T1: %g v_para: %g dFs_sq: %g dFc_sq: %g\n",
                    //     //         AA,     T1,    v_para,   dFs_sq,    dFc_sq);
                    //     // // printf("alpha_eq_p: %g alpha_lc: %g dalpha: %g Bo_ratio: %g\n",
                    //     //         alpha_eq_p,    alpha_lc,    dalpha,  EA.Bo_ratio);
                    //     // printf("flt_time: %g dalpha_eq: %g alpha_eq_p: %g alpha_eq: %g\n",
                    //     //         flt_time,    dalpha_eq,    alpha_eq_p,    alpha_eq);
                        
                    //     break;
                    // }
                }

                if(direction>0) {
                    flt_time = fabs(flt_const_N/v_para);
                } else {
                    flt_time = fabs(flt_const_S/v_para);
                }
                 
                // if (DEBUG) {printf("flt_time: %g dalpha_eq: %g alpha_eq_p: %g alpha_eq: %g\n",
                //                     flt_time,    dalpha_eq,    alpha_eq_p,    alpha_eq);}

                // Get time index into output array
                timei = round((t + flt_time)/TIME_STEP);

                if (timei < NUM_T) {
                    // Save it!
                    if (direction > 0) {
                        (*daN_ptr)[e_toti][timei] += dalpha_eq*dalpha_eq;
                    } else {
                        (*daS_ptr)[e_toti][timei] += dalpha_eq*dalpha_eq;
                    }
                } else {
                    // Do we want to track total scattering after TMAX?
                }
            } // v_para
        } // mres loop






    }



}