#include <WIPP_stencil.h>

// using namespace std;
// using namespace Eigen;

#define   NAPe    2.71828182845905

// -----------------------------------------------
// GLOBAL VARIABLES:
// -----------------------------------------------

// double      L_TARG;
int nancounter;

// -----------------------------------------------
// Constants to perform Gauss quadrature integration
// -----------------------------------------------
double beta5[] ={0.56888889, 0.47862867,  0.47862867, 0.23692689,  0.23692689};
double t5[] =   {0.,         0.53846931, -0.53846931, 0.90617985, -0.90617985};


/*
 * FUNCTION: compFlux
 * ------------------
 * This function will open a file with the appropriate hemisphere and
 * L-shell inserted into the name, calculate the precipitated flux  
 * due to each cell in the dAlpha_RMS matrix and write it into the 
 * file.
 *
 */

// void compFlux(double arr[NUM_E][NUM_TIMES], double out_lat, double out_lon, int k, string out_dir, 
//               double J[100][100], int flux_dist, int alpha_dist)
// {

void calc_flux(double *arr, flux_params params, double* J, double L_sh, double *phi_out) {
  // Calculate Phi for northern and southern hemispheres
  // (Bortnik 5.2).
  // Units are [#/cm^2/keV/sec]

  FILE *phiPtr, *QPtr, *NPtr, *alphaPtr;
  int ei, ti, i, nout;
  double da_pk, P, Q, epsm, alpha_eq, v, crunch;
  double I=0.0, x, g, field, Phi_p, b=1.0, vcm, fcgs;
  double gamma;
  double Phi_float, alpha;
  // char *NS, PhiFile[128], QFile[128], NFile[128], alphaFile[128];

  double t1, t2, t3, t4, t6;
  double max_alpha = 0;

  // Unpack the parameters structure for cleanliness:
  int NUM_E = params.NUM_E;
  int NUM_T = params.NUM_T;
  double E_EXP_BOT = params.E_EXP_BOT;
  double E_EXP_TOP = params.E_EXP_TOP;
  double DE_EXP = params.DE_EXP;
  int flux_dist = params.flux_dist;
  int alpha_dist = params.alpha_dist;
  int n_JL = params.n_JL;
  int n_JE = params.n_JE;

  double v_tot_arr[NUM_E], E_tot_arr[NUM_E], dE_arr[NUM_E];
  double Jdiff[NUM_E];

  // Input space
  double(*da_ptr)[NUM_E][NUM_T] = reinterpret_cast<double(*)[NUM_E][NUM_T]>(arr);
  // Output space
  double(*phi_out_ptr)[NUM_E][NUM_T] = reinterpret_cast<double(*)[NUM_E][NUM_T]>(phi_out);


  // double(*Jdd)[n_JL][n_JE] = reinterpret_cast<double(*)[n_JL][n_JE]>(J);

  // for (int row=0; row < n_JL; ++row) {
  //   cout << row << ": ";
  //   for (int col=0; col < n_JE; ++col) {
  //     cout << (*Jdd)[row][col] << " ";
  //   }
  //   cout << endl;
  // }



  
  // // Echo the input parameters:
  // cout << "--------- flux methods ---------" << endl;
  // cout << "NUM_E " << NUM_E << endl;
  // cout << "NUM_T " << NUM_T << endl;
  // cout << "dt "    << params.dt << endl;
  // cout << "DE_EXP " << DE_EXP << endl;
  // cout << "E_EXP_BOT " << E_EXP_BOT << endl;
  // cout << "E_EXP_TOP " << E_EXP_TOP << endl;
  // cout << "alpha dist " << alpha_dist << endl;
  // cout << "flux dist " << flux_dist << endl;
  // cout << "n_JL " << n_JL << endl;
  // cout << "n_JE " << n_JE << endl;



  // cout << "L-shell: " << L_sh << endl;
  epsm = (1/L_sh)*(R_E+H_IONO)/R_E;
  // (1/sin(alpha_lc)^2) -- geometric focusing term (Bortnik 5.2)
  crunch  = sqrt(1+3*(1-epsm))/pow(epsm,3); 
  // Loss-cone angle at equator:
  alpha_eq  = asin(sqrt( 1.0/crunch ));
  
  // printf("alpha_eq: %g\n",R2D*alpha_eq);
  // printf("crunch: %g\n",crunch);
  // Load flux file:
  // readJ(J, flux_filename);

  // Precalculate energy and velocity values
  // cout << "calculating energy / velocity vectors \n";
  for(i=0; i<NUM_E; i++) {
    E_tot_arr[i] = pow(10, (E_EXP_BOT+(DE_EXP/2)+DE_EXP*i) ); //E in eV
    // Jdiff[i] = getJdiff( J, n_JL, n_JE, E_tot_arr[i], alpha_eq, L_sh );
    v_tot_arr[i] = C*sqrt(1 - pow( (E_EL/(E_EL+E_tot_arr[i])) ,2) );
    // Energy differential dE in keV
    dE_arr[i] = 1e-3 * pow(10, (E_EXP_BOT+ (DE_EXP/2))) * 
      exp(DE_EXP*i / log10(NAPe)) * DE_EXP / log10(NAPe);
  }

  // Precalculate Jdiff:
  // cout << "getting Jdiff\n";
  for(i=0; i<NUM_E; i++) {

    Jdiff[i] = getJdiff( J, n_JL, n_JE, E_tot_arr[i], alpha_eq, L_sh );
    // cout << Jdiff[i] << " ";
  }

  // cout << "Doin thangs\n";
  // Loop over energies:
  for(ei=0; ei<NUM_E; ei++) {
     // cout << endl << "Ei: " <<  ei << endl;
    // Get flux magnitude at this energy and L-shell:
    switch(flux_dist) {
      case 0:
        // AE8-file distribution
        b = Jdiff[ei]*1000.;
        if (isnan(b)) {
            cout << "b isnan: " << b << " ei: " << ei << endl;
            cout << "Jdiff :" << Jdiff[ei-1] << ", " << Jdiff[ei] << ", " << Jdiff[ei + 1] << endl;
            cout << "L_sh: " << L_sh << " alpha_eq: " << alpha_eq << "Ein: " << E_tot_arr[ei] << endl;
         }
        break;
      case 1:
        // Suprathermal flux distribution
        // (Bortnik 5.4, Bell 2002 distribution)
        v = v_tot_arr[ei];
        vcm = v*100;      // v in cm for distrib fn calculation
        gamma = 1.0/sqrt( 1 - v*v/(C*C) );
        fcgs =  4.9e5/pow( (vcm*gamma) ,4) - 
                8.3e14/pow( (vcm*gamma) ,5) + 
                5.4e23/pow( (vcm*gamma) ,6);
     
        b = (v*v/M_EL)*pow( sqrt(1 - (v*v)/(C*C)), 3) * 1.6e-8 * fcgs;
        break;
      case 2:
        // Flat distribution (for 'propensity' study)
        // (omnidirectional flux -> differential flux) 
        b = 1.0/( PI * cos(alpha_eq) * (PI - 2*alpha_eq) );
; 
        break;
      default: 
        printf("no flux distribution selected!\n");
        exit(0);
      }

    // cout << "b is: " << b << endl;

    // Loop over timesteps:
    for(ti=0; ti<NUM_T; ti++) {
      // cout << ti << " ";
      // alpha = sqrt( arr[ei*NUM_TIMES+ti] );
      // alpha = sqrt(arr[ei][ti]);
      alpha = sqrt( (*da_ptr)[ei][ti] );

      if (alpha > max_alpha) { max_alpha = alpha; }

      if (alpha > 0) {

        // printf("ei: %d ti: %d alpha: %g\n",ei, ti, alpha);
      }
      // nout=fwrite(&alpha, sizeof(float), 1, alphaPtr);      

      // Peak change in pitch-angle at this time and energy:
      da_pk = sqrt(2.0)*alpha;  // sqrt(2)*alpha_RMS = peak

      // Integrate the distribution from (alpha = 0 to alpha_lc) 
      // using Gaussian Quadrature (5th order)

      // First we need to convert our limits from (alpha_lc - da_pk,  alpha_lc) to (0, 1):
      P = da_pk/2.;             //[ alpha_lc - (alpha_lc-da_pk) ] /2
      Q = alpha_eq - da_pk/2.;  //[ alpha_lc + (alpha_lc-da_pk) ] /2

      I = 0.0;


      if(da_pk != 0) {

        // Integrate wrt alpha:
        for(i=0; i<5; i++) {
          x = P*t5[i] + Q ;

          // Select which distribution function in pitch angle to use:
          switch(alpha_dist) {
            case 0:
              // Sine (ramp) distribution
              // g = (P/PI)*sin(2.0*x)*((x - alpha_eq)*( asin((x-alpha_eq)/da_pk)+ (PI/2.0) ) +
                // sqrt(da_pk*da_pk-pow((x-alpha_eq),2)));
              // same thing, but simplified to ditch numerical errors when t1 is small:
              t1 = (x - alpha_eq);
              t2 = fabs(da_pk)*sqrt(1 -0.25*pow(t5[i] - 1, 2));
              g = (P/PI)*sin(2.0*x)*(t1*( asin(t5[i]/2. - 0.5)+ (PI/2.0) ) + t2);
              break;
            case 1:
              // Square distribution
              // g = (P/PI) * sin(2.0*x) * (asin((x-alpha_eq)/da_pk)+ (PI/2.0) );
              // same thing - but avoids NaNs when da_pk is very very small:
              g = (P/PI) * sin(2.0*x) * (asin(t5[i]/2. - 0.5) + PI/2.0);   
              break;
            default:
              printf("no pitch-angle distribution selected\n");
              exit(0);
            }
        I += ( beta5[i]*g );

        } // for(i ... ) -> Gauss quad integration
      } // if da_pk != 0

      Phi_p = PI*crunch*b*I;
      
      if (isnan(Phi_p)){ 
        // cout << "Phi_p isnan at: " <<
        // " b=" << b <<
        // " crunch= " << crunch <<
        // " I= " << I << endl;  
        nancounter=nancounter + 1;
        
      };
      (*phi_out_ptr)[ei][ti] = Phi_p;

        // nout=fwrite(&Phi_p, sizeof(double), 1, phiPtr);
      // nout=fwrite(&Phi_float, sizeof(float), 1, phiPtr);
      
    } // for(ti ... )
  } // for(ei ... )

  // printf("Total NaNsies: %i\n",nancounter);

  // fclose(phiPtr);
  // fclose(alphaPtr);

  // printf("Total NaNs: %i\n",nancounter);
  // printf("max pitch-angle deflection: %g deg\n",R2D*max_alpha);

}









/*
 * FUNCTION: getJdiff
 * ------------------+
 * Using the AE8 data stored in J, calculate the differential flux 
 * by taking the (energy) derivative of the integral flux, dividing
 * by the total solid angle and extrapolating down in energy if 
 * need be.
 *
 */
double  getJdiff(double *Jp, int n_JL, int n_JE, double E, double alpha_lc, double L_sh)
{
  int row, i, botE;  //  topCol,
  int col;
  double J1, J2, I, x1, x2, y1, y2, m, c, x_ext, y_ext, J_ext;

  double(*J)[n_JL][n_JE] = reinterpret_cast<double(*)[n_JL][n_JE]>(Jp);

  // //print energies:
  // cout << "Energies: " << endl;
  // for (int k=0; k<n_JE; ++k) {
  //   cout << (*J)[0][k+1] << " ";
  // }
  // cout << endl << " L-shells: " << endl;
  // for (int k=0; k<n_JL; ++k) {
  //   cout << (*J)[k+1][0] << " ";
  // }
  // cout << endl;
  // row = (int)floor((L_sh + 0.11 - (*J)[1][0])/0.1); // to make sure! 
    // row = (int)floor((L_sh - (*J)[1][0])*10.);

  // find row:
  for (int i=1; i < n_JL; ++i) {
    if ((*J)[i][0] > L_sh) {
      row =i;
      break;
    }
    row = n_JL - 1;  // hit maximum
  }

  // find col:
  for (int i=1; i < n_JE; ++i) {
    if ((*J)[0][i]*1e6 > E) {
      col =i;
      break;
    }
    col = n_JE - 1; // Hit maximum
  }

  // cout << "row = " << row << ", col = " << col << endl;
  I = PI * cos(alpha_lc) * (PI - 2*alpha_lc);

  // // Find column corresponding to highest energy value
  // for(i=0; i<100; i++) {
  //   if((*J)[0][i+1] < 0.01) { 
  //     topCol = i; 
  //     break; 
  //   }
  // }


  // Case 1. E < E[0]; extrapolate down
  // -------------------
  if (col <= 1) {
    // cout << "case 1" << endl;
    // diff flux @ 100 keV and 200 keV
    J1 = 1e-6*fabs( (*J)[row][2] - (*J)[row][1]) / ((*J)[0][2] - (*J)[0][1]); 
    J2 = ((1e-6*fabs((*J)[row][3] - (*J)[row][2]) / ((*J)[0][3] - (*J)[0][2])) 
      + J1 )/2; // central difference

    // do extrapolation in log-log space for best fit 
    x1 = log10( (*J)[0][1]*1e6 );
    x2 = log10( (*J)[0][2]*1e6 );
    y1 = log10( J1 );
    y2 = log10( J2 );

    m = (y2-y1)/(x2-x1);            // gradient of line
    c = (y1*x2 - y2*x1)/(x2-x1) ;   // offset of line, i.e.
    
    // y = m*x + c
    x_ext = log10( E );
    y_ext = m*x_ext + c;
    J_ext = pow(10, y_ext);

    if (isnan(J_ext)) {
      cout << "section 1 : ";
      cout << "J_ext is nan at: " << row << "," << E << endl;
    }
    return (J_ext/I);

  } 

  // Case 2. E > E[n_JE]; extrapolate up
  if (col >= (n_JE - 1)) {
    // cout << "E: " << E << " ";
    // cout << "case 2" << endl;
    // cout << "col: " << col << endl;
    // If flux at 7 Mev = 0, flux above it is zero too
    if( (*J)[row][n_JE-1]==0)  { return 0; }

    // Otherwise need to extrapolate as in case 1.
    // diff flux @ 6.5 MeV and 7 MeV
    J2 = 1e-6*fabs( (*J)[row][col] - (*J)[row][col-1] ) 
      / ((*J)[0][col] - (*J)[0][col-1]); 

    J1 = ((1e-6*fabs( J[row][col-1] - J[row][col-2]) / 
       ((*J)[0][col-1] - (*J)[0][col-2]) ) + J2 )/2; // cdiff

    // do extrapolation in log-log space for best fit 
    x1 = log10( (*J)[0][col-1]*1e6 );
    x2 = log10( (*J)[0][col]*1e6 );
    y1 = log10( J1 );
    y2 = log10( J2 );

    m = (y2-y1)/(x2-x1);        // gradient of line
    c = (y1*x2 - y2*x1)/(x2-x1) ;   // offset of line, i.e.
                    // y = m*x + c
    x_ext = log10( E );
    y_ext = m*x_ext + c;
    J_ext = pow(10, y_ext);

    if(J_ext < 1e-10 ) J_ext = 0.0;

    // cout << " x1: " << x1;
    // cout << " x2: " << x2;
    // cout << " y1: " << y1;
    // cout << " y2: " << y2;
    // cout << " m: " << m;
    // cout << " c: " << c;
    // cout << " x_ext: " << x_ext;
    // cout << " y_ext: " << y_ext;
    // cout << " J_ext: " << J_ext << endl;

    // cout << "Normal case: " << J_ext/I << endl;
    return (J_ext/I);
  }

  // Case 3. In band; interpolate.
    // cout << "case 3" << endl;
    // Find column corresponding lower energy value
    // for(i=1; i<100; i++) {
    //   if( ((*J)[0][i+1]*1e6) > E ) { 
    // botE = i; 
    // break; 
    //   }
    // }
    botE = col - 1;

    // central diff flux @ lower and higher energies
    J1 = ( (1e-6 * fabs( (*J)[row][botE] - (*J)[row][botE-1] )
        / ( (*J)[0][botE] - (*J)[0][botE-1] ) ) + 
       (1e-6 * fabs( (*J)[row][botE+1] - (*J)[row][botE] )
        / ( (*J)[0][botE+1] - (*J)[0][botE] ) )  ) / 2;

    J2 = ( (1e-6 * fabs( (*J)[row][botE+1] - (*J)[row][botE] )
        / ( (*J)[0][botE+1] - (*J)[0][botE] ) ) + 
       (1e-6 * fabs( (*J)[row][botE+2] - (*J)[row][botE+1] )
        / ( (*J)[0][botE+2] - (*J)[0][botE+1] ) )  ) / 2;

    if(botE == 1)
      J1 =  (1e-6 * fabs( (*J)[row][botE+1] - (*J)[row][botE] )
          / ( (*J)[0][botE+1] - (*J)[0][botE] ) );
    
    if(botE == (n_JE-2))
      J2 = (1e-6 * fabs( (*J)[row][botE+1] - (*J)[row][botE] )
        / ( (*J)[0][botE+1] - (*J)[0][botE] ) );
    



    // If J1 = J2 = 0, interpolated value also 0
    if( J1==0 && J2==0 ) return 0;



    // If only J2 = 0, do linear interpolation
    if( J2 == 0 ) {
      J_ext = J1*( ( (*J)[0][botE+1]-(E*1e-6) )/
           ( (*J)[0][botE+1] - (*J)[0][botE] ) );
      return (J_ext/I);
    }



    // Otherwise interpolate as in case 1 (log-log space)

    x1 = log10( (*J)[0][botE]*1e6 );
    x2 = log10( (*J)[0][botE+1]*1e6 );
    y1 = log10( J1 );
    y2 = log10( J2 );

    m = (y2-y1)/(x2-x1);        // gradient of line
    c = (y1*x2 - y2*x1)/(x2-x1) ;   // offset of line, i.e.
                    // y = m*x + c
    x_ext = log10( E );
    y_ext = m*x_ext + c;
    J_ext = pow(10, y_ext);

    if (isnan(J_ext)) {
      cout << "section 3 : ";
      cout << "J_ext is nan at: " << row << ", "  << E << endl;
      cout <<" botE: " << botE << endl;
      cout << x1 << ", " << x2 << ", " << y1 << ", " << y2 << ", ";
      cout << m << ", " << c << ", " << x_ext << ", " << y_ext << ", " << J_ext << endl;
    }

    return (J_ext/I);
  }





  // // // Case 1. E < 100 keV
  // // // -------------------

  // // if( E <= 1e5 ) {
 
  // //   // diff flux @ 100 keV and 200 keV
  // //   J1 = 1e-6*fabs( (*J)[row][2] - (*J)[row][1]) / ((*J)[0][2] - (*J)[0][1]); 
  // //   J2 = ((1e-6*fabs((*J)[row][3] - (*J)[row][2]) / ((*J)[0][3] - (*J)[0][2])) 
  // //     + J1 )/2; // central difference

  // //   // do extrapolation in log-log space for best fit 
  // //   x1 = log10( (*J)[0][1]*1e6 );
  // //   x2 = log10( (*J)[0][2]*1e6 );
  // //   y1 = log10( J1 );
  // //   y2 = log10( J2 );

  // //   m = (y2-y1)/(x2-x1);            // gradient of line
  // //   c = (y1*x2 - y2*x1)/(x2-x1) ;   // offset of line, i.e.
    
  // //   // y = m*x + c
  // //   x_ext = log10( E );
  // //   y_ext = m*x_ext + c;
  // //   J_ext = pow(10, y_ext);

  // //   // if (isnan(J_ext)) {
  // //   //   cout << "section 1 : ";
  // //   //   cout << "J_ext is nan at: " << row << "," << E << endl;
  // //   // }
  // //   return (J_ext/I);

  // // }


  

  // // Case 2. E > 7 MeV
  // // -----------------

  // if( E >= 7e6 ) {
  
  //   // If flux at 7 Mev = 0, flux above it is zero too
  //   if( (*J)[row][topCol]==0 )  return 0;

  //   // Otherwise need to extrapolate as in case 1.
  //   // diff flux @ 6.5 MeV and 7 MeV
  //   J2 = 1e-6*fabs( (*J)[row][topCol] - (*J)[row][topCol-1] ) 
  //     / ((*J)[0][topCol] - (*J)[0][topCol-1]); 

  //   J1 = ((1e-6*fabs( J[row][topCol-1] - J[row][topCol-2]) / 
  //      ((*J)[0][topCol-1] - (*J)[0][topCol-2]) ) + J2 )/2; // cdiff

  //   // do extrapolation in log-log space for best fit 
  //   x1 = log10( (*J)[0][topCol-1]*1e6 );
  //   x2 = log10( (*J)[0][topCol]*1e6 );
  //   y1 = log10( J1 );
  //   y2 = log10( J2 );

  //   m = (y2-y1)/(x2-x1);            // gradient of line
  //   c = (y1*x2 - y2*x1)/(x2-x1) ;   // offset of line, i.e.
  //                   // y = m*x + c
  //   x_ext = log10( E );
  //   y_ext = m*x_ext + c;
  //   J_ext = pow(10, y_ext);

  //   if(J_ext < 1e-10 ) J_ext = 0.0;

  //   return (J_ext/I);
  // }


  // // Case 3. 100 keV < E < 7 MeV
  // if( E<7e6 && E>1e5 ) {


  //   // Find column corresponding lower energy value
  //   for(i=1; i<100; i++) {
  //     if( ((*J)[0][i+1]*1e6) > E ) { 
  //   botE = i; 
  //   break; 
  //     }
  //   }


  //   // central diff flux @ lower and higher energies
  //   J1 = ( (1e-6 * fabs( (*J)[row][botE] - (*J)[row][botE-1] )
  //       / ( (*J)[0][botE] - (*J)[0][botE-1] ) ) + 
  //      (1e-6 * fabs( (*J)[row][botE+1] - (*J)[row][botE] )
  //       / ( (*J)[0][botE+1] - (*J)[0][botE] ) )  ) / 2;

  //   J2 = ( (1e-6 * fabs( (*J)[row][botE+1] - (*J)[row][botE] )
  //       / ( (*J)[0][botE+1] - (*J)[0][botE] ) ) + 
  //      (1e-6 * fabs( (*J)[row][botE+2] - (*J)[row][botE+1] )
  //       / ( (*J)[0][botE+2] - (*J)[0][botE+1] ) )  ) / 2;

  //   if(botE == 1)
  //     J1 =  (1e-6 * fabs( (*J)[row][botE+1] - (*J)[row][botE] )
  //         / ( (*J)[0][botE+1] - (*J)[0][botE] ) );
    
  //   if(botE == (topCol-1))
  //     J2 = (1e-6 * fabs( (*J)[row][botE+1] - (*J)[row][botE] )
  //       / ( (*J)[0][botE+1] - (*J)[0][botE] ) );
    



  //   // If J1 = J2 = 0, interpolated value also 0
  //   if( J1==0 && J2==0 ) return 0;



  //   // If only J2 = 0, do linear interpolation
  //   if( J2 == 0 ) {
  //     J_ext = J1*( ( (*J)[0][botE+1]-(E*1e-6) )/
  //          ( (*J)[0][botE+1] - (*J)[0][botE] ) );
  //     return (J_ext/I);
  //   }



  //   // Otherwise interpolate as in case 1 (log-log space)

  //   x1 = log10( (*J)[0][botE]*1e6 );
  //   x2 = log10( (*J)[0][botE+1]*1e6 );
  //   y1 = log10( J1 );
  //   y2 = log10( J2 );

  //   m = (y2-y1)/(x2-x1);        // gradient of line
  //   c = (y1*x2 - y2*x1)/(x2-x1) ;   // offset of line, i.e.
  //                   // y = m*x + c
  //   x_ext = log10( E );
  //   y_ext = m*x_ext + c;
  //   J_ext = pow(10, y_ext);

  //   if (isnan(J_ext)) {
  //     cout << "section 4 : ";
  //     cout << "J_ext is nan at: " << row << ", "  << E << endl;
  //     cout << x1 << ", " << x2 << ", " << y1 << ", " << y2 << ", ";
  //     cout << m << ", " << c << ", " << x_ext << ", " << y_ext << ", " << J_ext << endl;
  //   }

  //   return (J_ext/I);
  // }

// }


// -----------------------------------------------
// UTILITIES
// -----------------------------------------------

/*
 * FUNCTION: readJ
 * ---------------
 * This function simply looks for a file 'filename' and reads 
 * it in.  The columns are energies and the rows are L-shells.
 * The first column is just a list of L-shells and the first row 
 * is just a list of energies (in MeV).
 *
 */
void readJ(double J[][100], string filename)
{
  // char *filename;
  FILE *filePtr;
  int i,j;
  int lines=0;
  int cols=0;
  char ch;
  // filename = "EQFLUXMA.dat";

  if( (filePtr = fopen( filename.c_str() ,"r")) == NULL ) {
    cout << "Error opening the flux file! path: " << filename.c_str() << endl;
    exit(1);
  }
  
  // Find number of lines in file:
  while(!feof(filePtr))
  {
    ch = fgetc(filePtr);
    if(ch == '\n') { lines++; }
    if(ch == ' ')  {  cols++; }
  }
  cols = cols/lines/2;  // Two spaces between each character
  // Back to the top
  rewind(filePtr);

  cout << "flux file: lines: " << lines << " cols: " << cols << endl;

  // INITIALIZE
  for(i=0; i<100; i++) {
    for(j=0; j<100; j++) {
      J[i][j] = 0.0;
    }
  }

  // READ IN VALUES
  // for(i=0; i<47; i++) {   // number of L-shells (rows)
    // for(j=0; j<31; j++) { // Number of energies (cols)
  for(i=0; i<=lines; i++) {   // number of L-shells (rows)
    for(j=0; j<=cols; j++) { // Number of energies (cols)
      fscanf(filePtr, "%le", &(J[i][j]));
    }
  }

  fclose(filePtr);
}

