#include <WIPP_stencil.h>/*
 * FUNCTION: Fresnel
 * -----------------
 * This function calculates the sine and cose Fresnel integrals and 
 * returns the values of either.  The integrals are defined as:
 *
 *    /x           /x
 *    | sin( pi/2*t^2 ) dt           | cos( pi/2*t^2 ) dt
 *    /0           /0
 *
 * This uses the methodology of  Klaus D. Mielenz, "Computation of
 * Fresnel Integrals II" Journal of Research of the National 
 * Institute of Standards and Technology, 105, 589 (2000)
 *
 */
void Fresnel(double x0, double *FS, double *FC)
{

  double fn[12] = { 0.318309844,
        9.34626e-8 , 
        -0.09676631, 
        0.000606222, 
        0.325539361, 
        0.325206461, 
        -7.450551455,
        32.20380908, 
        -78.8035274, 
        118.5343352, 
        -102.4339798,
        39.06207702 } ;

  double gn[12] = { 0.0 ,
        0.101321519, 
        -4.07292e-5, 
        -0.152068115, 
        -0.046292605, 
        1.622793598, 
        -5.199186089, 
        7.477942354, 
        -0.695291507, 
        -15.10996796,
        22.28401942, 
        -10.89968491 };
  
  double cn[12] = { 1.0 ,
        -0.24674011002723,
        0.02818550087789 ,
        -0.00160488313564 ,
        5.407413381408390e-05 ,
        -1.200097255860028e-06,
        1.884349911527268e-08,
        -2.202276925445466e-10, 
        1.989685792418021e-12,
        -1.430918973171519e-14,
        8.384729705118549e-17,
        -4.079981449233875e-19 } ;

  double sn[12] = {    0.52359877559830,
           -0.09228058535804,
           0.00724478420420,
           -3.121169423545791e-04,
           8.444272883545251e-06,
           -1.564714450092211e-07,
           2.108212193321454e-09,
           -2.157430680584343e-11,
           1.733410208887483e-13,
           -1.122324478798395e-15,
           5.980053239210401e-18,
           -2.667871362841397e-20 };

  double xpow, x_sq, fx=0, gx=0, x;
  int n;

  x = fabs(x0);
  *FS = 0.0;
  *FC = 0.0;
  x_sq = x*x;



  if(x<=1.6) {

    *FS =   sn[0]*pow(x,3) +  // it takes longer to write this out
      sn[1]*pow(x,7) +        // but we save valuable CPU cycles!
      sn[2]*pow(x,11) + 
      sn[3]*pow(x,15) + 
      sn[4]*pow(x,19) + 
      sn[5]*pow(x,23) + 
      sn[6]*pow(x,27) + 
      sn[7]*pow(x,31) + 
      sn[8]*pow(x,35) + 
      sn[9]*pow(x,39) + 
      sn[10]*pow(x,43) + 
      sn[11]*pow(x,47)  ; 

    *FC =   cn[0]*x + 
      cn[1]*pow(x,5) +  
      cn[2]*pow(x,9) + 
      cn[3]*pow(x,13) + 
      cn[4]*pow(x,17) + 
      cn[5]*pow(x,21) + 
      cn[6]*pow(x,25) + 
      cn[7]*pow(x,29) + 
      cn[8]*pow(x,33) + 
      cn[9]*pow(x,37) + 
      cn[10]*pow(x,41) + 
      cn[11]*pow(x,45)  ; 

  } else {
      
    
    for(n=0; n<=11; n++) {
      xpow = pow(x, (-2*n-1) );
      fx += fn[n]*xpow;
      gx += gn[n]*xpow;
    }     
    *FC = 0.5 + fx*sin(PI/2*x_sq) - gx*cos(PI/2*x_sq);
    *FS = 0.5 - gx*sin(PI/2*x_sq) - fx*cos(PI/2*x_sq);      
  }
 
  if(x0<0) {
    *FC = -(*FC);
    *FS = -(*FS);
  }
}
