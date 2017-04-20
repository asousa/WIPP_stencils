#ifndef wipp_H
#define wipp_H

#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <stdlib.h>
#include <iostream>
// #include <fstream>
// #include <sstream>
// #include <iterator>
// #include <string>
#include "consts.h"
using namespace std;

// #define DLLEXPORT extern "C" __declspec(dllexport)
struct EA_args{
    // int num_lats;
    double lat;
    double alpha_eq;
    double stixP;
    double stixR;
    double stixL;
    double alpha_lc;
    double wh;
    double ds;
    double dv_para_ds;
    double dwh_ds;
    double ftc_n;
    double ftc_s;
};

struct scattering_params{
    size_t NUM_E;
    size_t NUM_T;
    double dt;
    double DE_EXP;
    double E_EXP_BOT;
    double E_EXP_TOP;
    double * v_tot_arr;
};

void Fresnel(double x0, double *FS, double *FC);


extern "C" void ctypes_trial(double *indatav, size_t size, double *outdatav);

extern "C" void calc_scattering(double* crossings, size_t rows, double inp_pwr, EA_args EA,
                    scattering_params params, double* da_N, double* da_S);

// void calc_scattering(double *indata, size_t rows, size_t cols, 
//     EA_args EA,
//     int num_E, int num_T, double *da_N, double *da_S);


#endif
