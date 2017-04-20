#include <WIPP_stencil.h>


void ctypes_trial(double *indatav, size_t size, double *outdatav) {
    cout << "hi from herpderpderpherp!" << endl;
    size_t i;
    for (i = 0; i < size; ++i)
        outdatav[i] = indatav[i] * i;

    }