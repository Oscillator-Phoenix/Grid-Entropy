#ifndef _INCL_CORE
#define _INCL_CORE


// The main function would invoke the cudaCallback on each sample. Note that you
// don't have to (and shouldn't) free the space of sample and result by yourself
// since the main function have included the free statements already.
//
// To make the program work, you shouldn't modify the signature of cudaCallback.
void cudaCallback_base(int width, int height, float *sample, float **result);
void cudaCallback_opt (int width, int height, float *sample, float **result);

                                                           \

#endif