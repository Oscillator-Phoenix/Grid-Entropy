/*
 * In general, you don't need to modify this file to finish hw1.
 */
#ifndef _INCL_UTILS
#define _INCL_UTILS

#include "time.h"


// getTime gets the local time in nanoseconds.
long getTime() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (long)ts.tv_sec * 1000000000L + ts.tv_nsec;
}

// W_CHK macro is used to check if a file write is successfully or not.
#define W_CHK(call)                                         \
{                                                           \
    const int written = call;                               \
    if (written == 0) {                                     \
        printf("error written\n");                          \
        exit(1);                                            \
    }                                                       \
}                                                           \

#endif