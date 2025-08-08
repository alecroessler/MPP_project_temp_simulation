/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#ifndef __FILEH__
#define __FILEH__

#include <sys/time.h>

typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;

#ifdef __cplusplus
extern "C" {
#endif
void compute_temperature(double* T, double* T_new, double* T_diff, double T_amb, int grid_size, double alpha, double dt);
void max_diff_reduction(double* T_diff, double* max_diff, double* block_max, int N);
double max_temp(double* T, int N);
double min_temp(double* T, int N);
double avg_temp(double* T, int N);
void initVector(unsigned int **vec_h, unsigned int size, unsigned int num_bins);
void verify(int iter, double max_temp_T, double min_temp_T, double avg_temp_T);
void startTime(Timer* timer);
void stopTime(Timer* timer);
float elapsedTime(Timer timer);

#ifdef __cplusplus
}
#endif

#define FATAL(msg, ...) \
    do {\
        fprintf(stderr, "[%s:%d] %s\n", __FILE__, __LINE__, ##__VA_ARGS__);\
        exit(-1);\
    } while(0)

#if __BYTE_ORDER != __LITTLE_ENDIAN
# error "File I/O is not implemented for this system: wrong endianness."
#endif

#endif
