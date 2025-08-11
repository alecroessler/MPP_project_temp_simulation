/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "support.h"

void initVector(unsigned int **vec_h, unsigned int size, unsigned int num_bins)
{
    *vec_h = (unsigned int*)malloc(size*sizeof(unsigned int));

    if(*vec_h == NULL) {
        FATAL("Unable to allocate host");
    }

    for (unsigned int i=0; i < size; i++) {
        (*vec_h)[i] = (rand()%num_bins);
    }

}

void verify(int iter, double max_temp_T, double min_temp_T, double avg_temp_T) {
    const double threshold = 1e-2;  // You can adjust this threshold if needed

    if ((iter == 23754) &&
        (fabs(max_temp_T - 130.00) < threshold) &&
        (fabs(min_temp_T - 25.00) < threshold) &&
        (fabs(avg_temp_T - 52.32) < threshold)) {
        printf("TEST PASSED\n\n");
    } else {
        printf("Verification failed!\n");
        printf("iter = %d, max = %.4f, min = %.4f, avg = %.4f\n",
               iter, max_temp_T, min_temp_T, avg_temp_T);
        exit(1);
    }
}

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

