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

void verify(iter, max_temp_T, min_temp_T, avg_temp_T); {

  if (( iter == 23754) && (max_temp_T == 100.0) && (min_temp_T == 0.0) && (avg_temp_T == 50.0)) {
    printf("TEST PASSED\n\n");
  } else {
    printf("Verification failed!\n");
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

