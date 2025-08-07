#include <stdio.h>
#include <stdint.h>
#include "support.h"
#include "kernel.cu"




// Set up parameters
const int GRID_SIZE = 256;
const int total_size = GRID_SIZE * GRID_SIZE;
const char* POWER_MAP_FILE = "/home/roesslera/code/MPP_project_temp_simulation/data/power_map_256.csv";
const double T_amb = 25;  // Ambient temperature in Celcius
const int ITERATIONS = 50000;
const double DIE_WIDTH_M = 0.016;
const double h = DIE_WIDTH_M / GRID_SIZE;  
const double k = 150.0; // thermal conductivity (using silicon)


int load_power_map(const char* filename, double* q) {
    // Confirm file opens
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file:\n");
        return 1;
    }

    // Read file data
    for (int i = 0; i < total_size; i++) {
        fscanf(file, "%lf,", &q[i]);
    }

    fclose(file);
    return 0;
}




int main(int argc, char* argv[])
{
    Timer timer, total_timer;
    startTime(&total_timer);
    cudaError_t cuda_ret;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    double *q_h, *T_h, *T_new_h;

    q_h = (double*) malloc( sizeof(double) * total_size );
    T_h = (double*) malloc( sizeof(double) * total_size );
    T_new_h = (double*) malloc( sizeof(double) * total_size );

    for (unsigned int i=0; i < total_size; i++) { T_new_h[i] = T_amb; T_h[i] = T_amb; }
    if (load_power_map(POWER_MAP_FILE, q_h) != 0) {
        fprintf(stderr, "Failed to load power map.\n");
        return 1;
    }

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    double *q_d, *T_d, *T_new_d;

    // CUDA device variables for q, T, and T_new
    cuda_ret = cudaMalloc((void**)&q_d, sizeof(double)* total_size);
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for q_d");

    cuda_ret = cudaMalloc((void**)&T_d, sizeof(double)* total_size);
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for T_d");

    cuda_ret = cudaMalloc((void**)&T_new_d, sizeof(double)* total_size);
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for T_new_d");



    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    // Copy q, T, and T_new from host to device
    cuda_ret = cudaMemcpy(q_d, q_h, sizeof(double)*total_size, cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy q from host to device");
    cuda_ret = cudaMemcpy(T_d, T_h, sizeof(double)*total_size, cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy T from host to device");
    cuda_ret = cudaMemcpy(T_new_d, T_new_h, sizeof(double)*total_size, cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy T_new from host to device");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel ---------------------------
    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);
    
    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((GRID_SIZE + blockDim.x - 1) / blockDim.x, (GRID_SIZE + blockDim.y - 1) / blockDim.y);


    

    // Launch the kernel
    int iter;
    for (iter = 0; iter < ITERATIONS; iter++) {
        compute_temperature<<<gridDim, blockDim>>>(T_d, T_new_d, q_d, k, GRID_SIZE, h, T_amb);
        cuda_ret = cudaGetLastError();
        if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");

        // Copy T and T_new to host to check convergence
        cudaMemcpy(T_h, T_d, sizeof(double) * total_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(T_new_h, T_new_d, sizeof(double) * total_size, cudaMemcpyDeviceToHost);

        double max_change = max_abs_diff(T_h, T_new_h, total_size);

        if (max_change < 1e-3) {
            printf("Converged after %d iterations\n", iter);
            break;
        }
        
        if (iter % 100 == 0) {
            printf("Iteration %d: max change = %.5f\n", iter, max_change);
        }
        
        // Swap T and T_new pointers
        double* temp = T_d;
        T_d = T_new_d;
        T_new_d = temp;
    }

    cuda_ret = cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMemcpy(T_new_h, T_new_d, sizeof(double)*total_size, cudaMemcpyDeviceToHost);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory from device");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    max_temp_T = max_temp(T_new_h, GRID_SIZE);
    min_temp_T = min_temp(T_new_h, GRID_SIZE);
    avg_temp_T = avg_temp(T_new_h, GRID_SIZE);

    printf("Max Temp: %.2f C\n", max_temp_T);
    printf("Min Temp: %.2f C\n", min_temp_T);
    printf("Avg Temp: %.2f C\n", avg_temp_T);

    // Verify correctness -----------------------------------------------------
    
    printf("Verifying results..."); fflush(stdout);

    verify(iter, max_temp_T, min_temp_T, avg_temp_T);
    
    // Free memory ------------------------------------------------------------

    free(q_h);
    free(T_h);
    free(T_new_h);

    // Free device variables
    cudaFree(q_d);
    cudaFree(T_d);
    cudaFree(T_new_d);

    stopTime(&total_timer); printf("Total Execution Time: %f s\n", elapsedTime(total_timer));


    return 0;

    
}
