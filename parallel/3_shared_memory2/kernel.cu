// compute_temperature_tiled_dynamic.cu
#include <cuda.h>
#include <math.h>

#define ELEMS_PER_THREAD_X 4  // tune this (2, 4, maybe 8)

__global__ void compute_temperature_multiX(
    const double* T,
    double* T_new,
    const double* q,
    double k,
    int grid_size,
    double h,
    double T_amb
) {
    // thread's starting coordinates
    int x_start = (blockIdx.x * blockDim.x + threadIdx.x) * ELEMS_PER_THREAD_X;
    int y       = blockIdx.y * blockDim.y + threadIdx.y;

    // precompute constant coeff
    double hh_over_k = (h * h) / k;

    // loop over horizontal strip
    for (int i = 0; i < ELEMS_PER_THREAD_X; ++i) {
        int x = x_start + i;

        if (x >= grid_size || y >= grid_size) return;  // outside domain

        int idx = y * grid_size + x;

        // Apply Dirichlet boundary conditions
        if (x == 0 || x == grid_size - 1 || y == 0 || y == grid_size - 1) {
            T_new[idx] = T_amb;
            continue;
        }

        // Compute 1D indices for neighbors
        int top    = idx - grid_size;
        int bottom = idx + grid_size;
        int left   = idx - 1;
        int right  = idx + 1;

        double coeff = hh_over_k * q[idx];

        // stencil update
        T_new[idx] = (T[top] + T[bottom] + T[left] + T[right] + coeff) * 0.25;
    }
}


// Kernel for reduction to find maximum difference
__global__ void max_diff_reduction(double* T, double* T_new, double* max_diff, int total_size) {
    __shared__ double data[256];
    int local_index = threadIdx.y * blockDim.x + threadIdx.x;
    int global_index = blockIdx.x * blockDim.x * blockDim.y + local_index;


    // Compute difference for each thread
    double difference = 0.0;
    if (global_index < total_size) {
        difference = fabs(T_new[global_index] - T[global_index]);
    }

    data[local_index] = difference;
    __syncthreads();

    // Max reduction
    for (int stride = 128; stride > 0; stride /= 2) {
        if (local_index  < stride) {
            data[local_index] = fmax(data[local_index], data[local_index + stride]);
        }
        __syncthreads();
    }

    // Return the maximum difference at index 0
    if (local_index  == 0) {
        max_diff[blockIdx.x] = data[0];
    }
}


// Compute the maximum, minimum, and average temperature in the grid
double max_temp(double* arr, int grid_size) {
    double max_val = arr[0];
    for (int i = 0; i < grid_size * grid_size; i++) {
        if (arr[i] > max_val) max_val = arr[i];
    }
    return max_val;  
}
double min_temp(double* arr, int grid_size) {
    double min_val = arr[0];
    for (int i = 0; i < grid_size * grid_size; i++) {
        if (arr[i] < min_val) min_val = arr[i];
    }
    return min_val;
}
double avg_temp(double* arr, int grid_size) {
    double sum = 0.0;
    for (int i = 0; i < grid_size * grid_size; i++) {
        sum += arr[i];
    }
    return (sum / (grid_size * grid_size));
}
