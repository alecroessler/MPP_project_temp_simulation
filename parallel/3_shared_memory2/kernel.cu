// compute_temperature_tiled_dynamic.cu
#include <cuda.h>
#include <math.h>

// compute_temperature_tiled_simple.cu
extern "C" __global__
void compute_temperature(double* T, double* T_new, double* q, double k, 
    int grid_size, double h, double T_amb) 
{
    // block/thread ids
    const int bx = blockDim.x;
    const int by = blockDim.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * bx + tx;
    const int threads_per_block = bx * by;

    // top-left global coordinates of the block interior (center region, without halo)
    const int gx0 = blockIdx.x * bx;
    const int gy0 = blockIdx.y * by;

    // tile dimensions including 1-cell halo
    const int tile_w = bx + 2;
    const int tile_h = by + 2;
    const int tile_elems = tile_w * tile_h;

    // dynamic shared memory: tile only
    extern __shared__ double sTile[]; // size in bytes: sizeof(double) * tile_elems

    // cooperative load: each thread loads multiple elements of the shared tile
    for (int s = tid; s < tile_elems; s += threads_per_block) {
        int ly = s / tile_w;   // 0 .. tile_h-1
        int lx = s % tile_w;   // 0 .. tile_w-1

        // map tile local coords to global coords:
        // tile local (1,1) corresponds to global (gx0, gy0)
        int gx = gx0 + (lx - 1);
        int gy = gy0 + (ly - 1);

        double v;
        if (gx >= 0 && gx < grid_size && gy >= 0 && gy < grid_size) {
            v = T[gy * grid_size + gx];
        } else {
            // outside domain: use Dirichlet ambient
            v = T_amb;
        }
        sTile[s] = v;
    }

    __syncthreads(); // tile fully loaded

    // compute this thread's global coordinate for the interior cell it owns
    int gx = gx0 + tx;
    int gy = gy0 + ty;

    if (gx >= grid_size || gy >= grid_size) {
        // outside domain (thread mapped outside grid) -> nothing to do
        return;
    }

    int gidx = gy * grid_size + gx;

    // boundary cells: Dirichlet
    if (gx == 0 || gx == grid_size - 1 || gy == 0 || gy == grid_size - 1) {
        T_new[gidx] = T_amb;
        return;
    }

    // index in shared tile for center
    int sx = tx + 1; // 1..bx
    int sy = ty + 1; // 1..by
    int sc = sy * tile_w + sx;

    // fetch neighbors from shared memory (no global loads)
    double top    = sTile[sc - tile_w];
    double bottom = sTile[sc + tile_w];
    double left   = sTile[sc - 1];
    double right  = sTile[sc + 1];
    double coeff = (h * h / k) * q[idx];

    double newT = (top + bottom + left + right + coeff) * 0.25;
    T_new[gidx] = newT;
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
