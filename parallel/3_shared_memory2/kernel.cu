// compute_temperature_tiled_dynamic.cu
#include <cuda.h>
#include <math.h>

extern "C" __global__
void compute_temperature(
    double* T,        // old temperature array (global)
    double* T_new,    // new temperature array (global)
    double* q,        // power map (global)
    double  k,
    int     grid_size,
    double  h,
    double  T_amb,
    double* max_diff) // per-block output max diff (global), length = gridDim.x * gridDim.y
{
    // dynamic shared memory provided by caller
    extern __shared__ double smem[];  

    const int bx = blockDim.x;
    const int by = blockDim.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int local_idx = ty * bx + tx;        // 0 .. (bx*by-1)
    const int blockId = blockIdx.y * gridDim.x + blockIdx.x;

    // compute tile sizes that include 1-cell halo
    const int tile_w = bx + 2;
    const int tile_h = by + 2;
    const int tile_size = tile_w * tile_h;    // number of doubles used for tile
    const int block_elems = bx * by;          // number of doubles for reduction region

    // Partition dynamic shared memory
    double* sTile = smem;                     // [0 .. tile_size-1]
    double* sRed  = smem + tile_size;         // [tile_size .. tile_size + block_elems - 1]

    // Global coords for thread's center element
    const int gx = blockIdx.x * bx + tx;
    const int gy = blockIdx.y * by + ty;

    // flatten global idx (valid only when inside grid)
    const int gidx = gy * grid_size + gx;

    // local center coords in tile
    const int lx = tx + 1;
    const int ly = ty + 1;
    const int sCenter = ly * tile_w + lx;

    // 1) Load center
    if (gx < grid_size && gy < grid_size) {
        sTile[sCenter] = T[gidx];
    } else {
        sTile[sCenter] = T_amb;
    }

    // 2) Load halos using conditional threads to reduce duplicate loads
    // Left halo
    if (tx == 0) {
        int gxl = gx - 1;
        int gyl = gy;
        int sIdx = ly * tile_w + (lx - 1);
        if (gxl >= 0 && gyl >= 0 && gxl < grid_size && gyl < grid_size)
            sTile[sIdx] = T[gyl * grid_size + gxl];
        else
            sTile[sIdx] = T_amb;
    }
    // Right halo
    if (tx == bx - 1) {
        int gxr = gx + 1;
        int gyr = gy;
        int sIdx = ly * tile_w + (lx + 1);
        if (gxr >= 0 && gyr >= 0 && gxr < grid_size && gyr < grid_size)
            sTile[sIdx] = T[gyr * grid_size + gxr];
        else
            sTile[sIdx] = T_amb;
    }
    // Top halo
    if (ty == 0) {
        int gxt = gx;
        int gyt = gy - 1;
        int sIdx = (ly - 1) * tile_w + lx;
        if (gxt >= 0 && gyt >= 0 && gxt < grid_size && gyt < grid_size)
            sTile[sIdx] = T[gyt * grid_size + gxt];
        else
            sTile[sIdx] = T_amb;
    }
    // Bottom halo
    if (ty == by - 1) {
        int gxb = gx;
        int gyb = gy + 1;
        int sIdx = (ly + 1) * tile_w + lx;
        if (gxb >= 0 && gyb >= 0 && gxb < grid_size && gyb < grid_size)
            sTile[sIdx] = T[gyb * grid_size + gxb];
        else
            sTile[sIdx] = T_amb;
    }

    // Corners (a small number of threads handle these)
    if (tx == 0 && ty == 0) {
        int gxll = gx - 1, gytt = gy - 1;
        int sIdx = (ly - 1) * tile_w + (lx - 1);
        if (gxll >= 0 && gytt >= 0 && gxll < grid_size && gytt < grid_size)
            sTile[sIdx] = T[gytt * grid_size + gxll];
        else
            sTile[sIdx] = T_amb;
    }
    if (tx == bx - 1 && ty == 0) {
        int gxrr = gx + 1, gytt = gy - 1;
        int sIdx = (ly - 1) * tile_w + (lx + 1);
        if (gxrr >= 0 && gytt >= 0 && gxrr < grid_size && gytt < grid_size)
            sTile[sIdx] = T[gytt * grid_size + gxrr];
        else
            sTile[sIdx] = T_amb;
    }
    if (tx == 0 && ty == by - 1) {
        int gxll = gx - 1, gybb = gy + 1;
        int sIdx = (ly + 1) * tile_w + (lx - 1);
        if (gxll >= 0 && gybb >= 0 && gxll < grid_size && gybb < grid_size)
            sTile[sIdx] = T[gybb * grid_size + gxll];
        else
            sTile[sIdx] = T_amb;
    }
    if (tx == bx - 1 && ty == by - 1) {
        int gxrr = gx + 1, gybb = gy + 1;
        int sIdx = (ly + 1) * tile_w + (lx + 1);
        if (gxrr >= 0 && gybb >= 0 && gxrr < grid_size && gybb < grid_size)
            sTile[sIdx] = T[gybb * grid_size + gxrr];
        else
            sTile[sIdx] = T_amb;
    }

    // synchronize after loading tile+halo
    __syncthreads();

    // 3) Compute new temperature using shared memory
    double diff = 0.0;
    if (gx < grid_size && gy < grid_size) {
        // Dirichlet boundary
        if (gx == 0 || gx == grid_size - 1 || gy == 0 || gy == grid_size - 1) {
            T_new[gidx] = T_amb;
            diff = fabs(T_amb - sTile[sCenter]);
        } else {
            double top    = sTile[(ly - 1) * tile_w + lx];
            double bottom = sTile[(ly + 1) * tile_w + lx];
            double left   = sTile[ly * tile_w + (lx - 1)];
            double right  = sTile[ly * tile_w + (lx + 1)];
            double coeff  = (h * h / k) * q[gidx];

            double newT = (top + bottom + left + right + coeff) * 0.25;
            T_new[gidx] = newT;
            diff = fabs(newT - sTile[sCenter]);
        }
    } else {
        diff = 0.0;
    }

    // 4) Block-level reduction into sRed (dynamic region)
    sRed[local_idx] = diff;
    __syncthreads();

    int stride = block_elems / 2;
    while (stride > 0) {
        if (local_idx < stride) {
            double a = sRed[local_idx];
            double b = sRed[local_idx + stride];
            sRed[local_idx] = (a > b) ? a : b;
        }
        __syncthreads();
        stride >>= 1;
    }

    if (local_idx == 0) {
        max_diff[blockId] = sRed[0];
    }
}

// Kernel for reduction to find maximum difference
__global__ void max_diff_reduction(double* T, double* T_new, double* max_diff, int total_size) {
    __shared__ double data[512];
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
    for (int stride = 256; stride > 0; stride /= 2) {
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
