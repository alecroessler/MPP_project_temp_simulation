// compute_temperature_tiled_dynamic.cu
#include <cuda.h>
#include <math.h>

extern "C" __global__
void compute_temperature_shared_tiled_multi(
    const double* __restrict__ T,
    double* __restrict__ T_new,
    const double* __restrict__ q,
    double k,
    int grid_size,
    double h,
    double T_amb)
{
    // Configurable: elements processed per thread horizontally
    constexpr int ELEMS_PER_THREAD_X = 4;

    // Block and thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockDim.x;
    const int by = blockDim.y;

    // Compute global start coords for this tile (without halo)
    // Tile width is blockDim.x * ELEMS_PER_THREAD_X horizontally
    const int tile_w = bx * ELEMS_PER_THREAD_X;
    const int tile_h = by;

    const int gx0 = blockIdx.x * tile_w;
    const int gy0 = blockIdx.y * tile_h;

    // Shared memory tile includes halo (1 cell border)
    // So shared memory size: (tile_w + 2) * (tile_h + 2)
    extern __shared__ double sTile[];

    const int sTile_w = tile_w + 2;
    const int sTile_h = tile_h + 2;

    // Each thread loads multiple elements into shared memory cooperatively
    // Number of elements in shared tile
    const int sTile_size = sTile_w * sTile_h;

    // Flattened thread id inside block
    const int tid = ty * bx + tx;
    const int block_threads = bx * by;

    // Cooperative loading of the shared tile + halo from global memory
    for (int i = tid; i < sTile_size; i += block_threads)
    {
        int sy = i / sTile_w;  // shared mem y
        int sx = i % sTile_w;  // shared mem x

        // Map shared mem coords to global coords (halo offset -1)
        int gx = gx0 + (sx - 1);
        int gy = gy0 + (sy - 1);

        double val = T_amb; // default ambient for out of bounds

        if (gx >= 0 && gx < grid_size && gy >= 0 && gy < grid_size)
        {
            val = T[gy * grid_size + gx];
        }

        sTile[sy * sTile_w + sx] = val;
    }

    __syncthreads();

    // Now each thread computes ELEMS_PER_THREAD_X elements horizontally

    // y coord inside tile for this thread
    int local_y = ty;
    int global_y = gy0 + local_y;

    if (global_y >= grid_size) return; // outside grid vertically

    // Start x for this thread inside tile
    int local_x_start = tx * ELEMS_PER_THREAD_X;

    for (int i = 0; i < ELEMS_PER_THREAD_X; ++i)
    {
        int local_x = local_x_start + i;
        int global_x = gx0 + local_x;

        // Check if inside grid horizontally
        if (global_x >= grid_size) continue;

        // Apply Dirichlet boundary conditions on edges
        if (global_x == 0 || global_x == grid_size - 1 ||
            global_y == 0 || global_y == grid_size - 1)
        {
            T_new[global_y * grid_size + global_x] = T_amb;
            continue;
        }

        // Index inside shared tile (+1 offset for halo)
        int sc = (local_y + 1) * sTile_w + (local_x + 1);

        // Fetch neighbors from shared memory
        double top    = sTile[sc - sTile_w];
        double bottom = sTile[sc + sTile_w];
        double left   = sTile[sc - 1];
        double right  = sTile[sc + 1];

        double coeff = (h * h / k) * q[global_y * grid_size + global_x];

        double newT = (top + bottom + left + right + coeff) * 0.25;

        T_new[global_y * grid_size + global_x] = newT;
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
