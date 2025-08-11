// Kernel algorithm for temperature computation
#define TILE 16  // hard-coded tile size; launch blocks with (TILE, TILE)

__global__ void compute_temperature(
    const double* __restrict__ T,
    double* __restrict__ T_new,
    const double* __restrict__ q,
    double k,
    int grid_size,
    double h,
    double T_amb)
{
    // block / thread indices
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // global coordinates of this thread's "center" element
    const int x = bx * TILE + tx;
    const int y = by * TILE + ty;
    const int idx = y * grid_size + x;

    // shared tile with 1-cell halo on each side
    __shared__ double sT[TILE + 2][TILE + 2];

    // shared indices (shift by +1 for halo)
    const int sx = tx + 1;
    const int sy = ty + 1;

    // --- Load center element (or ambient if out of grid) ---
    double centerVal = T_amb;
    if (x < grid_size && y < grid_size) {
        centerVal = T[idx];
    }
    sT[sy][sx] = centerVal;

    // --- Load halos: edges and corners (only threads on edges do these loads) ---
    // Left halo (column = sx-1)
    if (tx == 0) {
        int xl = x - 1;
        int yl = y;
        double v = T_amb;
        if (xl >= 0 && yl >= 0 && xl < grid_size && yl < grid_size) {
            v = T[yl * grid_size + xl];
        }
        sT[sy][0] = v;
    }

    // Right halo (column = sx+1)
    if (tx == TILE - 1) {
        int xr = x + 1;
        int yr = y;
        double v = T_amb;
        if (xr >= 0 && yr >= 0 && xr < grid_size && yr < grid_size) {
            v = T[yr * grid_size + xr];
        }
        sT[sy][TILE + 1] = v;
    }

    // Top halo (row = sy-1)
    if (ty == 0) {
        int xt = x;
        int yt = y - 1;
        double v = T_amb;
        if (xt >= 0 && yt >= 0 && xt < grid_size && yt < grid_size) {
            v = T[yt * grid_size + xt];
        }
        sT[0][sx] = v;
    }

    // Bottom halo (row = sy+1)
    if (ty == TILE - 1) {
        int xb = x;
        int yb = y + 1;
        double v = T_amb;
        if (xb >= 0 && yb >= 0 && xb < grid_size && yb < grid_size) {
            v = T[yb * grid_size + xb];
        }
        sT[TILE + 1][sx] = v;
    }

    // Corners
    if (tx == 0 && ty == 0) { // top-left
        int xc = x - 1; int yc = y - 1;
        double v = T_amb;
        if (xc >= 0 && yc >= 0 && xc < grid_size && yc < grid_size) v = T[yc * grid_size + xc];
        sT[0][0] = v;
    }
    if (tx == TILE - 1 && ty == 0) { // top-right
        int xc = x + 1; int yc = y - 1;
        double v = T_amb;
        if (xc >= 0 && yc >= 0 && xc < grid_size && yc < grid_size) v = T[yc * grid_size + xc];
        sT[0][TILE + 1] = v;
    }
    if (tx == 0 && ty == TILE - 1) { // bottom-left
        int xc = x - 1; int yc = y + 1;
        double v = T_amb;
        if (xc >= 0 && yc >= 0 && xc < grid_size && yc < grid_size) v = T[yc * grid_size + xc];
        sT[TILE + 1][0] = v;
    }
    if (tx == TILE - 1 && ty == TILE - 1) { // bottom-right
        int xc = x + 1; int yc = y + 1;
        double v = T_amb;
        if (xc >= 0 && yc >= 0 && xc < grid_size && yc < grid_size) v = T[yc * grid_size + xc];
        sT[TILE + 1][TILE + 1] = v;
    }

    __syncthreads();

    // Out-of-range threads should not compute/stores - guard early
    if (x >= grid_size || y >= grid_size) return;

    // Dirichlet boundaries: keep ambient temperature
    if (x == 0 || x == grid_size - 1 || y == 0 || y == grid_size - 1) {
        T_new[idx] = T_amb;
        return;
    }

    // Read neighbors from shared memory (fast)
    double top    = sT[sy - 1][sx];
    double bottom = sT[sy + 1][sx];
    double left   = sT[sy][sx - 1];
    double right  = sT[sy][sx + 1];

    // Compute coefficient and new temperature
    double coeff = (h * h / k) * q[idx];
    T_new[idx] = (top + bottom + left + right + coeff) * 0.25;
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

