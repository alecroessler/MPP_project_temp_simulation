// Kernel algorithm for temperature computation
__global__ void compute_temperature(double* T, double* T_new, double* q, double coeff, 
    int grid_size, double T_amb, double* max_diff_per_block) {

    // 2D indices
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * grid_size + x;
    
    // Shared memory for block reduction
    __shared__ double sdata[256]; // assuming 16x16 block size = 256 threads/block
    
    int local_idx = threadIdx.y * blockDim.x + threadIdx.x;

    double diff = 0.0;

    if (x < grid_size && y < grid_size) {
        if (x == 0 || x == grid_size - 1 || y == 0 || y == grid_size - 1) {
            // Boundary condition
            T_new[idx] = T_amb;
        } else {
            int top    = (y - 1) * grid_size + x;
            int bottom = (y + 1) * grid_size + x;
            int left   = y * grid_size + (x - 1);
            int right  = y * grid_size + (x + 1);
            
            coeff *= q[idx];
            T_new[idx] = (T[top] + T[bottom] + T[left] + T[right] + coeff) / 4.0;
        }
        diff = fabs(T_new[idx] - T[idx]);
    }

    sdata[local_idx] = diff;
    __syncthreads();

    // Parallel reduction in shared memory to find max diff per block
    for (int stride = blockDim.x * blockDim.y / 2; stride > 0; stride /= 2) {
        if (local_idx < stride) {
            sdata[local_idx] = fmax(sdata[local_idx], sdata[local_idx + stride]);
        }
        __syncthreads();
    }

    // Write max diff of this block to global memory
    if (local_idx == 0) {
        int block_id = blockIdx.y * gridDim.x + blockIdx.x;
        max_diff_per_block[block_id] = sdata[0];
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

