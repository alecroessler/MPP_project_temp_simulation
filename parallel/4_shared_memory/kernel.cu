// Kernel algorithm for temperature computation
__global__ void compute_temperature(double* T, double* T_new, double* q, double k, 
    int grid_size, double h, double T_amb) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int shared_x = threadIdx.x + 1; // +1 for halo
    int shared_y = threadIdx.y + 1; 

    __shared__ double s_T[18][19]; // (blockDim.x + 2 x blockDim.y + 2) including halos

    // Boundary check and load center cell
    if (x < grid_size && y < grid_size) {
        s_T[shared_y][shared_x] = T[y * grid_size + x];
    } else {
        s_T[shared_y][shared_x] = T_amb;
    }


    // Load halos: left
    if (threadIdx.x == 0) {
        int halo_x = x - 1;
        int halo_y = y;
        s_T[shared_y][0] = (halo_x >= 0 && halo_y < grid_size) ? T[halo_y * grid_size + halo_x] : T_amb;
    }
    // Right halo
    if (threadIdx.x == 15) {
        int halo_x = x + 1;
        int halo_y = y;
        s_T[shared_y][17] = (halo_x < grid_size && halo_y < grid_size) ? T[halo_y * grid_size + halo_x] : T_amb;
    }
    // Top halo
    if (threadIdx.y == 0) {
        int halo_x = x;
        int halo_y = y - 1;
        s_T[0][shared_x] = (halo_y >= 0 && halo_x < grid_size) ? T[halo_y * grid_size + halo_x] : T_amb;
    }
    // Bottom halo
    if (threadIdx.y == 15) {
        int halo_x = x;
        int halo_y = y + 1;
        s_T[17][shared_x] = (halo_y < grid_size && halo_x < grid_size) ? T[halo_y * grid_size + halo_x] : T_amb;
    }

    __syncthreads();

    if (x >= grid_size || y >= grid_size) return; // bounds check

    int idx = y * grid_size + x;

    // Apply Dirichlet boundary conditions
    if (x == 0 || x == grid_size - 1 || y == 0 || y == grid_size - 1) {
        T_new[idx] = T_amb;
        return;
    }

    // Load values for neighbors
    double top = s_T[shared_y - 1][shared_x];
    double bottom = s_T[shared_y + 1][shared_x];
    double left = s_T[shared_y][shared_x - 1];
    double right = s_T[shared_y][shared_x + 1];

    double coeff = (h * h / k) * q[idx];

    T_new[idx] = (top + bottom + left + right + coeff) / 4.0;
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

