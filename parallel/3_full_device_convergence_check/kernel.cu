// Kernel algorithm for temperature computation
__global__ void compute_temperature(double* T, double* T_new, double* q, double k, 
    int grid_size, double h, double T_amb) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= grid_size || y >= grid_size) return;

    int idx = y * grid_size + x;

    // Apply Dirichlet boundary conditions
    if (x == 0 || x == grid_size - 1 || y == 0 || y == grid_size - 1) {
        T_new[idx] = T_amb;
        return;
    }

    // Compute 1D indices for neighbors
    int top    = (y - 1) * grid_size + x;
    int bottom = (y + 1) * grid_size + x;
    int left   = y * grid_size + (x - 1);
    int right  = y * grid_size + (x + 1);

    double coeff = (h * h / k) * q[idx];

    T_new[idx] = (T[top] + T[bottom] + T[left] + T[right] + coeff) / 4.0;
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


// 
__global__ void max_diff_final_reduction(double* max_diff, double* final_max_diff, int num_blocks) {
    __shared__ double shared_max[256];

    // Load block maximums into shared memory
    if (threadIdx.x < num_blocks) {
        shared_max[threadIdx.x] = max_diff[threadIdx.x];
    } else {
        shared_max[threadIdx.x] = 0.0; 
    }
    __syncthreads();

    // Find global maximum
    for (int stride = 128; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared_max[threadIdx.x] = fmax(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    // Write the final result to global memory
    if (threadIdx.x == 0) {
        *final_max_diff = shared_max[0];
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

