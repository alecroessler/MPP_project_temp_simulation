



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


__global__ void compute_diff(double* T, double* T_new, double* diff, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        diff[idx] = fabs(T[idx] - T_new[idx]);
    }
}

__global__ void reduce_max(double* input, double* output, int size) {
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    double max_val = 0.0;

    // Load elements into shared memory (2 elements per thread to optimize)
    if (idx < size)
        max_val = input[idx];
    if (idx + blockDim.x < size) {
        double other = input[idx + blockDim.x];
        if (other > max_val)
            max_val = other;
    }

    sdata[tid] = max_val;
    __syncthreads();

    // Do reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid])
                sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the block's maximum to output array
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}




double max_abs_diff(double* a, double* b, int size) {
    double max_diff = 0.0;
    for (int i = 0; i < size; ++i) {
        double diff = fabs(a[i] - b[i]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}



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

