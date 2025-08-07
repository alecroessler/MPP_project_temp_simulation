#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Set up parameters
const int GRID_SIZE = 256;
const char* POWER_MAP_FILE = "../data/power_map_256.csv";
const double T_amb = 25;  // Ambient temperature in Celcius
const int ITERATIONS = 50000;
const double DIE_WIDTH_M = 0.016; // 16 mm
const double h = DIE_WIDTH_M / GRID_SIZE;  
const double k = 150.0; // thermal conductivity (using silicon)

// load power map q from CSV file
int load_power_map(const char* filename, double q[GRID_SIZE][GRID_SIZE]) {
    // Confirm file opens
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file:\n");
        return 1;
    }

    // Read file data
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            fscanf(file, "%lf,", &q[i][j]);
        }
    }

    fclose(file);
    return 0;
}

// Compute the maximum absolute difference between two temperature grids (element by element)
double max_abs_diff(double a[GRID_SIZE][GRID_SIZE], double b[GRID_SIZE][GRID_SIZE]) {
    double max_diff = 0.0;
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            double diff = fabs(a[i][j] - b[i][j]);
            if (diff > max_diff) {
                max_diff = diff;
            }
        }
    }
    return max_diff;
}

// Compute the maximum, minimum, and average temperatures in a grid
double max_temp(double arr[GRID_SIZE][GRID_SIZE]) {
    double max_val = arr[0][0];
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            if (arr[i][j] > max_val) max_val = arr[i][j];
        }
    }
    return max_val; 
}
double min_temp(double arr[GRID_SIZE][GRID_SIZE]) {
    double min_val = arr[0][0];
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            if (arr[i][j] < min_val) min_val = arr[i][j];
        }
    }
    return min_val;
}
double avg_temp(double arr[GRID_SIZE][GRID_SIZE]) {
    double sum = 0.0;
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            sum += arr[i][j];
        }
    }
    return (sum / (GRID_SIZE * GRID_SIZE));
}





int main() {
    clock_t start_time = clock();

    // Initialize arrays for power map and temperatures
    double q[GRID_SIZE][GRID_SIZE];
    double T[GRID_SIZE][GRID_SIZE];
    double T_new[GRID_SIZE][GRID_SIZE];

    // Load the power map from the CSV file
    if (load_power_map(POWER_MAP_FILE, q) != 0) {
        printf("Failed to load power map.\n");
        return 1;
    }

    // Initialize T with T_amb
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            T[i][j] = T_amb;
            T_new[i][j] = T_amb;
        }
    }

    clock_t setup_time = clock();
    double setup_elapsed = (double)(setup_time - start_time) / CLOCKS_PER_SEC;
    printf("Setup and allocation time: %.2f seconds\n", setup_elapsed);


    // Jacobi discrete heat equation (propogation simulation loop)
    for (int iter = 0; iter < ITERATIONS; iter++) {
        // Update internal grid points
        for (int i = 1; i < GRID_SIZE - 1; i++) {
            for (int j = 1; j < GRID_SIZE - 1; j++) {
                T_new[i][j] = (T[i+1][j] + T[i-1][j] + T[i][j+1] + T[i][j-1] + (h*h / k) * q[i][j]) / 4.0;
            }
        }

        // Apply Dirichlet boundary conditions (fixed ambient temperature of 25 degrees Celsius)
        for (int i = 0; i < GRID_SIZE; i++) {
            T_new[i][0] = T_amb;
            T_new[i][GRID_SIZE - 1] = T_amb;
            T_new[0][i] = T_amb;
            T_new[GRID_SIZE - 1][i] = T_amb;
        }
        
        // Check for convergence and exit
        double max_change = max_abs_diff(T, T_new);
        if (max_change < 1e-3) {
            printf("Converged after %d iterations\n", iter);
            break;
        }

        double max_temperature = max_temp(T_new);
        if (iter % 1000 == 0) {
            printf("Iteration %d: max change = %.5f, max temp = %.5f\n", iter, max_change, max_temperature);
        }


        // Copy T_new to T for next iteration
        for (int i = 0; i < GRID_SIZE; i++) {
            for (int j = 0; j < GRID_SIZE; j++) {
                T[i][j] = T_new[i][j];
            }
        }
    }

    clock_t simulation_time = clock();
    double simulation_elapsed = (double)(simulation_time - setup_time) / CLOCKS_PER_SEC;
    printf("Simulation time: %.2f seconds\n", simulation_elapsed);

    printf("Max Temp: %.2f C\n", max_temp(T_new));
    printf("Min Temp: %.2f C\n", min_temp(T_new));
    printf("Avg Temp: %.2f C\n", avg_temp(T_new));

    clock_t end_time = clock();

    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Total Execution time: %.2f seconds\n", elapsed_time);


    // Save results to csv file
    FILE* file = fopen("/home/roesslera/code/MPP_project_temp_simulation/data/results.csv", "w");
    if (!file) {
        printf("Error opening results file for writing.\n");
        return 1;
    }

    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            fprintf(file, "%.6f", T[i][j]);
            if (j < GRID_SIZE - 1) {
                fprintf(file, ",");
            }
        }
        fprintf(file, "\n");
    }

    fclose(file);

    return 0;
}