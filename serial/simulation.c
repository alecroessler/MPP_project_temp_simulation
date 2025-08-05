#include <stdio.h>
#include <stdlib.h>



const int GRID_SIZE = 128;
const char* POWER_MAP_FILE = "../data/power_map_128.csv";



int load_power_map(const char* filename, double q[GRID_SIZE][GRID_SIZE]) {
    // Confirm file opens
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file:\n");
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



int main() {
    double q[GRID_SIZE][GRID_SIZE];

    if (load_power_map(POWER_MAP_FILE, q) != 0) {
        fprintf(stderr, "Failed to load power map.\n");
        return 1;
    }

    printf("Top-left corner of power map:\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%.3f ", q[i][j]);
        }
        printf("\n");
    }



    return 0;
}