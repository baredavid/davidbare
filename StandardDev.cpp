#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Function to calculate the mean of an array of numbers
double calculate_mean(double *data, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    return sum / size;
}

// Function to calculate the median of an array of numbers
double calculate_median(double *data, int size) {
    // Sort the array to find the median
    for (int i = 0; i < size - 1; i++) {
        for (int j = i + 1; j < size; j++) {
            if (data[i] > data[j]) {
                double temp = data[i];
                data[i] = data[j];
                data[j] = temp;
            }
        }
    }

    if (size % 2 == 0) {
        // If even, return the average of the two middle elements
        return (data[size / 2 - 1] + data[size / 2]) / 2.0;
    } else {
        // If odd, return the middle element
        return data[size / 2];
    }
}

// Function to calculate the standard deviation of an array of numbers
double calculate_standard_deviation(double *data, int size, double mean) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += pow(data[i] - mean, 2);
    }
    return sqrt(sum / size);
}

// Function to read data from a file into an array
int read_data_from_file(const char *filename, double **data) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file.\n");
        return -1;
    }

    int size = 0;
    double *temp_data = (double *)malloc(sizeof(double) * 100); // Assume max 100 numbers
    while (fscanf(file, "%lf", &temp_data[size]) != EOF) {
        size++;
    }
    fclose(file);

    *data = temp_data;
    return size;
}

// Function to write the results to a file
void write_results_to_file(const char *filename, double mean, double median, double std_dev) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening file to write results.\n");
        return;
    }

    fprintf(file, "Mean: %.2lf\n", mean);
    fprintf(file, "Median: %.2lf\n", median);
    fprintf(file, "Standard Deviation: %.2lf\n", std_dev);

    fclose(file);
}

int main() {
    double *data;
    int size = read_data_from_file("data.txt", &data);
    if (size == -1) {
        return 1;
    }

    // Calculate mean, median, and standard deviation
    double mean = calculate_mean(data, size);
    double median = calculate_median(data, size);
    double std_dev = calculate_standard_deviation(data, size, mean);

    // Output the results
    printf("Mean: %.2lf\n", mean);
    printf("Median: %.2lf\n", median);
    printf("Standard Deviation: %.2lf\n", std_dev);

    // Write the results to a file
    write_results_to_file("results.txt", mean, median, std_dev);

    // Free dynamically allocated memory
    free(data);

    return 0;
}
