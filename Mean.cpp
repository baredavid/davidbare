#include <stdio.h>
#include <stdlib.h>

// Function to calculate the mean of an array of numbers
double calculate_mean(double *data, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    return sum / size;
}

// Function to calculate the slope (m) and intercept (b) of the line y = mx + b using the least squares method
void linear_regression(double *x, double *y, int size, double *m, double *b) {
    double x_mean = calculate_mean(x, size);
    double y_mean = calculate_mean(y, size);
    
    double numerator = 0.0;
    double denominator = 0.0;
    
    for (int i = 0; i < size; i++) {
        numerator += (x[i] - x_mean) * (y[i] - y_mean);
        denominator += (x[i] - x_mean) * (x[i] - x_mean);
    }
    
    *m = numerator / denominator;  // Slope
    *b = y_mean - (*m * x_mean);  // Intercept
}

// Function to predict y values based on the regression model
void predict(double *x, int size, double m, double b, double *predictions) {
    for (int i = 0; i < size; i++) {
        predictions[i] = m * x[i] + b;
    }
}

// Function to calculate the sum of squared residuals (SSR) for goodness of fit
double calculate_ssr(double *y, double *predictions, int size) {
    double ssr = 0.0;
    for (int i = 0; i < size; i++) {
        ssr += (y[i] - predictions[i]) * (y[i] - predictions[i]);
    }
    return ssr;
}

// Function to read data from a file (x and y values)
int read_data_from_file(const char *filename, double **x, double **y) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file.\n");
        return -1;
    }

    int size = 0;
    double *temp_x = (double *)malloc(sizeof(double) * 100);  // Assuming max 100 data points
    double *temp_y = (double *)malloc(sizeof(double) * 100);
    
    while (fscanf(file, "%lf,%lf", &temp_x[size], &temp_y[size]) != EOF) {
        size++;
    }
    fclose(file);
    
    *x = temp_x;
    *y = temp_y;
    
    return size;
}

// Function to write the results (slope, intercept, and SSR) to a file
void write_results_to_file(const char *filename, double m, double b, double ssr) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening file to write results.\n");
        return;
    }

    fprintf(file, "Slope (m): %.4lf\n", m);
    fprintf(file, "Intercept (b): %.4lf\n", b);
    fprintf(file, "Sum of Squared Residuals (SSR): %.4lf\n", ssr);

    fclose(file);
}

int main() {
    double *x, *y;
    int size = read_data_from_file("data.csv", &x, &y);
    if (size == -1) {
        return 1;
    }

    // Calculate the slope (m) and intercept (b)
    double m, b;
    linear_regression(x, y, size, &m, &b);

    // Predict the y values based on the regression model
    double *predictions = (double *)malloc(sizeof(double) * size);
    predict(x, size, m, b, predictions);

    // Calculate the Sum of Squared Residuals (SSR) for goodness of fit
    double ssr = calculate_ssr(y, predictions, size);

    // Output the results
    printf("Slope (m): %.4lf\n", m);
    printf("Intercept (b): %.4lf\n", b);
    printf("Sum of Squared Residuals (SSR): %.4lf\n", ssr);

    // Write the results to a file
    write_results_to_file("results.csv", m, b, ssr);

    // Free dynamically allocated memory
    free(x);
    free(y);
    free(predictions);

    return 0;
}
