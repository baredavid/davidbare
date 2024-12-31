#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// Function to compute the logistic function
double logistic(double z) {
    return 1.0 / (1.0 + exp(-z));
}

// Function to compute the gradient of the loss with respect to the parameters (beta_0 and beta_1)
void compute_gradients(double *x, double *y, double *beta, int size, double *gradients) {
    double sum_beta_0 = 0.0;
    double sum_beta_1 = 0.0;
    
    for (int i = 0; i < size; i++) {
        double prediction = logistic(beta[0] + beta[1] * x[i]);
        double error = prediction - y[i];
        
        sum_beta_0 += error;
        sum_beta_1 += error * x[i];
    }

    gradients[0] = sum_beta_0 / size;  // Gradient for beta_0
    gradients[1] = sum_beta_1 / size;  // Gradient for beta_1
}

// Function to perform logistic regression using gradient descent
void logistic_regression(double *x, double *y, int size, double *beta, double learning_rate, int iterations) {
    double gradients[2];
    
    for (int i = 0; i < iterations; i++) {
        // Compute the gradients
        compute_gradients(x, y, beta, size, gradients);
        
        // Update the parameters (beta_0 and beta_1) using the gradients
        beta[0] -= learning_rate * gradients[0];  // Update beta_0
        beta[1] -= learning_rate * gradients[1];  // Update beta_1
        
        // Optionally print the parameters after every 100 iterations
        if (i % 100 == 0) {
            printf("Iteration %d: beta_0 = %.4lf, beta_1 = %.4lf\n", i, beta[0], beta[1]);
        }
    }
}

// Function to predict probabilities for new data using the trained model
double predict(double x, double *beta) {
    return logistic(beta[0] + beta[1] * x);
}

// Function to print the results (beta values and predicted probabilities)
void print_results(double *beta, double *x, double *y, int size) {
    printf("Logistic Regression Model: h(x) = %.4lf + %.4lfx\n", beta[0], beta[1]);
    printf("Predicted probabilities:\n");
    for (int i = 0; i < size; i++) {
        double prediction = predict(x[i], beta);
        printf("x = %.2lf, y (actual) = %.2lf, y (predicted) = %.4lf\n", x[i], y[i], prediction);
    }
}

int main() {
    // Sample data for x and y (for demonstration)
    double x[] = {1, 2, 3, 4, 5};
    double y[] = {0, 0, 1, 1, 1};  // Binary outcome for logistic regression
    int size = sizeof(x) / sizeof(x[0]);

    // Initialize beta_0 and beta_1 to zero
    double beta[2] = {0.0, 0.0};  // [beta_0, beta_1]

    // Set the learning rate and number of iterations for gradient descent
    double learning_rate = 0.1;
    int iterations = 1000;

    // Perform logistic regression
    logistic_regression(x, y, size, beta, learning_rate, iterations);

    // Print the final results
    print_results(beta, x, y, size);

    return 0;
}
