#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define MAX_POINTS 100  // Maximum number of data points

// Structure for a data point
typedef struct {
    double x;    // x-coordinate (feature)
    double y;    // y-coordinate (feature)
    int label;   // Label (target value)
} DataPoint;

// Function to calculate the Euclidean distance between two data points
double euclidean_distance(DataPoint p1, DataPoint p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

// Function to compare two data points based on their distance (for sorting)
int compare(const void *a, const void *b) {
    DataPoint *point1 = (DataPoint *)a;
    DataPoint *point2 = (DataPoint *)b;
    
    // Distance is compared here (assumed precomputed)
    return point1->label - point2->label;
}

// Function to predict the label using KNN
int knn(DataPoint *train_data, int train_size, DataPoint test_point, int k) {
    // Calculate the distance from test_point to each training data point
    double distances[MAX_POINTS];
    for (int i = 0; i < train_size; i++) {
        distances[i] = euclidean_distance(train_data[i], test_point);
    }
    
    // Sort training data by distance (using a basic selection of the nearest k points)
    for (int i = 0; i < train_size - 1; i++) {
        for (int j = 0; j < train_size - i - 1; j++) {
            if (distances[j] > distances[j + 1]) {
                double temp = distances[j];
                distances[j] = distances[j + 1];
                distances[j + 1] = temp;
                
                // Swap labels
                int temp_label = train_data[j].label;
                train_data[j].label = train_data[j + 1].label;
                train_data[j + 1].label = temp_label;
            }
        }
    }
    
    // Find the majority label among the k nearest neighbors
    int label_count[2] = {0, 0}; // Assuming binary classification (0 or 1)
    for (int i = 0; i < k; i++) {
        label_count[train_data[i].label]++;
    }
    
    // Return the majority label
    if (label_count[0] > label_count[1]) {
        return 0;  // Class 0
    } else {
        return 1;  // Class 1
    }
}

// Main function to run the program
int main() {
    // Example training data (x, y, label)
    DataPoint train_data[] = {
        {1.0, 2.0, 0},
        {2.0, 3.0, 0},
        {3.0, 3.0, 1},
        {6.0, 6.0, 1},
        {7.0, 8.0, 1}
    };
    
    int train_size = sizeof(train_data) / sizeof(train_data[0]);
    
    // Test point
    DataPoint test_point = {5.0, 5.0, -1};  // Label is unknown
    
    // Set the value of k (number of neighbors)
    int k = 3;
    
    // Predict the label of the test point using KNN
    int predicted_label = knn(train_data, train_size, test_point, k);
    
    // Output the prediction
    printf("Predicted label for test point (%.2f, %.2f): %d\n", test_point.x, test_point.y, predicted_label);
    
    return 0;
}
