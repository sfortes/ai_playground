#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* Define the sigmoid activation function and its derivative */
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

/* Function to initialize weights and biases randomly */
void initialize_weights(double *weights, int size) {
    for (int i = 0; i < size; i++) {
        weights[i] = ((double)rand() / RAND_MAX) - 0.5; // Random values between -0.5 and 0.5
    }
}

void initialize_biases(double *biases, int size) {
    for (int i = 0; i < size; i++) {
        biases[i] = ((double)rand() / RAND_MAX) - 0.5;
    }
}

/* Loss function (Mean Squared Error) */
double calculate_loss(double output, double target) {
    return 0.5 * pow(output - target, 2);
}

void print_learned_results(double inputs[][2], double targets[], double *hidden_weights, double *hidden_biases, double *output_weights, double *output_biases, int num_inputs, int num_hidden, int num_output) {
    printf("Trained Results:\n");
    for (int i = 0; i < 4; i++) {
        double *hidden_layer_input = (double *)malloc(num_hidden * sizeof(double));
        double *hidden_layer_output = (double *)malloc(num_hidden * sizeof(double));
        double *output_layer_input = (double *)malloc(num_output * sizeof(double));
        double *output_layer_output = (double *)malloc(num_output * sizeof(double));

        for (int j = 0; j < num_hidden; j++) {
            hidden_layer_input[j] = 0;
            for (int k = 0; k < num_inputs; k++) {
                hidden_layer_input[j] += inputs[i][k] * hidden_weights[k * num_hidden + j];
            }
            hidden_layer_input[j] += hidden_biases[j];
            hidden_layer_output[j] = sigmoid(hidden_layer_input[j]);
        }

        for (int j = 0; j < num_output; j++) {
            output_layer_input[j] = 0;
            for (int k = 0; k < num_hidden; k++) {
                output_layer_input[j] += hidden_layer_output[k] * output_weights[k * num_output + j];
            }
            output_layer_input[j] += output_biases[j];
            output_layer_output[j] = sigmoid(output_layer_input[j]);
        }
        printf("Input: %d %d, Output: %f, Target: %d\n", (int)inputs[i][0], (int)inputs[i][1], output_layer_output[0], (int)targets[i]);

        free(hidden_layer_input);
        free(hidden_layer_output);
        free(output_layer_input);
        free(output_layer_output);
    }
}

int main() {
    /* XOR dataset */
    double inputs[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double targets[] = {0, 1, 1, 0};

    /* Network architecture: 2 inputs, 2 hidden neurons, 1 output neuron */
    int num_inputs = 2;
    int num_hidden = 2;
    int num_output = 1;

    /* Allocate memory for weights and biases */
    double *hidden_weights = (double *)malloc(num_inputs * num_hidden * sizeof(double));
    double *hidden_biases = (double *)malloc(num_hidden * sizeof(double));
    double *output_weights = (double *)malloc(num_hidden * num_output * sizeof(double));
    double *output_biases = (double *)malloc(num_output * sizeof(double));

    /* Initialize weights and biases randomly */
    srand(time(NULL));
    initialize_weights(hidden_weights, num_inputs * num_hidden);
    initialize_biases(hidden_biases, num_hidden);
    initialize_weights(output_weights, num_hidden * num_output);
    initialize_biases(output_biases, num_output);

    /* Learning rate, number of epochs, and regularization parameter */
    double learning_rate = 0.1;
    int epochs = 10000;
    /* L2 regularization parameter */
    double lambda = 0.01;

    /* Training loop */
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0; // Track the total loss for each epoch
        for (int i = 0; i < 4; i++) {
            /* Forward propagation */
            double *hidden_layer_input = (double *)malloc(num_hidden * sizeof(double));
            double *hidden_layer_output = (double *)malloc(num_hidden * sizeof(double));
            double *output_layer_input = (double *)malloc(num_output * sizeof(double));
            double *output_layer_output = (double *)malloc(num_output * sizeof(double));

            /* Hidden layer calculations */
            for (int j = 0; j < num_hidden; j++) {
                hidden_layer_input[j] = 0;
                for (int k = 0; k < num_inputs; k++) {
                    hidden_layer_input[j] += inputs[i][k] * hidden_weights[k * num_hidden + j];
                }
                hidden_layer_input[j] += hidden_biases[j];
                hidden_layer_output[j] = sigmoid(hidden_layer_input[j]);
            }

            /* Output layer calculations */
            for (int j = 0; j < num_output; j++) {
                output_layer_input[j] = 0;
                for (int k = 0; k < num_hidden; k++) {
                    output_layer_input[j] += hidden_layer_output[k] * output_weights[k * num_output + j];
                }
                output_layer_input[j] += output_biases[j];
                output_layer_output[j] = sigmoid(output_layer_input[j]);
            }

            /* Calculate loss */
            double loss = calculate_loss(output_layer_output[0], targets[i]);
            total_loss += loss;

            /* Backpropagation */
            double output_error = targets[i] - output_layer_output[0];
            double output_delta = output_error * sigmoid_derivative(output_layer_output[0]);

            double *hidden_errors = (double *)malloc(num_hidden * sizeof(double));
            double *hidden_deltas = (double *)malloc(num_hidden * sizeof(double));

            for (int j = 0; j < num_hidden; j++) {
                hidden_errors[j] = output_delta * output_weights[j];
                hidden_deltas[j] = hidden_errors[j] * sigmoid_derivative(hidden_layer_output[j]);
            }

            /* Update weights and biases (Gradient Descent with L2 Regularization) */
            for (int j = 0; j < num_output; j++) {
                for (int k = 0; k < num_hidden; k++) {
                    output_weights[k * num_output + j] += learning_rate * (output_delta * hidden_layer_output[k] - lambda * output_weights[k*num_output + j]);
                }
                output_biases[j] += learning_rate * output_delta;
            }

            for (int j = 0; j < num_hidden; j++) {
                for (int k = 0; k < num_inputs; k++) {
                    hidden_weights[k * num_hidden + j] += learning_rate * (hidden_deltas[j] * inputs[i][k] - lambda * hidden_weights[k*num_hidden + j]);
                }
                hidden_biases[j] += learning_rate * hidden_deltas[j];
            }

            /* Free allocated memory */
            free(hidden_layer_input);
            free(hidden_layer_output);
            free(output_layer_input);
            free(output_layer_output);
            free(hidden_errors);
            free(hidden_deltas);
        }
        if (epoch % 1000 == 0) {
            printf("Epoch %d, Loss: %f\n", epoch, total_loss / 4.0); // Print average loss
        }
    }

    /* Print the learned results */
    print_learned_results(inputs, targets, hidden_weights, hidden_biases, output_weights, output_biases, num_inputs, num_hidden, num_output);


    /* Free allocated memory */
    free(hidden_weights);
    free(hidden_biases);
    free(output_weights);
    free(output_biases);

    return 0;
}
