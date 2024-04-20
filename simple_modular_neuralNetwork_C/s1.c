#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define TRAIN_COUNT 4

// Define activation functions
typedef float (*ActivationFunction)(float);

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Define a struct for a neuron layer
typedef struct {
    float *weights;
    float bias;
    ActivationFunction activation;
} Layer;

// Create a neuron layer
Layer create_layer(size_t input_size, ActivationFunction activation) {
    Layer layer;
    layer.weights = (float *)malloc(input_size * sizeof(float));
    for (size_t i = 0; i < input_size; ++i) {
        layer.weights[i] = ((float)rand() / (float)RAND_MAX) - 0.5f; // Random initialization
    }
    layer.bias = ((float)rand() / (float)RAND_MAX) - 0.5f; // Random initialization
    layer.activation = activation;
    return layer;
}

// Forward pass through a neuron layer
float layer_forward(Layer *layer, float *inputs, size_t input_size) {
    float output = layer->bias;
    for (size_t i = 0; i < input_size; ++i) {
        output += inputs[i] * layer->weights[i];
    }
    return layer->activation(output);
}

// Calculate the mean squared error
float mean_squared_error(float predicted, float expected) {
    float d = predicted - expected;
    return d * d;
}

// Perform backpropagation and update weights and bias
void back_propagation(Layer *layer, float *inputs, size_t input_size, float predicted, float expected, float learning_rate) {
    float error = predicted - expected;
    float gradient = error * predicted * (1.0f - predicted);

    for (size_t i = 0; i < input_size; ++i) {
        layer->weights[i] -= learning_rate * gradient * inputs[i];
    }
    layer->bias -= learning_rate * gradient;
}

int main() {
    srand(time(0));

    // Create a neural network layer
    Layer hidden_layer = create_layer(2, sigmoid);

    // Define the training data
    float train[][3] = {
        {0, 0, 1},
        {0, 1, 1},
        {1, 0, 1},
        {1, 1, 1},
    };

    float learning_rate = 0.1f;
    float total_error = 0.0f;

    for (size_t epoch = 0; epoch < 100*10000; ++epoch) {
        total_error = 0.0f;
        for (size_t i = 0; i < TRAIN_COUNT; ++i) {
            float *inputs = train[i];
            float expected = inputs[2];

            // Forward pass
            float output = layer_forward(&hidden_layer, inputs, 2);

            // Calculate error
            total_error += mean_squared_error(output, expected);

            // Backpropagation
            back_propagation(&hidden_layer, inputs, 2, output, expected, learning_rate);
        }

        // Print average error every 1000 epochs
        //if (epoch % 1000 == 0) {
          //  printf("Epoch %zu, Average Error: %f\n", epoch, total_error / TRAIN_COUNT);
        //}
    }

    // Print final weights and bias
    printf("Final Weights: [%f, %f], Final Bias: %f\n", hidden_layer.weights[0], hidden_layer.weights[1], hidden_layer.bias);

    // Test the network
    printf("Testing the network:\n");
    for (size_t i = 0; i < TRAIN_COUNT; ++i) {
        float *inputs = train[i];
        float output = layer_forward(&hidden_layer, inputs, 2);
        printf("Input: [%f, %f], Predicted Output: %f\n", inputs[0], inputs[1], output);
    }

    free(hidden_layer.weights); // Free allocated memory
    return 0;
}
