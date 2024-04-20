#include "neural_network.h"
#include <stdlib.h>

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

float layer_forward(Layer *layer, float *inputs, size_t input_size) {
    float output = layer->bias;
    for (size_t i = 0; i < input_size; ++i) {
        output += inputs[i] * layer->weights[i];
    }
    return layer->activation(output);
}

void back_propagation(Layer *layer, float *inputs, size_t input_size, float predicted, float expected, float learning_rate) {
    float error = predicted - expected;
    float gradient = error * predicted * (1.0f - predicted);

    for (size_t i = 0; i < input_size; ++i) {
        layer->weights[i] -= learning_rate * gradient * inputs[i];
    }
    layer->bias -= learning_rate * gradient;
}

float mean_squared_error(float predicted, float expected) {
    float d = predicted - expected;
    return d * d;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}