#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stddef.h> // Include stddef.h for size_t
#include <math.h>

// Define activation functions    
typedef float (*ActivationFunction)(float);

typedef struct {
    float *weights;
    float bias;
    ActivationFunction activation;
} Layer;

typedef float (*ActivationFunction)(float);

Layer create_layer(size_t input_size, ActivationFunction activation);
float layer_forward(Layer *layer, float *inputs, size_t input_size);
void back_propagation(Layer *layer, float *inputs, size_t input_size, float predicted, float expected, float learning_rate);
float mean_squared_error(float predicted, float expected);
float sigmoid(float x);

#endif /* NEURAL_NETWORK_H */
