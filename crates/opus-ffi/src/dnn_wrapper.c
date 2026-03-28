/* C wrapper functions exposing DNN internals for cross-validation testing.
   These are non-variadic functions callable from Rust via FFI. */

#include "nnet.h"
#include "nnet_arch.h"
#include <string.h>
#include <stdlib.h>

/* ============ Layer 0: Activations ============ */

void wrap_compute_activation(float *output, const float *input, int N, int activation) {
    compute_activation(output, input, N, activation, 0);
}

/* ============ Layer 1: Linear ============ */

/* Create a simple float-weight LinearLayer for testing.
   Caller provides weights (column-major, rows*cols floats) and bias (rows floats).
   Returns a heap-allocated LinearLayer. Caller must free via wrap_free_linear_layer. */
void wrap_compute_linear(float *out, const float *weights, const float *bias,
                         int nb_inputs, int nb_outputs, const float *input) {
    LinearLayer layer;
    memset(&layer, 0, sizeof(layer));
    layer.float_weights = weights;
    layer.bias = bias;
    layer.nb_inputs = nb_inputs;
    layer.nb_outputs = nb_outputs;
    compute_linear(&layer, out, input, 0);
}

/* Quantized int8 linear layer (cgemv8x4 path). */
void wrap_compute_linear_int8(float *out, const opus_int8 *weights, const float *bias,
                              const float *scale, int nb_inputs, int nb_outputs,
                              const float *input) {
    LinearLayer layer;
    memset(&layer, 0, sizeof(layer));
    layer.weights = weights;
    layer.bias = bias;
    layer.scale = scale;
    layer.nb_inputs = nb_inputs;
    layer.nb_outputs = nb_outputs;
    compute_linear(&layer, out, input, 0);
}

/* ============ Layer 2: Composite Ops ============ */

void wrap_compute_generic_dense(float *output, const float *input,
                                const float *weights, const float *bias,
                                int nb_inputs, int nb_outputs, int activation) {
    LinearLayer layer;
    memset(&layer, 0, sizeof(layer));
    layer.float_weights = weights;
    layer.bias = bias;
    layer.nb_inputs = nb_inputs;
    layer.nb_outputs = nb_outputs;
    compute_generic_dense(&layer, output, input, activation, 0);
}

void wrap_compute_generic_gru(float *state,
                              const float *input_weights, const float *input_bias,
                              const float *recurrent_weights, const float *recurrent_bias,
                              const float *recurrent_diag,
                              int nb_inputs, int nb_neurons,
                              const float *input) {
    LinearLayer input_layer, recurrent_layer;
    memset(&input_layer, 0, sizeof(input_layer));
    memset(&recurrent_layer, 0, sizeof(recurrent_layer));

    input_layer.float_weights = input_weights;
    input_layer.bias = input_bias;
    input_layer.nb_inputs = nb_inputs;
    input_layer.nb_outputs = 3 * nb_neurons;

    recurrent_layer.float_weights = recurrent_weights;
    recurrent_layer.bias = recurrent_bias;
    recurrent_layer.diag = recurrent_diag;
    recurrent_layer.nb_inputs = nb_neurons;
    recurrent_layer.nb_outputs = 3 * nb_neurons;

    compute_generic_gru(&input_layer, &recurrent_layer, state, input, 0);
}
