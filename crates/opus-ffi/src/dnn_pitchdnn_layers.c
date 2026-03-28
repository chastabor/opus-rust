/* PitchDNN divergence investigation wrappers.
   Exposes tanh/sigmoid batch operations and single-layer dense+tanh
   using the public nnet API for Rust-vs-C layer comparison. */

#include "nnet.h"
#include <string.h>
#include <stdlib.h>


/* Sparse float sgemv8x4: same as internal sparse_sgemv8x4 from vec.h.
   Exposed for direct Rust-vs-C comparison testing. */
void wrap_sparse_sgemv8x4(float *out, const float *w, const int *idx,
                           int rows, const float *x) {
    int i, j;
    memset(out, 0, rows * sizeof(float));
    for (i = 0; i < rows; i += 8) {
        int cols = *idx++;
        for (j = 0; j < cols; j++) {
            int pos = *idx++;
            float xj0 = x[pos + 0];
            float xj1 = x[pos + 1];
            float xj2 = x[pos + 2];
            float xj3 = x[pos + 3];
            float *y = &out[i];
            y[0] += w[0]*xj0;  y[1] += w[1]*xj0;  y[2] += w[2]*xj0;  y[3] += w[3]*xj0;
            y[4] += w[4]*xj0;  y[5] += w[5]*xj0;  y[6] += w[6]*xj0;  y[7] += w[7]*xj0;
            y[0] += w[8]*xj1;  y[1] += w[9]*xj1;  y[2] += w[10]*xj1; y[3] += w[11]*xj1;
            y[4] += w[12]*xj1; y[5] += w[13]*xj1; y[6] += w[14]*xj1; y[7] += w[15]*xj1;
            y[0] += w[16]*xj2; y[1] += w[17]*xj2; y[2] += w[18]*xj2; y[3] += w[19]*xj2;
            y[4] += w[20]*xj2; y[5] += w[21]*xj2; y[6] += w[22]*xj2; y[7] += w[23]*xj2;
            y[0] += w[24]*xj3; y[1] += w[25]*xj3; y[2] += w[26]*xj3; y[3] += w[27]*xj3;
            y[4] += w[28]*xj3; y[5] += w[29]*xj3; y[6] += w[30]*xj3; y[7] += w[31]*xj3;
            w += 32;
        }
    }
}

/* Dense + tanh from a weight blob. Loads specific weight arrays by name. */
void wrap_dense_tanh_from_blob(const void *blob, int blob_len,
                                const char *bias_name, const char *weights_name,
                                float *output, const float *input,
                                int nb_inputs, int nb_outputs) {
    WeightArray *list = NULL;
    int ret = parse_weights(&list, blob, blob_len);
    if (ret < 0 || list == NULL) return;

    LinearLayer layer;
    memset(&layer, 0, sizeof(layer));
    layer.nb_inputs = nb_inputs;
    layer.nb_outputs = nb_outputs;

    int i;
    for (i = 0; list[i].name != NULL; i++) {
        if (strcmp(list[i].name, bias_name) == 0) {
            layer.bias = (const float *)list[i].data;
        }
        if (strcmp(list[i].name, weights_name) == 0) {
            if (list[i].type == 0) /* WEIGHT_TYPE_float */
                layer.float_weights = (const float *)list[i].data;
        }
    }

    compute_generic_dense(&layer, output, input, ACTIVATION_TANH, 0);
    free(list);
}
