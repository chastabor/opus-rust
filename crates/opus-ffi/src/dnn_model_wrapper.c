/* C wrapper functions exposing DNN model-level operations for cross-validation.
   These load actual weights and run full model inference, callable from Rust via FFI.

   Only compiled when DNN weight data files are present (OPUS_DRED/OPUS_OSCE enabled). */

#include "nnet.h"
#include "pitchdnn.h"
#include "fargan.h"
#include "dred_rdovae.h"
#include "dred_rdovae_enc.h"
#include "dred_rdovae_dec.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* ============ PitchDNN ============ */

/* Allocate a PitchDNNState, load model from blob, run compute, return result.
   Returns pitch confidence (float). State is zeroed before each call for reproducibility. */
float wrap_pitchdnn_compute(const void *blob, int blob_len,
                            const float *if_features, const float *xcorr_features) {
    PitchDNNState st;
    pitchdnn_init(&st);
    if (pitchdnn_load_model(&st, blob, blob_len) != 0) {
        return -1.0f;
    }
    return compute_pitchdnn(&st, if_features, xcorr_features, 0);
}

/* Multi-step PitchDNN: init once, call compute N times, return last result.
   This tests GRU state accumulation across frames. */
float wrap_pitchdnn_multi_step(const void *blob, int blob_len,
                               const float *if_features_seq,
                               const float *xcorr_features_seq,
                               int nb_if_features, int nb_xcorr_features,
                               int n_steps) {
    PitchDNNState st;
    pitchdnn_init(&st);
    if (pitchdnn_load_model(&st, blob, blob_len) != 0) {
        return -1.0f;
    }
    float result = 0.0f;
    for (int i = 0; i < n_steps; i++) {
        result = compute_pitchdnn(&st,
                                  if_features_seq + i * nb_if_features,
                                  xcorr_features_seq + i * nb_xcorr_features,
                                  0);
    }
    return result;
}

/* ============ RDOVAE Encoder ============ */

/* Encode one frame through the RDOVAE encoder.
   Outputs latents and initial_state. */
int wrap_rdovae_encode_dframe(const void *blob, int blob_len,
                              float *latents, float *initial_state,
                              const float *input) {
    WeightArray *arrays = NULL;
    int ret = parse_weights(&arrays, blob, blob_len);
    if (ret < 0 || arrays == NULL) return -1;

    RDOVAEEnc model;
    if (init_rdovaeenc(&model, arrays) != 0) {
        free(arrays);
        return -2;
    }

    RDOVAEEncState enc_state;
    memset(&enc_state, 0, sizeof(enc_state));

    dred_rdovae_encode_dframe(&enc_state, &model, latents, initial_state, input, 0);

    free(arrays);
    return 0;
}

/* ============ RDOVAE Decoder ============ */

/* Decode latents through the RDOVAE decoder.
   initial_state: state vector from encoder
   latents: encoded latent vectors
   nb_latents: how many latent frames to decode
   output: reconstructed features (nb_latents * output_dim floats) */
int wrap_rdovae_decode_all(const void *blob, int blob_len,
                           float *output,
                           const float *initial_state,
                           const float *latents,
                           int nb_latents) {
    WeightArray *arrays = NULL;
    int ret = parse_weights(&arrays, blob, blob_len);
    if (ret < 0 || arrays == NULL) return -1;

    RDOVAEDec model;
    if (init_rdovaedec(&model, arrays) != 0) {
        free(arrays);
        return -2;
    }

    DRED_rdovae_decode_all(&model, output, initial_state, latents, nb_latents, 0);

    free(arrays);
    return 0;
}

/* ============ FARGAN ============ */

/* Synthesize one frame of PCM from features using FARGAN.
   cont_pcm: FARGAN_CONT_SAMPLES (320) floats of continuity audio
   cont_features: 5 * NB_FEATURES (100) floats of continuity features
   features: NB_FEATURES (20) floats for the synthesis frame
   pcm_out: FARGAN_FRAME_SIZE (160) floats of output audio. */
int wrap_fargan_synthesize(const void *blob, int blob_len,
                           float *pcm_out,
                           const float *cont_pcm, const float *cont_features,
                           const float *features) {
    FARGANState *st = (FARGANState *)calloc(1, sizeof(FARGANState));
    if (!st) return -3;
    int ret = fargan_load_model(st, blob, blob_len);
    if (ret != 0) { free(st); return -1; }
    fargan_cont(st, cont_pcm, cont_features);
    fargan_synthesize(st, pcm_out, features);
    free(st);
    return 0;
}

/* Multi-frame FARGAN synthesis with continuity init. */
int wrap_fargan_synthesize_multi(const void *blob, int blob_len,
                                 float *pcm_out,
                                 const float *cont_pcm, const float *cont_features,
                                 const float *features_seq,
                                 int nb_features_per_frame,
                                 int n_frames) {
    FARGANState *st = (FARGANState *)calloc(1, sizeof(FARGANState));
    if (!st) return -3;
    int ret = fargan_load_model(st, blob, blob_len);
    if (ret != 0) { free(st); return -1; }
    fargan_cont(st, cont_pcm, cont_features);
    for (int i = 0; i < n_frames; i++) {
        fargan_synthesize(st, pcm_out + i * FARGAN_FRAME_SIZE,
                         features_seq + i * nb_features_per_frame);
    }
    free(st);
    return 0;
}
