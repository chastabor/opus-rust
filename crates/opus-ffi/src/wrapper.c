/*
 * Non-variadic shims for opus_encoder_ctl / opus_decoder_ctl.
 * Rust FFI cannot call C variadic functions, so we provide thin
 * wrappers that forward to the real CTL macros.
 */

#include "opus.h"

/* ── Encoder CTLs ── */

int opus_enc_set_bitrate(OpusEncoder *enc, opus_int32 val) {
    return opus_encoder_ctl(enc, OPUS_SET_BITRATE(val));
}

int opus_enc_set_complexity(OpusEncoder *enc, opus_int32 val) {
    return opus_encoder_ctl(enc, OPUS_SET_COMPLEXITY(val));
}

int opus_enc_set_max_bandwidth(OpusEncoder *enc, opus_int32 val) {
    return opus_encoder_ctl(enc, OPUS_SET_MAX_BANDWIDTH(val));
}

int opus_enc_set_bandwidth(OpusEncoder *enc, opus_int32 val) {
    return opus_encoder_ctl(enc, OPUS_SET_BANDWIDTH(val));
}

int opus_enc_set_vbr(OpusEncoder *enc, opus_int32 val) {
    return opus_encoder_ctl(enc, OPUS_SET_VBR(val));
}

int opus_enc_set_signal(OpusEncoder *enc, opus_int32 val) {
    return opus_encoder_ctl(enc, OPUS_SET_SIGNAL(val));
}

int opus_enc_set_force_channels(OpusEncoder *enc, opus_int32 val) {
    return opus_encoder_ctl(enc, OPUS_SET_FORCE_CHANNELS(val));
}

int opus_enc_set_inband_fec(OpusEncoder *enc, opus_int32 val) {
    return opus_encoder_ctl(enc, OPUS_SET_INBAND_FEC(val));
}

int opus_enc_set_packet_loss_perc(OpusEncoder *enc, opus_int32 val) {
    return opus_encoder_ctl(enc, OPUS_SET_PACKET_LOSS_PERC(val));
}

int opus_enc_get_final_range(OpusEncoder *enc, opus_uint32 *val) {
    return opus_encoder_ctl(enc, OPUS_GET_FINAL_RANGE(val));
}

int opus_enc_reset(OpusEncoder *enc) {
    return opus_encoder_ctl(enc, OPUS_RESET_STATE);
}

/* ── Decoder CTLs ── */

int opus_dec_get_final_range(OpusDecoder *dec, opus_uint32 *val) {
    return opus_decoder_ctl(dec, OPUS_GET_FINAL_RANGE(val));
}

int opus_dec_reset(OpusDecoder *dec) {
    return opus_decoder_ctl(dec, OPUS_RESET_STATE);
}
