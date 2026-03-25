/*
 * C shims for CELT internal functions.
 *
 * Many CELT functions are static-inline (not linkable from Rust) or require
 * opaque types (CELTMode*, kiss_fft_state*, mdct_lookup*).  This file wraps
 * them in plain extern-C functions that Rust FFI can call.
 *
 * Compiled WITHOUT FLOAT_APPROX to match the default cmake build of libopus.
 * Note: the Rust celt_exp2 uses the polynomial approximation (FLOAT_APPROX
 * path in C) while celt_log2 uses standard math.  Tests should use
 * appropriate tolerances for celt_exp2.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Required by stack_alloc.h (pulled in via rate.h / cwrs.h) */
#define VAR_ARRAYS

/* CELT internal headers (order matters: modes.h pulls in celt.h, arch.h, etc.) */
#include "modes.h"
#include "mathops.h"
#include "pitch.h"
#include "celt_lpc.h"
#include "bands.h"
#include "vq.h"
#include "rate.h"
#include "kiss_fft.h"
#include "quant_bands.h"
#include "opus_custom.h"

/* ── CELTMode singleton for 48 kHz / 960 ── */

static CELTMode* get_celt_mode_48000(void) {
    static CELTMode *mode = NULL;
    if (!mode) {
        mode = opus_custom_mode_create(48000, 960, NULL);
    }
    return mode;
}

/* ══════════════════════════════════════════════════════════════════════
 * Static-inline math wrappers
 * ══════════════════════════════════════════════════════════════════════ */

float wrap_celt_exp2(float x) {
    return celt_exp2(x);
}

float wrap_celt_log2(float x) {
    return celt_log2(x);
}

float wrap_celt_inner_prod(const float *x, const float *y, int N) {
    return celt_inner_prod_c(x, y, N);
}

float wrap_celt_maxabs16(const float *x, int len) {
    return celt_maxabs16(x, len);
}

float wrap_celt_rcp(float x) {
    return celt_rcp(x);
}

opus_int32 wrap_frac_mul16(opus_int32 a, opus_int32 b) {
    return FRAC_MUL16(a, b);
}

/* ══════════════════════════════════════════════════════════════════════
 * FFT wrapper
 * ══════════════════════════════════════════════════════════════════════ */

void wrap_opus_fft(int nfft,
                   const float *fin_r, const float *fin_i,
                   float *fout_r, float *fout_i) {
    int i;
    kiss_fft_state *st = opus_fft_alloc(nfft, NULL, NULL, 0);
    if (!st) return;

    kiss_fft_cpx *fin  = (kiss_fft_cpx*)malloc((size_t)nfft * sizeof(kiss_fft_cpx));
    kiss_fft_cpx *fout = (kiss_fft_cpx*)malloc((size_t)nfft * sizeof(kiss_fft_cpx));
    if (!fin || !fout) {
        free(fin);
        free(fout);
        opus_fft_free(st, 0);
        return;
    }

    for (i = 0; i < nfft; i++) {
        fin[i].r = fin_r[i];
        fin[i].i = fin_i[i];
    }

    opus_fft_c(st, fin, fout);

    for (i = 0; i < nfft; i++) {
        fout_r[i] = fout[i].r;
        fout_i[i] = fout[i].i;
    }

    free(fin);
    free(fout);
    opus_fft_free(st, 0);
}

/* ══════════════════════════════════════════════════════════════════════
 * MDCT wrappers
 * ══════════════════════════════════════════════════════════════════════ */

void wrap_clt_mdct_forward(float *input, float *output,
                           int N, int overlap, int shift, int stride) {
    mdct_lookup l;
    CELTMode *mode = get_celt_mode_48000();
    clt_mdct_init(&l, N, 3, 0);
    clt_mdct_forward_c(&l, input, output, mode->window, overlap, shift, stride, 0);
    clt_mdct_clear(&l, 0);
}

void wrap_clt_mdct_backward(float *input, float *output,
                            int N, int overlap, int shift, int stride) {
    mdct_lookup l;
    CELTMode *mode = get_celt_mode_48000();
    clt_mdct_init(&l, N, 3, 0);
    clt_mdct_backward_c(&l, input, output, mode->window, overlap, shift, stride, 0);
    clt_mdct_clear(&l, 0);
}

/* ══════════════════════════════════════════════════════════════════════
 * Pitch wrappers
 * ══════════════════════════════════════════════════════════════════════ */

void wrap_pitch_downsample_mono(float *x, float *x_lp, int len) {
    float *channels[1] = { x };
    pitch_downsample(channels, x_lp, len, 1, 2, 0);
}

void wrap_pitch_search(const float *x_lp, float *y,
                       int len, int max_pitch, int *pitch) {
    pitch_search(x_lp, y, len, max_pitch, pitch, 0);
}

float wrap_remove_doubling(float *x, int maxperiod, int minperiod,
                           int N, int *T0, int prev_period,
                           float prev_gain) {
    return remove_doubling(x, maxperiod, minperiod, N, T0,
                           prev_period, prev_gain, 0);
}

void wrap_comb_filter(float *y, float *x, int T0, int T1, int N,
                      float g0, float g1, int tapset0, int tapset1,
                      int overlap) {
    CELTMode *mode = get_celt_mode_48000();
    comb_filter(y, x, T0, T1, N, g0, g1, tapset0, tapset1,
                mode->window, overlap, 0);
}

/* ══════════════════════════════════════════════════════════════════════
 * Band processing wrappers (CELTMode-dependent)
 * ══════════════════════════════════════════════════════════════════════ */

void wrap_compute_band_energies(const float *X, float *bandE,
                                int end, int C, int LM) {
    compute_band_energies(get_celt_mode_48000(), X, bandE, end, C, LM, 0);
}

void wrap_normalise_bands(const float *freq, float *X, const float *bandE,
                          int end, int C, int M) {
    normalise_bands(get_celt_mode_48000(), freq, X, bandE, end, C, M);
}

void wrap_denormalise_bands(const float *X, float *freq,
                            const float *bandLogE,
                            int start, int end, int M,
                            int downsample, int silence) {
    denormalise_bands(get_celt_mode_48000(), X, freq, bandLogE,
                      start, end, M, downsample, silence);
}

/* ══════════════════════════════════════════════════════════════════════
 * Rate allocation wrappers (CELTMode-dependent)
 * ══════════════════════════════════════════════════════════════════════ */

int wrap_bits2pulses(int band, int LM, int bits) {
    return bits2pulses(get_celt_mode_48000(), band, LM, bits);
}

int wrap_pulses2bits(int band, int LM, int pulses) {
    return pulses2bits(get_celt_mode_48000(), band, LM, pulses);
}

void wrap_init_caps(int *cap, int LM, int C) {
    init_caps(get_celt_mode_48000(), cap, LM, C);
}

/* ══════════════════════════════════════════════════════════════════════
 * Energy quantization wrappers (CELTMode + ec_enc/ec_dec)
 * ══════════════════════════════════════════════════════════════════════ */

int wrap_encode_coarse_energy(
    int start, int end,
    const float *eBands, float *oldEBands, float *error,
    unsigned char *ec_buf, int ec_buf_size,
    int C, int LM, int nbAvailableBytes,
    int force_intra, int loss_rate, int lfe)
{
    ec_enc enc;
    ec_enc_init(&enc, ec_buf, (opus_uint32)ec_buf_size);
    CELTMode *mode = get_celt_mode_48000();
    float delayedIntra = 0;
    quant_coarse_energy(mode, start, end, end, eBands, oldEBands,
                        (opus_uint32)(ec_buf_size * 8), error, &enc, C, LM,
                        nbAvailableBytes, force_intra, &delayedIntra,
                        1 /* two_pass */, loss_rate, lfe);
    ec_enc_done(&enc);
    return ec_range_bytes(&enc);
}

void wrap_decode_coarse_energy(
    int start, int end,
    float *oldEBands,
    const unsigned char *ec_buf, int ec_bytes,
    int C, int LM)
{
    ec_dec dec;
    ec_dec_init(&dec, (unsigned char *)ec_buf, (opus_uint32)ec_bytes);
    CELTMode *mode = get_celt_mode_48000();
    int intra = ec_dec_bit_logp(&dec, 3);
    unquant_coarse_energy(mode, start, end, oldEBands, intra, &dec, C, LM);
}

int wrap_encode_fine_energy(
    int start, int end,
    float *oldEBands, float *error,
    const int *fine_quant,
    unsigned char *ec_buf, int ec_buf_size,
    int C)
{
    ec_enc enc;
    ec_enc_init(&enc, ec_buf, (opus_uint32)ec_buf_size);
    CELTMode *mode = get_celt_mode_48000();
    int extra_quant[25] = {0};
    quant_fine_energy(mode, start, end, oldEBands, error,
                      (int *)fine_quant, extra_quant, &enc, C);
    ec_enc_done(&enc);
    return ec_range_bytes(&enc);
}

void wrap_decode_fine_energy(
    int start, int end,
    float *oldEBands,
    const int *fine_quant,
    const unsigned char *ec_buf, int ec_bytes,
    int C)
{
    ec_dec dec;
    ec_dec_init(&dec, (unsigned char *)ec_buf, (opus_uint32)ec_bytes);
    CELTMode *mode = get_celt_mode_48000();
    int extra_quant[25] = {0};
    unquant_fine_energy(mode, start, end, oldEBands,
                        (int *)fine_quant, extra_quant, &dec, C);
}
