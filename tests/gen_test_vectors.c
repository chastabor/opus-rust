/*
 * Generate test vectors for Opus decoder cross-validation.
 * Creates sequences of CELT-only, SILK-only, and hybrid packets
 * in both mono and stereo configurations.
 * For each test case: encodes N_WARMUP+1 frames, saves ALL packets and
 * the reference decoded PCM for each frame.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "opus.h"

#define SAMPLE_RATE 48000
#define FRAME_MS 20
#define FRAME_SIZE (SAMPLE_RATE * FRAME_MS / 1000)  /* 960 */
#define MAX_PACKET 4000
#define N_WARMUP 5
#define PI_F 3.14159265358979f

static void write_file(const char *path, const void *data, size_t len) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    fwrite(data, 1, len, f);
    fclose(f);
}

/* Generate mono sine wave */
static void gen_sine(float *buf, int samples, int offset, float freq, float amp) {
    for (int i = 0; i < samples; i++) {
        buf[i] = amp * sinf(2.0f * PI_F * freq * (i + offset) / SAMPLE_RATE);
    }
}

/* Generate stereo sine waves with different frequencies per channel */
static void gen_stereo_sine(float *buf, int samples, int offset,
                            float freq_l, float freq_r, float amp) {
    for (int i = 0; i < samples; i++) {
        float t = (float)(i + offset) / SAMPLE_RATE;
        buf[i * 2]     = amp * sinf(2.0f * PI_F * freq_l * t);  /* left */
        buf[i * 2 + 1] = amp * sinf(2.0f * PI_F * freq_r * t);  /* right */
    }
}

static void gen_silence(float *buf, int samples_x_channels) {
    memset(buf, 0, samples_x_channels * sizeof(float));
}

/*
 * Encode N_WARMUP+1 frames of continuous audio. Save every packet
 * and every decoded PCM frame so the Rust side can replay the full
 * sequence with identical decoder state.
 *
 * channels: 1 for mono, 2 for stereo
 * freq / freq_r: for stereo, freq is left channel, freq_r is right channel
 *                for mono, only freq is used
 * force_channels: if > 0, force encoder to use this many channels
 */
static int run_test_case(
    const char *name, const char *outdir,
    int channels, int application, int max_bandwidth, int bitrate,
    float freq, float freq_r, float amp,
    int force_channels
) {
    int err;
    OpusEncoder *enc = opus_encoder_create(SAMPLE_RATE, channels, application, &err);
    if (err != OPUS_OK) {
        fprintf(stderr, "Failed to create encoder for %s: %s\n", name, opus_strerror(err));
        return -1;
    }
    OpusDecoder *dec = opus_decoder_create(SAMPLE_RATE, channels, &err);
    if (err != OPUS_OK) {
        fprintf(stderr, "Failed to create decoder for %s: %s\n", name, opus_strerror(err));
        return -1;
    }

    opus_encoder_ctl(enc, OPUS_SET_MAX_BANDWIDTH(max_bandwidth));
    opus_encoder_ctl(enc, OPUS_SET_COMPLEXITY(10));
    opus_encoder_ctl(enc, OPUS_SET_BITRATE(bitrate));
    if (force_channels > 0) {
        opus_encoder_ctl(enc, OPUS_SET_FORCE_CHANNELS(force_channels));
    }

    int total_frames = N_WARMUP + 1;
    float input[FRAME_SIZE * 2];  /* max 2 channels */

    char path[512];

    /* packets file: [u32 count][u32 len0][bytes0][u32 len1][bytes1]... */
    snprintf(path, sizeof(path), "%s/%s.packets", outdir, name);
    FILE *fpkt = fopen(path, "wb");

    /* pcm file: all decoded f32 frames concatenated */
    snprintf(path, sizeof(path), "%s/%s.pcm", outdir, name);
    FILE *fpcm = fopen(path, "wb");

    unsigned int count = total_frames;
    fwrite(&count, 4, 1, fpkt);

    for (int frame = 0; frame < total_frames; frame++) {
        int sample_offset = frame * FRAME_SIZE;

        if (freq == 0.0f && freq_r == 0.0f) {
            gen_silence(input, FRAME_SIZE * channels);
        } else if (channels == 2) {
            gen_stereo_sine(input, FRAME_SIZE, sample_offset, freq, freq_r, amp);
        } else {
            gen_sine(input, FRAME_SIZE, sample_offset, freq, amp);
        }

        unsigned char packet[MAX_PACKET];
        int pkt_len = opus_encode_float(enc, input, FRAME_SIZE, packet, MAX_PACKET);
        if (pkt_len < 0) {
            fprintf(stderr, "Encode error frame %d of %s: %s\n", frame, name, opus_strerror(pkt_len));
            return -1;
        }

        float pcm_out[FRAME_SIZE * 2];  /* max 2 channels */
        int decoded = opus_decode_float(dec, packet, pkt_len, pcm_out, FRAME_SIZE, 0);
        if (decoded < 0) {
            fprintf(stderr, "Decode error frame %d of %s: %s\n", frame, name, opus_strerror(decoded));
            return -1;
        }

        unsigned int len32 = pkt_len;
        fwrite(&len32, 4, 1, fpkt);
        fwrite(packet, 1, pkt_len, fpkt);
        fwrite(pcm_out, sizeof(float), decoded * channels, fpcm);

        if (frame == 0) {
            int bw = opus_packet_get_bandwidth(packet);
            int mode;
            if (packet[0] & 0x80) mode = 1002;
            else if ((packet[0] & 0x60) == 0x60) mode = 1001;
            else mode = 1000;
            int nb_ch = opus_packet_get_nb_channels(packet);
            printf("  %s: pkt[0]=%d bytes, mode=%d, bw=%d, ch=%d\n",
                   name, pkt_len, mode, bw, nb_ch);
        }
    }

    fclose(fpkt);
    fclose(fpcm);

    /* Write info */
    snprintf(path, sizeof(path), "%s/%s.info", outdir, name);
    FILE *finfo = fopen(path, "w");
    fprintf(finfo, "frames=%d\nsamples_per_frame=%d\nchannels=%d\n",
            total_frames, FRAME_SIZE, channels);
    fclose(finfo);

    opus_encoder_destroy(enc);
    opus_decoder_destroy(dec);
    return 0;
}

int main(int argc, char **argv) {
    const char *outdir = "tests/vectors";
    if (argc > 1) outdir = argv[1];

    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", outdir);
    system(cmd);

    /* ============================================================
     * Mono test cases (existing)
     * ============================================================ */

    printf("=== CELT-only mono test cases ===\n");
    run_test_case("celt_silence", outdir,
        1, OPUS_APPLICATION_RESTRICTED_LOWDELAY, OPUS_BANDWIDTH_FULLBAND,
        64000, 0.0f, 0.0f, 0.0f, 1);

    run_test_case("celt_sine440", outdir,
        1, OPUS_APPLICATION_RESTRICTED_LOWDELAY, OPUS_BANDWIDTH_FULLBAND,
        64000, 440.0f, 0.0f, 0.5f, 1);

    run_test_case("celt_sine1k_hbr", outdir,
        1, OPUS_APPLICATION_RESTRICTED_LOWDELAY, OPUS_BANDWIDTH_FULLBAND,
        128000, 1000.0f, 0.0f, 0.8f, 1);

    run_test_case("celt_lowbr", outdir,
        1, OPUS_APPLICATION_RESTRICTED_LOWDELAY, OPUS_BANDWIDTH_FULLBAND,
        16000, 300.0f, 0.0f, 0.3f, 1);

    printf("\n=== SILK-only mono test cases ===\n");
    run_test_case("silk_nb_silence", outdir,
        1, OPUS_APPLICATION_VOIP, OPUS_BANDWIDTH_NARROWBAND,
        12000, 0.0f, 0.0f, 0.0f, 1);

    run_test_case("silk_nb_sine200", outdir,
        1, OPUS_APPLICATION_VOIP, OPUS_BANDWIDTH_NARROWBAND,
        12000, 200.0f, 0.0f, 0.5f, 1);

    run_test_case("silk_wb_sine500", outdir,
        1, OPUS_APPLICATION_VOIP, OPUS_BANDWIDTH_WIDEBAND,
        20000, 500.0f, 0.0f, 0.6f, 1);

    run_test_case("silk_mb_sine350", outdir,
        1, OPUS_APPLICATION_VOIP, OPUS_BANDWIDTH_MEDIUMBAND,
        16000, 350.0f, 0.0f, 0.4f, 1);

    /* ============================================================
     * Stereo test cases (new)
     * ============================================================ */

    printf("\n=== CELT-only stereo test cases ===\n");

    /* Stereo silence - baseline for stereo CELT */
    run_test_case("celt_stereo_silence", outdir,
        2, OPUS_APPLICATION_RESTRICTED_LOWDELAY, OPUS_BANDWIDTH_FULLBAND,
        64000, 0.0f, 0.0f, 0.0f, 0);

    /* Stereo sine - different freqs per channel, mid bitrate */
    run_test_case("celt_stereo_sine", outdir,
        2, OPUS_APPLICATION_RESTRICTED_LOWDELAY, OPUS_BANDWIDTH_FULLBAND,
        96000, 440.0f, 880.0f, 0.5f, 0);

    /* Stereo low bitrate - likely triggers intensity stereo */
    run_test_case("celt_stereo_lowbr", outdir,
        2, OPUS_APPLICATION_RESTRICTED_LOWDELAY, OPUS_BANDWIDTH_FULLBAND,
        32000, 300.0f, 300.0f, 0.3f, 0);

    /* Stereo high bitrate - likely uses dual stereo */
    run_test_case("celt_stereo_hbr", outdir,
        2, OPUS_APPLICATION_RESTRICTED_LOWDELAY, OPUS_BANDWIDTH_FULLBAND,
        128000, 1000.0f, 500.0f, 0.7f, 0);

    printf("\n=== SILK-only stereo test cases ===\n");

    /* Stereo SILK wideband */
    run_test_case("silk_stereo_wb", outdir,
        2, OPUS_APPLICATION_VOIP, OPUS_BANDWIDTH_WIDEBAND,
        32000, 400.0f, 600.0f, 0.5f, 0);

    /* Stereo SILK narrowband */
    run_test_case("silk_stereo_nb", outdir,
        2, OPUS_APPLICATION_VOIP, OPUS_BANDWIDTH_NARROWBAND,
        20000, 200.0f, 300.0f, 0.4f, 0);

    printf("\n=== Hybrid stereo test cases ===\n");

    /* Hybrid mode: SILK + CELT, stereo
     * Use OPUS_APPLICATION_VOIP with superwideband/fullband to force hybrid.
     * The encoder picks hybrid when bandwidth >= SWB and application is VOIP/AUDIO. */
    run_test_case("hybrid_stereo", outdir,
        2, OPUS_APPLICATION_VOIP, OPUS_BANDWIDTH_SUPERWIDEBAND,
        32000, 200.0f, 1000.0f, 0.5f, 0);

    /* Hybrid mode: fullband, moderate bitrate - forces hybrid for speech+music */
    run_test_case("hybrid_stereo_fb", outdir,
        2, OPUS_APPLICATION_VOIP, OPUS_BANDWIDTH_FULLBAND,
        36000, 440.0f, 880.0f, 0.6f, 0);

    printf("\nDone.\n");
    return 0;
}
