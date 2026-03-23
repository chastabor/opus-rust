/*
 * Generate multistream test vectors for Opus decoder cross-validation.
 * Creates surround-sound multistream packets and reference decoded PCM.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "opus.h"
#include "opus_multistream.h"

#define SAMPLE_RATE 48000
#define FRAME_MS 20
#define FRAME_SIZE (SAMPLE_RATE * FRAME_MS / 1000)  /* 960 */
#define MAX_PACKET 16000
#define N_WARMUP 5
#define PI_F 3.14159265358979f

static void gen_multichannel_sine(float *buf, int samples, int channels,
                                   int offset, const float *freqs, float amp) {
    for (int i = 0; i < samples; i++) {
        float t = (float)(i + offset) / SAMPLE_RATE;
        for (int ch = 0; ch < channels; ch++) {
            buf[i * channels + ch] = amp * sinf(2.0f * PI_F * freqs[ch] * t);
        }
    }
}

static int run_ms_test_case(
    const char *name, const char *outdir,
    int channels, int streams, int coupled_streams,
    const unsigned char *mapping,
    int application, int bitrate,
    const float *freqs, float amp
) {
    int err;
    OpusMSEncoder *enc = opus_multistream_encoder_create(
        SAMPLE_RATE, channels, streams, coupled_streams, mapping,
        application, &err
    );
    if (err != OPUS_OK || !enc) {
        fprintf(stderr, "Failed to create MS encoder for %s: %d\n", name, err);
        return -1;
    }
    OpusMSDecoder *dec = opus_multistream_decoder_create(
        SAMPLE_RATE, channels, streams, coupled_streams, mapping, &err
    );
    if (err != OPUS_OK || !dec) {
        fprintf(stderr, "Failed to create MS decoder for %s: %d\n", name, err);
        return -1;
    }

    opus_multistream_encoder_ctl(enc, OPUS_SET_BITRATE(bitrate));
    opus_multistream_encoder_ctl(enc, OPUS_SET_COMPLEXITY(10));

    int total_frames = N_WARMUP + 1;
    float input[FRAME_SIZE * 8];  /* max 8 channels */

    char path[512];

    snprintf(path, sizeof(path), "%s/%s.packets", outdir, name);
    FILE *fpkt = fopen(path, "wb");
    snprintf(path, sizeof(path), "%s/%s.pcm", outdir, name);
    FILE *fpcm = fopen(path, "wb");

    unsigned int count = total_frames;
    fwrite(&count, 4, 1, fpkt);

    for (int frame = 0; frame < total_frames; frame++) {
        gen_multichannel_sine(input, FRAME_SIZE, channels,
                              frame * FRAME_SIZE, freqs, amp);

        unsigned char packet[MAX_PACKET];
        int pkt_len = opus_multistream_encode_float(
            enc, input, FRAME_SIZE, packet, MAX_PACKET
        );
        if (pkt_len < 0) {
            fprintf(stderr, "MS encode error frame %d of %s: %s\n",
                    frame, name, opus_strerror(pkt_len));
            return -1;
        }

        float pcm_out[FRAME_SIZE * 8];
        int decoded = opus_multistream_decode_float(
            dec, packet, pkt_len, pcm_out, FRAME_SIZE, 0
        );
        if (decoded < 0) {
            fprintf(stderr, "MS decode error frame %d of %s: %s\n",
                    frame, name, opus_strerror(decoded));
            return -1;
        }

        unsigned int len32 = pkt_len;
        fwrite(&len32, 4, 1, fpkt);
        fwrite(packet, 1, pkt_len, fpkt);
        fwrite(pcm_out, sizeof(float), decoded * channels, fpcm);

        if (frame == 0) {
            printf("  %s: pkt=%d bytes, ch=%d, streams=%d, coupled=%d\n",
                   name, pkt_len, channels, streams, coupled_streams);
        }
    }

    fclose(fpkt);
    fclose(fpcm);

    /* Write info */
    snprintf(path, sizeof(path), "%s/%s.info", outdir, name);
    FILE *finfo = fopen(path, "w");
    fprintf(finfo, "frames=%d\nsamples_per_frame=%d\nchannels=%d\n"
            "streams=%d\ncoupled_streams=%d\nmapping=",
            total_frames, FRAME_SIZE, channels,
            streams, coupled_streams);
    for (int i = 0; i < channels; i++) {
        fprintf(finfo, "%d%s", mapping[i], i < channels - 1 ? "," : "");
    }
    fprintf(finfo, "\n");
    fclose(finfo);

    opus_multistream_encoder_destroy(enc);
    opus_multistream_decoder_destroy(dec);
    return 0;
}

int main(int argc, char **argv) {
    const char *outdir = "tests/vectors";
    if (argc > 1) outdir = argv[1];

    printf("=== Multistream test cases ===\n");

    /* Quad (4 channels): 2 coupled streams */
    {
        const unsigned char mapping[] = {0, 1, 2, 3};
        const float freqs[] = {200.0f, 400.0f, 600.0f, 800.0f};
        run_ms_test_case("ms_quad", outdir,
            4, 2, 2, mapping,
            OPUS_APPLICATION_AUDIO, 128000,
            freqs, 0.4f);
    }

    /* 5.1 surround (6 channels): 2 coupled + 2 mono
     * Vorbis channel order: FL, C, FR, RL, RR, LFE
     * Standard surround mapping for 6 channels */
    {
        const unsigned char mapping[] = {0, 4, 1, 2, 3, 5};
        const float freqs[] = {220.0f, 440.0f, 330.0f, 550.0f, 660.0f, 100.0f};
        run_ms_test_case("ms_surround51", outdir,
            6, 4, 2, mapping,
            OPUS_APPLICATION_AUDIO, 256000,
            freqs, 0.3f);
    }

    printf("\nDone.\n");
    return 0;
}
