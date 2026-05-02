/* Compare cwrsi output between C and print results for specific N,K cases */
#include <stdio.h>
#include <stdlib.h>
#include "entdec.h"
#include "cwrs.h"
#include "bands.h"
#include "os_support.h"

/* Test cwrsi for specific N,K,i values */
static void test_cwrsi(int n, int k, unsigned int idx) {
    int y[256];
    int i;
    /* Use the decode_pulses path which calls cwrsi internally */
    /* Or directly call cwrsi - but it's static in non-SMALL_FOOTPRINT */
    /* Instead, compute V(N,K) and encode/decode roundtrip */

    /* For non-SMALL_FOOTPRINT, we use the table-based approach */
    /* Let's just enumerate all possible index values and print the decoded vector */

    printf("cwrsi(N=%d, K=%d, i=%u):", n, k, idx);

    /* We need to use encode_pulses/decode_pulses for a roundtrip test */
    /* Actually, let me just call decode_pulses with a crafted packet */

    /* Create a packet that encodes the index 'idx' as a uint with range V(N,K) */
    unsigned char buf[256];
    ec_enc enc;
    ec_dec dec;

    ec_enc_init(&enc, buf, sizeof(buf));

    /* Need to get V(N,K) */
    /* V(N,K) = CELT_PVQ_V(N,K) but this is a macro in the C code */
    /* For simplicity, encode a known pulse vector and see what we get */

    /* Alternative: manually construct the test by encoding a known vector */
    int test_y[16] = {0};

    /* Let's just print V(N,K) for verification */
    /* Access through decode_pulses */

    /* Actually, let's use the ec_enc_uint/ec_dec_uint approach */
    /* First determine V(N,K) by trying to encode a zero vector - won't work */

    /* Simpler: just use known test vectors */
    printf(" (skipping direct cwrsi test, need to verify via roundtrip)\n");
}

/* Main: generate specific CWRS test data */
int main() {
    /* For the failing band: N=16, K=7 */
    /* Generate all possible vectors with K=1 pulses for small N */
    /* And a specific case for N=16, K=7 */

    int n, k;
    unsigned char buf[4096];
    ec_enc enc;
    ec_dec dec;

    /* Test: encode specific pulse vectors and show the decoded result */
    int test_cases[][2] = {
        {2, 1}, {4, 2}, {8, 3}, {16, 7}, {16, 1}, {8, 7},
    };
    int ntests = sizeof(test_cases) / sizeof(test_cases[0]);

    for (int t = 0; t < ntests; t++) {
        n = test_cases[t][0];
        k = test_cases[t][1];

        /* Create a simple test: encode a known vector, decode it */
        int y_enc[16] = {0};
        int y_dec[16] = {0};

        /* Put all pulses in first position */
        y_enc[0] = k;

        ec_enc_init(&enc, buf, sizeof(buf));
        encode_pulses(y_enc, n, k, &enc);
        ec_enc_done(&enc);

        ec_dec_init(&dec, buf, enc.storage);
        decode_pulses(y_dec, n, k, &dec);

        printf("N=%2d K=%d: enc=[", n, k);
        for (int i = 0; i < n; i++) printf("%d%s", y_enc[i], i<n-1?",":"");
        printf("] -> dec=[");
        for (int i = 0; i < n; i++) printf("%d%s", y_dec[i], i<n-1?",":"");
        printf("] %s\n", memcmp(y_enc, y_dec, n*sizeof(int)) == 0 ? "OK" : "MISMATCH");
    }

    /* More interesting: test with various pulse distributions */
    printf("\n--- Specific pulse vectors for N=16 K=7 ---\n");
    {
        int vectors[][16] = {
            {7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7},
            {1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0},
            {-1,1,-1,1,-1,1,-1,0,0,0,0,0,0,0,0,0},
            {3,2,1,1,0,0,0,0,0,0,0,0,0,0,0,0},
            {-2,0,3,0,0,0,0,-1,0,0,0,1,0,0,0,0},
        };
        int nvec = sizeof(vectors) / sizeof(vectors[0]);

        for (int v = 0; v < nvec; v++) {
            int y_dec[16] = {0};

            ec_enc_init(&enc, buf, sizeof(buf));
            encode_pulses(vectors[v], 16, 7, &enc);
            ec_enc_done(&enc);

            ec_dec_init(&dec, buf, enc.storage);
            decode_pulses(y_dec, 16, 7, &dec);

            int match = 1;
            for (int i = 0; i < 16; i++) {
                if (vectors[v][i] != y_dec[i]) { match = 0; break; }
            }

            printf("  [");
            for (int i = 0; i < 16; i++) printf("%2d%s", vectors[v][i], i<15?",":"");
            printf("] -> [");
            for (int i = 0; i < 16; i++) printf("%2d%s", y_dec[i], i<15?",":"");
            printf("] %s\n", match ? "OK" : "MISMATCH");

            /* Also print the raw encoded bytes for Rust comparison */
            printf("    buf[0..%u] =", enc.storage);
            for (unsigned int i = 0; i < enc.storage && i < 20; i++) printf(" %02x", buf[i]);
            printf("\n");
        }
    }

    return 0;
}
