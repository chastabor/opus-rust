/* Dump C WeightArray tables to binary blob files loadable by Rust parse_weights.
   Each record: 64-byte header (WeightHead) + data padded to 64-byte boundary. */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "nnet.h"

/* Include all model weight data. */
#include "pitchdnn_data.c"
#include "plc_data.c"
#include "fargan_data.c"
#include "lace_data.c"
#include "nolace_data.c"
#include "dred_rdovae_enc_data.c"
#include "dred_rdovae_dec_data.c"
#include "dred_rdovae_stats_data.c"
#include "bbwenet_data.c"

#define WEIGHT_BLOCK_SIZE 64

static int write_blob(const char *filename, const WeightArray *arrays) {
    FILE *f = fopen(filename, "wb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", filename); return 1; }

    int i;
    for (i = 0; arrays[i].name != NULL; i++) {
        /* Write 64-byte header */
        char header[WEIGHT_BLOCK_SIZE];
        memset(header, 0, WEIGHT_BLOCK_SIZE);
        memcpy(header, "wght", 4);                           /* head */
        int version = 0;
        memcpy(header + 4, &version, 4);                     /* version */
        int type = arrays[i].type;
        memcpy(header + 8, &type, 4);                        /* type */
        int size = arrays[i].size;
        memcpy(header + 12, &size, 4);                       /* size */
        int block_size = ((size + WEIGHT_BLOCK_SIZE - 1) / WEIGHT_BLOCK_SIZE) * WEIGHT_BLOCK_SIZE;
        memcpy(header + 16, &block_size, 4);                 /* block_size */
        strncpy(header + 20, arrays[i].name, 43);            /* name (null-terminated) */
        fwrite(header, 1, WEIGHT_BLOCK_SIZE, f);

        /* Write data + padding */
        if (size > 0 && arrays[i].data != NULL) {
            fwrite(arrays[i].data, 1, size, f);
        }
        int padding = block_size - size;
        if (padding > 0) {
            char zeros[WEIGHT_BLOCK_SIZE];
            memset(zeros, 0, WEIGHT_BLOCK_SIZE);
            fwrite(zeros, 1, padding, f);
        }
    }

    fclose(f);
    fprintf(stderr, "Wrote %s (%d arrays)\n", filename, i);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <output_dir>\n", argv[0]);
        return 1;
    }
    char path[1024];

    snprintf(path, sizeof(path), "%s/pitchdnn.bin", argv[1]);
    write_blob(path, pitchdnn_arrays);

    snprintf(path, sizeof(path), "%s/plcmodel.bin", argv[1]);
    write_blob(path, plcmodel_arrays);

    snprintf(path, sizeof(path), "%s/fargan.bin", argv[1]);
    write_blob(path, fargan_arrays);

    snprintf(path, sizeof(path), "%s/lace.bin", argv[1]);
    write_blob(path, lacelayers_arrays);

    snprintf(path, sizeof(path), "%s/nolace.bin", argv[1]);
    write_blob(path, nolacelayers_arrays);

    snprintf(path, sizeof(path), "%s/rdovae_enc.bin", argv[1]);
    write_blob(path, rdovaeenc_arrays);

    snprintf(path, sizeof(path), "%s/rdovae_dec.bin", argv[1]);
    write_blob(path, rdovaedec_arrays);

    snprintf(path, sizeof(path), "%s/bbwenet.bin", argv[1]);
    write_blob(path, bbwenetlayers_arrays);

    return 0;
}
