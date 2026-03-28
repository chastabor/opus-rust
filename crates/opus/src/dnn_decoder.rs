//! DNN decoder integration: DRED extension parsing, PLC, and OSCE.

use crate::decoder::OpusDecoder;
use crate::extensions::OpusExtensionData;

/// DRED process_stage value indicating latents have been decoded.
const DRED_STAGE_DECODED: i32 = 1;

/// Parse DRED data from a decoded Opus extension (ID 126).
/// Extracts DRED latents from the extension payload and feeds
/// decoded FEC features to the PLC system.
///
/// NOTE: The DRED entropy decoder (`dred_ec_decode`) is currently a stub
/// that does not perform actual range decoding. Full implementation requires
/// opus-range-coder laplace integration and the quantization statistics
/// tables from the model weight data. Until then, this function is
/// structurally complete but produces no decoded latents.
pub fn decoder_process_dred_extension(
    decoder: &mut OpusDecoder,
    extension: &OpusExtensionData,
    dred_frame_offset: i32,
) {
    let dred_ext_id = opus_dnn::dred::DRED_EXTENSION_ID as i32;
    if extension.id != dred_ext_id {
        return;
    }
    let Some(dnn) = decoder.dnn.as_mut() else { return };
    if !dnn.loaded {
        return;
    }

    let _nb_latents = opus_dnn::dred::decoder::dred_ec_decode(
        &mut dnn.dred,
        &extension.data,
        extension.data.len(),
        opus_dnn::dred::DRED_NUM_REDUNDANCY_FRAMES,
        dred_frame_offset,
        &dnn.dred_stats,
    );

    if dnn.dred.nb_latents > 0 && dnn.dred.process_stage == DRED_STAGE_DECODED {
        opus_dnn::dred::rdovae_dec::dred_rdovae_decode_all(
            &mut dnn.rdovae_dec_state,
            &dnn.rdovae_dec,
            &mut dnn.dred.fec_features,
            &dnn.dred.state,
            &dnn.dred.latents,
            dnn.dred.nb_latents,
            dnn.dred.latent_dim,
        );

        for i in 0..dnn.dred.nb_latents * 2 {
            let feature_start = i * opus_dnn::dred::DRED_NUM_FEATURES;
            let feature_end = feature_start + opus_dnn::fargan::NB_FEATURES;
            if feature_end <= dnn.dred.fec_features.len() {
                opus_dnn::lpcnet::plc::lpcnet_plc_fec_add(
                    &mut dnn.plc,
                    Some(&dnn.dred.fec_features[feature_start..feature_end]),
                );
            }
        }
    }
}

/// Update PLC state with a successfully decoded (good) packet.
pub fn decoder_plc_update(decoder: &mut OpusDecoder, pcm: &[i16]) {
    let Some(dnn) = decoder.dnn.as_mut() else { return };
    if !dnn.loaded {
        return;
    }
    opus_dnn::lpcnet::plc::lpcnet_plc_update(&mut dnn.plc, pcm);
    opus_dnn::lpcnet::plc::lpcnet_plc_fec_clear(&mut dnn.plc);
}

/// Conceal a lost packet using DNN-based PLC.
/// Returns true if DNN PLC was applied, false if not available.
pub fn decoder_plc_conceal(decoder: &mut OpusDecoder, pcm: &mut [i16]) -> bool {
    let Some(dnn) = decoder.dnn.as_mut() else { return false };
    if !dnn.loaded {
        return false;
    }
    opus_dnn::lpcnet::plc::lpcnet_plc_conceal(&mut dnn.plc, pcm);
    true
}
