//! Top-level DNN model loading from binary weight blobs.
//!
//! These functions parse a single binary weight blob (the format loaded by
//! `OPUS_SET_DNN_BLOB` in C libopus) and initialize the DNN models needed
//! by the encoder or decoder.

use crate::nnet::weights::{WeightError, parse_weights};

/// Load encoder DNN models from a binary weight blob.
///
/// Initializes the RDOVAE encoder model and PitchDNN (via LpcnetEncState)
/// needed for DRED encoding. Matches the C `dred_encoder_load_model` path.
///
/// Returns `(rdovae_enc, lpcnet_enc_state)` on success.
pub fn load_encoder_dnn(
    data: &[u8],
) -> Result<
    (
        crate::dred::rdovae_enc::RdovaeEnc,
        crate::lpcnet::enc::LpcnetEncState,
    ),
    WeightError,
> {
    let arrays = parse_weights(data).ok_or(WeightError)?;

    // Init RDOVAE encoder model
    let rdovae_enc = crate::dred::rdovae_enc::init_rdovae_enc(&arrays)?;

    // Init PitchDNN model (needed by LpcnetEncState for feature extraction)
    let pitchdnn_model = crate::pitchdnn::init_pitchdnn(&arrays)?;
    let pitchdnn_state = crate::pitchdnn::pitchdnn_state_init(pitchdnn_model);
    let lpcnet_enc_state = crate::lpcnet::enc::LpcnetEncState::new(pitchdnn_state);

    Ok((rdovae_enc, lpcnet_enc_state))
}

/// Load decoder DNN models from a binary weight blob.
///
/// Initializes the RDOVAE decoder, PLC model (PlcModel + FARGAN + PitchDNN),
/// and OSCE models needed for DNN-enhanced decoding. Matches the C
/// `lpcnet_plc_load_model` + `silk_LoadOSCEModels` path.
///
/// Returns `(rdovae_dec, plc_state, osce_model)` on success.
pub fn load_decoder_dnn(
    data: &[u8],
) -> Result<
    (
        crate::dred::rdovae_dec::RdovaeDec,
        crate::lpcnet::plc::LpcnetPlcState,
        crate::osce::structs::OsceModel,
    ),
    WeightError,
> {
    let arrays = parse_weights(data).ok_or(WeightError)?;

    // Init RDOVAE decoder model
    let rdovae_dec = crate::dred::rdovae_dec::init_rdovae_dec(&arrays)?;

    // Init PLC: PlcModel + FARGAN + PitchDNN -> LpcnetPlcState
    let plc_model = crate::lpcnet::plc::init_plcmodel(&arrays)?;
    let pitchdnn_model = crate::pitchdnn::init_pitchdnn(&arrays)?;
    let pitchdnn_state = crate::pitchdnn::pitchdnn_state_init(pitchdnn_model);
    let enc_state = crate::lpcnet::enc::LpcnetEncState::new(pitchdnn_state);
    let fargan_model = crate::fargan::init_fargan(&arrays)?;
    let fargan_state = crate::fargan::fargan_state_init(fargan_model);
    let plc_state = crate::lpcnet::plc::lpcnet_plc_init(plc_model, fargan_state, enc_state);

    // Init OSCE models
    let mut osce_model = crate::osce::structs::OsceModel::default();
    let _ = crate::osce::osce_load_models(&mut osce_model, data, data.len());

    Ok((rdovae_dec, plc_state, osce_model))
}
