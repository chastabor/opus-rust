//! DNN-related types for DRED, OSCE, and deep PLC.
//!
//! These types are available when the `dnn` feature is enabled.
//! They provide CTL-like configuration and the public API surface
//! for DNN features in the Opus encoder and decoder.

/// CTL request IDs for DNN features, matching C opus_defines.h.
pub const OPUS_SET_DRED_DURATION_REQUEST: i32 = 4050;
pub const OPUS_GET_DRED_DURATION_REQUEST: i32 = 4051;
pub const OPUS_SET_DNN_BLOB_REQUEST: i32 = 4052;
pub const OPUS_SET_OSCE_BWE_REQUEST: i32 = 4054;
pub const OPUS_GET_OSCE_BWE_REQUEST: i32 = 4055;

/// DNN decoder state, wrapping the opus-dnn components.
/// Holds the DRED decoder, PLC state, and OSCE model.
///
/// This is stored as an `Option` in `OpusDecoder` — `None` when
/// DNN is not loaded (no weight blob provided).
pub struct DnnDecoderState {
    /// Whether DNN models are loaded and ready.
    pub(crate) loaded: bool,
    /// DRED decoded state (latents, features, offsets).
    pub(crate) dred: opus_dnn::dred::decoder::OpusDred,
    /// DRED quantization statistics (from model weight data).
    pub(crate) dred_stats: opus_dnn::dred::decoder::DredStats,
    /// DRED RDOVAE decoder model.
    pub(crate) rdovae_dec: opus_dnn::dred::rdovae_dec::RdovaeDec,
    /// RDOVAE decoder state (reusable across calls).
    pub(crate) rdovae_dec_state: opus_dnn::dred::rdovae_dec::RdovaeDecState,
    /// LPCNet PLC state (deep packet loss concealment).
    pub(crate) plc: opus_dnn::lpcnet::plc::LpcnetPlcState,
    /// OSCE model (speech enhancement).
    pub(crate) osce: opus_dnn::osce::structs::OsceModel,
}

/// DNN encoder state, wrapping the opus-dnn DRED encoder.
///
/// Stored as `Option` in `OpusEncoder` — `None` when DRED is disabled.
pub struct DnnEncoderState {
    /// Whether DNN models are loaded and ready.
    pub(crate) loaded: bool,
    /// DRED encoder state (latent computation + encoding).
    pub(crate) dred_enc: opus_dnn::dred::encoder::DredEnc,
    /// DRED duration in frames (0 = disabled).
    pub(crate) dred_duration: i32,
}
