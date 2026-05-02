pub mod decoder;
#[cfg(feature = "dnn")]
pub mod dnn_decoder;
#[cfg(feature = "dnn")]
pub mod dnn_silk_bridge;
#[cfg(feature = "dnn")]
pub mod dnn_types;
pub mod encoder;
pub mod error;
pub mod extensions;
pub mod multistream;
pub mod multistream_encoder;
pub mod packet;
pub mod projection;
pub mod repacketizer;
pub mod types;

pub use decoder::OpusDecoder;
pub use encoder::OpusEncoder;
pub use error::OpusError;
pub use extensions::{
    OPUS_MAX_EXTENSIONS, OpusExtensionData, opus_packet_extensions_generate,
    opus_packet_extensions_parse,
};
pub use multistream::OpusMSDecoder;
pub use multistream_encoder::OpusMSEncoder;
pub use packet::{
    ParsedPacket, opus_packet_get_bandwidth, opus_packet_get_mode, opus_packet_get_nb_channels,
    opus_packet_get_nb_frames, opus_packet_get_nb_samples, opus_packet_get_samples_per_frame,
    opus_packet_parse, opus_packet_parse_self_delimited,
};
pub use projection::{MappingMatrix, OpusProjectionEncoder};
pub use repacketizer::OpusRepacketizer;
pub use types::*;
