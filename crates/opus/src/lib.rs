pub mod error;
pub mod types;
pub mod packet;
pub mod decoder;
pub mod encoder;
pub mod repacketizer;
pub mod multistream;
pub mod multistream_encoder;
pub mod projection;
pub mod extensions;

pub use error::OpusError;
pub use types::*;
pub use decoder::OpusDecoder;
pub use encoder::OpusEncoder;
pub use multistream::OpusMSDecoder;
pub use multistream_encoder::OpusMSEncoder;
pub use projection::{OpusProjectionEncoder, MappingMatrix};
pub use extensions::{
    opus_packet_extensions_parse, opus_packet_extensions_generate,
    OpusExtensionData, OPUS_MAX_EXTENSIONS,
};
pub use packet::{
    opus_packet_get_bandwidth, opus_packet_get_mode, opus_packet_get_nb_channels,
    opus_packet_get_nb_frames, opus_packet_get_nb_samples, opus_packet_get_samples_per_frame,
    opus_packet_parse, opus_packet_parse_self_delimited, ParsedPacket,
};
pub use repacketizer::OpusRepacketizer;
