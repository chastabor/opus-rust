pub mod error;
pub mod packet;
pub mod decoder;
pub mod repacketizer;

pub use error::OpusError;
pub use decoder::OpusDecoder;
pub use packet::{
    opus_packet_get_bandwidth, opus_packet_get_mode, opus_packet_get_nb_channels,
    opus_packet_get_nb_frames, opus_packet_get_nb_samples, opus_packet_get_samples_per_frame,
    opus_packet_parse, ParsedPacket,
    OPUS_BANDWIDTH_NARROWBAND, OPUS_BANDWIDTH_MEDIUMBAND, OPUS_BANDWIDTH_WIDEBAND,
    OPUS_BANDWIDTH_SUPERWIDEBAND, OPUS_BANDWIDTH_FULLBAND,
    MODE_SILK_ONLY, MODE_HYBRID, MODE_CELT_ONLY,
};
pub use repacketizer::OpusRepacketizer;
