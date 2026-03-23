pub mod error;
pub mod packet;
pub mod decoder;
pub mod encoder;
pub mod repacketizer;
pub mod multistream;

pub use error::OpusError;
pub use decoder::OpusDecoder;
pub use encoder::OpusEncoder;
pub use encoder::{
    OPUS_APPLICATION_VOIP, OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_RESTRICTED_LOWDELAY,
    OPUS_SIGNAL_VOICE, OPUS_SIGNAL_MUSIC, OPUS_AUTO, OPUS_BITRATE_MAX,
};
pub use multistream::OpusMSDecoder;
pub use packet::{
    opus_packet_get_bandwidth, opus_packet_get_mode, opus_packet_get_nb_channels,
    opus_packet_get_nb_frames, opus_packet_get_nb_samples, opus_packet_get_samples_per_frame,
    opus_packet_parse, opus_packet_parse_self_delimited, ParsedPacket,
    OPUS_BANDWIDTH_NARROWBAND, OPUS_BANDWIDTH_MEDIUMBAND, OPUS_BANDWIDTH_WIDEBAND,
    OPUS_BANDWIDTH_SUPERWIDEBAND, OPUS_BANDWIDTH_FULLBAND,
    MODE_SILK_ONLY, MODE_HYBRID, MODE_CELT_ONLY,
};
pub use repacketizer::OpusRepacketizer;
