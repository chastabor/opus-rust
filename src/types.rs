//! Type-safe enums for Opus codec parameters.
//!
//! These enums replace the C-style integer constants from libopus,
//! making invalid parameter values a compile-time error.

use crate::error::OpusError;

/// Opus application type — controls encoder optimization strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Application {
    /// Best for most VoIP/videoconference applications where listening quality
    /// and intelligibility matter most.
    Voip = 2048,
    /// Best for broadcast/high-fidelity application where the decoded audio
    /// should be as close as possible to the input.
    Audio = 2049,
    /// Only use when lowest-achievable latency is what matters most.
    RestrictedLowDelay = 2051,
}

impl From<Application> for i32 {
    fn from(app: Application) -> i32 {
        app as i32
    }
}

impl TryFrom<i32> for Application {
    type Error = OpusError;
    fn try_from(value: i32) -> Result<Self, OpusError> {
        match value {
            2048 => Ok(Application::Voip),
            2049 => Ok(Application::Audio),
            2051 => Ok(Application::RestrictedLowDelay),
            _ => Err(OpusError::BadArg),
        }
    }
}

/// Audio bandwidth — controls the frequency range encoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Bandwidth {
    /// 4 kHz passband.
    Narrowband = 1101,
    /// 6 kHz passband.
    Mediumband = 1102,
    /// 8 kHz passband.
    Wideband = 1103,
    /// 12 kHz passband.
    Superwideband = 1104,
    /// 20 kHz passband (full band).
    Fullband = 1105,
}

impl From<Bandwidth> for i32 {
    fn from(bw: Bandwidth) -> i32 {
        bw as i32
    }
}

impl TryFrom<i32> for Bandwidth {
    type Error = OpusError;
    fn try_from(value: i32) -> Result<Self, OpusError> {
        match value {
            1101 => Ok(Bandwidth::Narrowband),
            1102 => Ok(Bandwidth::Mediumband),
            1103 => Ok(Bandwidth::Wideband),
            1104 => Ok(Bandwidth::Superwideband),
            1105 => Ok(Bandwidth::Fullband),
            _ => Err(OpusError::BadArg),
        }
    }
}

/// Codec mode — which internal codec is used for a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Mode {
    /// SILK encoder only (narrowband to wideband).
    SilkOnly = 1000,
    /// Hybrid mode (SILK + CELT, superwideband/fullband).
    Hybrid = 1001,
    /// CELT encoder only (mediumband to fullband).
    CeltOnly = 1002,
}

impl From<Mode> for i32 {
    fn from(mode: Mode) -> i32 {
        mode as i32
    }
}

impl TryFrom<i32> for Mode {
    type Error = OpusError;
    fn try_from(value: i32) -> Result<Self, OpusError> {
        match value {
            1000 => Ok(Mode::SilkOnly),
            1001 => Ok(Mode::Hybrid),
            1002 => Ok(Mode::CeltOnly),
            _ => Err(OpusError::BadArg),
        }
    }
}

/// Signal type hint — tells the encoder what type of audio to expect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Signal {
    /// Let the encoder detect the signal type automatically.
    Auto,
    /// Bias toward voice-optimized encoding.
    Voice,
    /// Bias toward music-optimized encoding.
    Music,
}

impl From<Signal> for i32 {
    fn from(sig: Signal) -> i32 {
        match sig {
            Signal::Auto => -1000,
            Signal::Voice => 3001,
            Signal::Music => 3002,
        }
    }
}

impl TryFrom<i32> for Signal {
    type Error = OpusError;
    fn try_from(value: i32) -> Result<Self, OpusError> {
        match value {
            -1000 => Ok(Signal::Auto),
            3001 => Ok(Signal::Voice),
            3002 => Ok(Signal::Music),
            _ => Err(OpusError::BadArg),
        }
    }
}

/// Bitrate configuration for the encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bitrate {
    /// Let the encoder choose a bitrate automatically.
    Auto,
    /// Use the maximum bitrate the encoder can produce.
    Max,
    /// Target a specific bitrate in bits per second (clamped to 500..=512000).
    BitsPerSecond(i32),
}

impl From<Bitrate> for i32 {
    fn from(br: Bitrate) -> i32 {
        match br {
            Bitrate::Auto => -1000,
            Bitrate::Max => -1,
            Bitrate::BitsPerSecond(bps) => bps,
        }
    }
}

/// Sample rate for the Opus encoder/decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SampleRate {
    /// 8000 Hz — narrowband telephony.
    Hz8000 = 8000,
    /// 12000 Hz — mediumband.
    Hz12000 = 12000,
    /// 16000 Hz — wideband.
    Hz16000 = 16000,
    /// 24000 Hz — superwideband.
    Hz24000 = 24000,
    /// 48000 Hz — fullband (recommended).
    Hz48000 = 48000,
}

impl From<SampleRate> for i32 {
    fn from(sr: SampleRate) -> i32 {
        sr as i32
    }
}

impl TryFrom<i32> for SampleRate {
    type Error = OpusError;
    fn try_from(value: i32) -> Result<Self, OpusError> {
        match value {
            8000 => Ok(SampleRate::Hz8000),
            12000 => Ok(SampleRate::Hz12000),
            16000 => Ok(SampleRate::Hz16000),
            24000 => Ok(SampleRate::Hz24000),
            48000 => Ok(SampleRate::Hz48000),
            _ => Err(OpusError::BadArg),
        }
    }
}

/// Channel count for the Opus encoder/decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Channels {
    /// Mono (1 channel).
    Mono = 1,
    /// Stereo (2 channels).
    Stereo = 2,
}

impl From<Channels> for i32 {
    fn from(ch: Channels) -> i32 {
        ch as i32
    }
}

impl TryFrom<i32> for Channels {
    type Error = OpusError;
    fn try_from(value: i32) -> Result<Self, OpusError> {
        match value {
            1 => Ok(Channels::Mono),
            2 => Ok(Channels::Stereo),
            _ => Err(OpusError::BadArg),
        }
    }
}

/// Force the encoder to use a specific channel count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ForceChannels {
    /// Let the encoder decide based on the signal.
    Auto,
    /// Force mono encoding.
    Mono,
    /// Force stereo encoding.
    Stereo,
}

impl From<ForceChannels> for i32 {
    fn from(fc: ForceChannels) -> i32 {
        match fc {
            ForceChannels::Auto => -1,
            ForceChannels::Mono => 1,
            ForceChannels::Stereo => 2,
        }
    }
}

impl TryFrom<i32> for ForceChannels {
    type Error = OpusError;
    fn try_from(value: i32) -> Result<Self, OpusError> {
        match value {
            -1 => Ok(ForceChannels::Auto),
            1 => Ok(ForceChannels::Mono),
            2 => Ok(ForceChannels::Stereo),
            _ => Err(OpusError::BadArg),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn application_round_trip() {
        for app in [
            Application::Voip,
            Application::Audio,
            Application::RestrictedLowDelay,
        ] {
            let raw = i32::from(app);
            assert_eq!(Application::try_from(raw).unwrap(), app);
        }
        assert!(Application::try_from(9999).is_err());
    }

    #[test]
    fn bandwidth_round_trip() {
        for bw in [
            Bandwidth::Narrowband,
            Bandwidth::Mediumband,
            Bandwidth::Wideband,
            Bandwidth::Superwideband,
            Bandwidth::Fullband,
        ] {
            let raw = i32::from(bw);
            assert_eq!(Bandwidth::try_from(raw).unwrap(), bw);
        }
        assert!(Bandwidth::try_from(0).is_err());
    }

    #[test]
    fn bandwidth_ordering() {
        assert!(Bandwidth::Narrowband < Bandwidth::Mediumband);
        assert!(Bandwidth::Mediumband < Bandwidth::Wideband);
        assert!(Bandwidth::Wideband < Bandwidth::Superwideband);
        assert!(Bandwidth::Superwideband < Bandwidth::Fullband);
    }

    #[test]
    fn mode_round_trip() {
        for mode in [Mode::SilkOnly, Mode::Hybrid, Mode::CeltOnly] {
            let raw = i32::from(mode);
            assert_eq!(Mode::try_from(raw).unwrap(), mode);
        }
        assert!(Mode::try_from(0).is_err());
    }

    #[test]
    fn signal_round_trip() {
        for sig in [Signal::Auto, Signal::Voice, Signal::Music] {
            let raw = i32::from(sig);
            assert_eq!(Signal::try_from(raw).unwrap(), sig);
        }
        assert!(Signal::try_from(42).is_err());
    }

    #[test]
    fn bitrate_conversion() {
        assert_eq!(i32::from(Bitrate::Auto), -1000);
        assert_eq!(i32::from(Bitrate::Max), -1);
        assert_eq!(i32::from(Bitrate::BitsPerSecond(64000)), 64000);
    }

    #[test]
    fn sample_rate_round_trip() {
        for sr in [
            SampleRate::Hz8000,
            SampleRate::Hz12000,
            SampleRate::Hz16000,
            SampleRate::Hz24000,
            SampleRate::Hz48000,
        ] {
            let raw = i32::from(sr);
            assert_eq!(SampleRate::try_from(raw).unwrap(), sr);
        }
        assert!(SampleRate::try_from(44100).is_err());
    }

    #[test]
    fn channels_round_trip() {
        assert_eq!(Channels::try_from(1).unwrap(), Channels::Mono);
        assert_eq!(Channels::try_from(2).unwrap(), Channels::Stereo);
        assert!(Channels::try_from(3).is_err());
        assert_eq!(i32::from(Channels::Mono), 1);
        assert_eq!(i32::from(Channels::Stereo), 2);
    }

    #[test]
    fn force_channels_round_trip() {
        for fc in [
            ForceChannels::Auto,
            ForceChannels::Mono,
            ForceChannels::Stereo,
        ] {
            let raw = i32::from(fc);
            assert_eq!(ForceChannels::try_from(raw).unwrap(), fc);
        }
        assert!(ForceChannels::try_from(5).is_err());
    }
}
