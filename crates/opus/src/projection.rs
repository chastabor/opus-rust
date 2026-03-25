//! Projection encoder for Opus.
//!
//! Port of C opus_projection_encoder.c. The projection encoder applies a
//! mixing matrix to the input channels before feeding them to a multistream
//! encoder. This allows encoding Ambisonic and other non-channel-based
//! spatial audio formats.

use crate::error::OpusError;
use crate::multistream_encoder::OpusMSEncoder;
use crate::types::*;

/// A mixing/demixing matrix in column-major order.
///
/// Used to transform input channels before encoding (mixing matrix) and to
/// transform decoded channels back to the original domain (demixing matrix).
///
/// The matrix has `rows * cols` entries stored in column-major order,
/// meaning `data[row + col * rows]` is the element at (row, col).
///
/// Values are stored as Q15 fixed-point (i16), with an additional gain
/// in S7.8 dB format applied to the entire matrix.
#[derive(Clone)]
pub struct MappingMatrix {
    /// Number of rows.
    pub rows: i32,
    /// Number of columns.
    pub cols: i32,
    /// Overall gain in Q8 dB (S7.8 format: value/256 = dB).
    pub gain: i32,
    /// Matrix data in column-major order, Q15 fixed-point.
    pub data: Vec<i16>,
}

impl MappingMatrix {
    /// Create a new mapping matrix.
    ///
    /// `data` must have exactly `rows * cols` entries in column-major order.
    pub fn new(rows: i32, cols: i32, gain: i32, data: Vec<i16>) -> Result<Self, OpusError> {
        if rows <= 0 || cols <= 0 {
            return Err(OpusError::BadArg);
        }
        if data.len() != (rows * cols) as usize {
            return Err(OpusError::BadArg);
        }
        Ok(MappingMatrix {
            rows,
            cols,
            gain,
            data,
        })
    }

    /// Create an identity matrix of given size with Q15 representation.
    pub fn identity(size: i32) -> Self {
        let n = size as usize;
        let mut data = vec![0i16; n * n];
        for i in 0..n {
            // Q15: 1.0 = 32767 (approximately)
            data[i + i * n] = 32767;
        }
        MappingMatrix {
            rows: size,
            cols: size,
            gain: 0,
            data,
        }
    }

    /// Get element at (row, col) (column-major indexing).
    pub fn get(&self, row: i32, col: i32) -> i16 {
        self.data[(row + col * self.rows) as usize]
    }

    /// Apply the matrix to input samples, producing output samples.
    ///
    /// - `input`: interleaved samples with `input_channels` channels
    /// - `input_channels`: number of input channels (must equal `self.cols`)
    /// - `output`: interleaved output with `output_channels` channels
    /// - `output_channels`: number of output channels (must equal `self.rows`)
    /// - `frame_size`: number of samples per channel
    pub fn apply(
        &self,
        input: &[f32],
        input_channels: usize,
        output: &mut [f32],
        output_channels: usize,
        frame_size: usize,
    ) -> Result<(), OpusError> {
        if input_channels != self.cols as usize {
            return Err(OpusError::BadArg);
        }
        if output_channels != self.rows as usize {
            return Err(OpusError::BadArg);
        }
        if input.len() < frame_size * input_channels {
            return Err(OpusError::BadArg);
        }
        if output.len() < frame_size * output_channels {
            return Err(OpusError::BadArg);
        }

        // Compute the gain multiplier from Q8 dB
        let gain_db = self.gain as f64 / 256.0;
        let gain_lin = 10.0f64.powf(gain_db / 20.0);

        // Q15 normalization factor
        let q15_scale = 1.0 / 32768.0;

        for i in 0..frame_size {
            for out_ch in 0..output_channels {
                let mut sum = 0.0f64;
                for in_ch in 0..input_channels {
                    let coeff =
                        self.data[out_ch as usize + in_ch * output_channels] as f64 * q15_scale;
                    sum += coeff * input[i * input_channels + in_ch] as f64;
                }
                output[i * output_channels + out_ch] = (sum * gain_lin) as f32;
            }
        }

        Ok(())
    }
}

/// Projection encoder that wraps a multistream encoder with mixing/demixing
/// matrices.
///
/// The projection encoder applies a mixing matrix to the input audio before
/// encoding with the multistream encoder. A corresponding demixing matrix
/// is stored for later use by the projection decoder.
pub struct OpusProjectionEncoder {
    /// Matrix applied to input channels to produce encoder input.
    mixing_matrix: MappingMatrix,
    /// Matrix for the decoder side (stored but not used during encoding).
    demixing_matrix: MappingMatrix,
    /// The underlying multistream encoder.
    ms_encoder: OpusMSEncoder,
    /// Number of input channels (before mixing).
    input_channels: usize,
}

impl OpusProjectionEncoder {
    /// Create a new projection encoder.
    ///
    /// - `fs`: Sample rate (8000, 12000, 16000, 24000, or 48000 Hz)
    /// - `input_channels`: Number of input channels
    /// - `mixing_matrix`: Matrix that maps input channels to encoder streams.
    ///   Must have `rows = encoder_channels` and `cols = input_channels`.
    /// - `demixing_matrix`: Matrix for the decoder (stored for metadata).
    /// - `streams`: Number of streams for the multistream encoder
    /// - `coupled_streams`: Number of coupled (stereo) streams
    /// - `mapping`: Channel mapping for the multistream encoder
    /// - `application`: Opus application type
    pub fn new(
        sample_rate: SampleRate,
        input_channels: usize,
        mixing_matrix: MappingMatrix,
        demixing_matrix: MappingMatrix,
        streams: usize,
        coupled_streams: usize,
        mapping: &[u8],
        application: Application,
    ) -> Result<Self, OpusError> {
        if input_channels == 0 || input_channels > 255 {
            return Err(OpusError::BadArg);
        }
        if mixing_matrix.cols as usize != input_channels {
            return Err(OpusError::BadArg);
        }

        let encoder_channels = mixing_matrix.rows as usize;
        if encoder_channels == 0 || encoder_channels > 255 {
            return Err(OpusError::BadArg);
        }

        let ms_encoder = OpusMSEncoder::new(
            sample_rate,
            encoder_channels,
            streams,
            coupled_streams,
            mapping,
            application,
        )?;

        Ok(OpusProjectionEncoder {
            mixing_matrix,
            demixing_matrix,
            ms_encoder,
            input_channels,
        })
    }

    /// Create a simple projection encoder with an identity mixing matrix.
    ///
    /// This is a convenience constructor that creates a projection encoder
    /// that behaves identically to a standard multistream encoder.
    pub fn new_identity(
        sample_rate: SampleRate,
        channels: usize,
        streams: usize,
        coupled_streams: usize,
        mapping: &[u8],
        application: Application,
    ) -> Result<Self, OpusError> {
        let identity = MappingMatrix::identity(channels as i32);
        let demix_identity = MappingMatrix::identity(channels as i32);

        Self::new(
            sample_rate,
            channels,
            identity,
            demix_identity,
            streams,
            coupled_streams,
            mapping,
            application,
        )
    }

    /// Set the total target bitrate for all streams.
    pub fn set_bitrate(&mut self, bitrate: i32) {
        self.ms_encoder.set_bitrate(bitrate);
    }

    /// Get the number of input channels.
    pub fn input_channels(&self) -> usize {
        self.input_channels
    }

    /// Get the number of streams.
    pub fn streams(&self) -> usize {
        self.ms_encoder.streams()
    }

    /// Get the number of coupled (stereo) streams.
    pub fn coupled_streams(&self) -> usize {
        self.ms_encoder.coupled_streams()
    }

    /// Get the sample rate.
    pub fn sample_rate(&self) -> i32 {
        self.ms_encoder.sample_rate()
    }

    /// Get a reference to the mixing matrix.
    pub fn mixing_matrix(&self) -> &MappingMatrix {
        &self.mixing_matrix
    }

    /// Get a reference to the demixing matrix.
    pub fn demixing_matrix(&self) -> &MappingMatrix {
        &self.demixing_matrix
    }

    /// Encode from float PCM.
    ///
    /// - `pcm`: Interleaved input with `input_channels` channels
    /// - `frame_size`: Samples per channel
    /// - `data`: Output buffer
    /// - `max_data_bytes`: Maximum output size
    ///
    /// Returns the number of bytes written.
    pub fn encode_float(
        &mut self,
        pcm: &[f32],
        frame_size: i32,
        data: &mut [u8],
        max_data_bytes: i32,
    ) -> Result<i32, OpusError> {
        if frame_size <= 0 {
            return Err(OpusError::BadArg);
        }

        let encoder_channels = self.mixing_matrix.rows as usize;

        // Apply mixing matrix to transform input channels to encoder channels
        let mut mixed = vec![0.0f32; frame_size as usize * encoder_channels];

        self.mixing_matrix.apply(
            pcm,
            self.input_channels,
            &mut mixed,
            encoder_channels,
            frame_size as usize,
        )?;

        // Encode the mixed audio with the multistream encoder
        self.ms_encoder
            .encode_float(&mixed, frame_size, data, max_data_bytes)
    }

    /// Encode from i16 PCM.
    pub fn encode(
        &mut self,
        pcm: &[i16],
        frame_size: i32,
        data: &mut [u8],
        max_data_bytes: i32,
    ) -> Result<i32, OpusError> {
        let total_samples = frame_size as usize * self.input_channels;
        if pcm.len() < total_samples {
            return Err(OpusError::BadArg);
        }

        let float_pcm: Vec<f32> = pcm[..total_samples]
            .iter()
            .map(|&s| s as f32 * (1.0 / 32768.0))
            .collect();

        self.encode_float(&float_pcm, frame_size, data, max_data_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mapping_matrix_create() {
        // 2x2 identity matrix in Q15
        let data = vec![32767i16, 0, 0, 32767];
        let m = MappingMatrix::new(2, 2, 0, data).unwrap();
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 2);
        assert_eq!(m.get(0, 0), 32767);
        assert_eq!(m.get(0, 1), 0);
        assert_eq!(m.get(1, 0), 0);
        assert_eq!(m.get(1, 1), 32767);
    }

    #[test]
    fn test_mapping_matrix_invalid() {
        // Wrong data length
        assert!(MappingMatrix::new(2, 2, 0, vec![0; 3]).is_err());
        // Zero dimensions
        assert!(MappingMatrix::new(0, 2, 0, vec![]).is_err());
        assert!(MappingMatrix::new(2, 0, 0, vec![]).is_err());
    }

    #[test]
    fn test_mapping_matrix_identity() {
        let m = MappingMatrix::identity(3);
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 3);
        assert_eq!(m.get(0, 0), 32767);
        assert_eq!(m.get(1, 1), 32767);
        assert_eq!(m.get(2, 2), 32767);
        assert_eq!(m.get(0, 1), 0);
        assert_eq!(m.get(1, 0), 0);
    }

    #[test]
    fn test_mapping_matrix_apply_identity() {
        let m = MappingMatrix::identity(2);
        let input = vec![1.0f32, 0.5, 0.3, 0.7]; // 2 samples, 2 channels
        let mut output = vec![0.0f32; 4];

        m.apply(&input, 2, &mut output, 2, 2).unwrap();

        // Identity matrix should pass through (approximately, due to Q15 rounding)
        for i in 0..4 {
            let diff = (output[i] - input[i]).abs();
            assert!(
                diff < 0.001,
                "Sample {} should be approximately unchanged: expected {}, got {} (diff={})",
                i,
                input[i],
                output[i],
                diff
            );
        }
    }

    #[test]
    fn test_mapping_matrix_apply_downmix() {
        // 1x2 matrix: mono downmix (0.5 * L + 0.5 * R)
        // Q15: 0.5 ~= 16384
        let data = vec![16384i16, 16384];
        let m = MappingMatrix::new(1, 2, 0, data).unwrap();

        let input = vec![1.0f32, 0.0]; // 1 sample: L=1.0, R=0.0
        let mut output = vec![0.0f32; 1];

        m.apply(&input, 2, &mut output, 1, 1).unwrap();

        // Should be approximately 0.5
        let diff = (output[0] - 0.5).abs();
        assert!(
            diff < 0.01,
            "Downmix should produce ~0.5, got {} (diff={})",
            output[0],
            diff
        );
    }

    #[test]
    fn test_projection_create() {
        // Create with identity matrix: 2-channel, 1 coupled stream
        let enc = OpusProjectionEncoder::new_identity(
            SampleRate::Hz48000,
            2,
            1,
            1,
            &[0, 1],
            Application::Audio,
        );
        assert!(
            enc.is_ok(),
            "Projection encoder creation failed: {:?}",
            enc.err()
        );
        let enc = enc.unwrap();
        assert_eq!(enc.input_channels(), 2);
        assert_eq!(enc.streams(), 1);
        assert_eq!(enc.coupled_streams(), 1);
    }

    #[test]
    fn test_projection_create_with_matrix() {
        // 2-input to 2-encoder, identity-like
        let mixing = MappingMatrix::new(2, 2, 0, vec![32767, 0, 0, 32767]).unwrap();
        let demixing = MappingMatrix::new(2, 2, 0, vec![32767, 0, 0, 32767]).unwrap();

        let enc = OpusProjectionEncoder::new(
            SampleRate::Hz48000,
            2,
            mixing,
            demixing,
            1,
            1,
            &[0, 1],
            Application::Audio,
        );
        assert!(enc.is_ok());
    }

    #[test]
    fn test_projection_encode_stereo() {
        let fs = 48000;
        let frame_size = 960;

        let mut enc = OpusProjectionEncoder::new_identity(
            SampleRate::Hz48000,
            2,
            1,
            1,
            &[0, 1],
            Application::Audio,
        )
        .unwrap();
        enc.set_bitrate(64000);

        // Generate stereo sine
        let mut pcm = vec![0.0f32; frame_size * 2];
        for i in 0..frame_size {
            let s = 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / fs as f32).sin();
            pcm[i * 2] = s;
            pcm[i * 2 + 1] = s;
        }

        let mut data = vec![0u8; 4000];
        let nbytes = enc
            .encode_float(&pcm, frame_size as i32, &mut data, 4000)
            .expect("Projection encode should succeed");
        assert!(nbytes >= 2, "Should produce a non-trivial packet");
    }

    #[test]
    fn test_projection_encode_with_downmix() {
        // 2 input channels -> 1 encoder channel via downmix matrix
        let fs = 48000;
        let frame_size = 960;

        // 1x2 downmix matrix: mono = 0.5*L + 0.5*R (Q15: 16384)
        let mixing = MappingMatrix::new(1, 2, 0, vec![16384, 16384]).unwrap();
        // 2x1 upmix for decoder side
        let demixing = MappingMatrix::new(2, 1, 0, vec![32767, 32767]).unwrap();

        let mut enc = OpusProjectionEncoder::new(
            SampleRate::Hz48000,
            2, // input channels
            mixing,
            demixing,
            1,    // streams
            0,    // coupled streams (mono)
            &[0], // mapping for the 1 encoder channel
            Application::Audio,
        )
        .unwrap();
        enc.set_bitrate(32000);

        let mut pcm = vec![0.0f32; frame_size * 2];
        for i in 0..frame_size {
            let s = 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / fs as f32).sin();
            pcm[i * 2] = s;
            pcm[i * 2 + 1] = s;
        }

        let mut data = vec![0u8; 4000];
        let nbytes = enc
            .encode_float(&pcm, frame_size as i32, &mut data, 4000)
            .expect("Projection encode with downmix should succeed");
        assert!(nbytes >= 1, "Should produce a packet");
    }
}
