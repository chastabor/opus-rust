// Port of silk/float/LTP_scale_ctrl_FLP.c: silk_LTP_scale_ctrl_FLP
// Controls LTP scaling based on packet loss rate and SNR.
// Higher loss → more LTP scaling → less reliance on pitch prediction.

use crate::{silk_smulbb, silk_log2lin, CODE_INDEPENDENTLY};
use crate::tables::SILK_LTP_SCALES_TABLE_Q14;

/// Result of LTP scale control.
pub struct LtpScaleResult {
    pub ltp_scale_index: i8,
    pub ltp_scale: f32,
}

/// Port of silk_LTP_scale_ctrl_FLP.
///
/// Determines how much to scale down the LTP prediction based on expected
/// packet loss. Higher loss → larger scale index → more conservative LTP.
///
/// `ltp_pred_cod_gain`: LTP prediction coding gain (dB, Q7 integer from
///   silk_quant_LTP_gains). In the C code this is `psEncCtrl->LTPredCodGain`
///   which is stored as `opus_int` (integer Q7 dB).
pub fn silk_ltp_scale_ctrl_flp(
    ltp_pred_cod_gain_q7: i32,
    snr_db_q7: i32,
    packet_loss_perc: i32,
    n_frames_per_packet: i32,
    lbrr_flag: bool,
    cond_coding: i32,
) -> LtpScaleResult {
    let ltp_scale_index;

    if cond_coding == CODE_INDEPENDENTLY {
        let mut round_loss = packet_loss_perc * n_frames_per_packet;
        if lbrr_flag {
            // LBRR reduces effective loss: square the loss but floor at 2%
            round_loss = 2 + silk_smulbb(round_loss, round_loss) / 100;
        }
        // Two threshold comparisons determine scale index 0, 1, or 2
        let product = silk_smulbb(ltp_pred_cod_gain_q7, round_loss);
        let thresh1 = silk_log2lin(2900 - snr_db_q7);
        let thresh2 = silk_log2lin(3900 - snr_db_q7);
        ltp_scale_index = (product > thresh1) as i8 + (product > thresh2) as i8;
    } else {
        ltp_scale_index = 0;
    }

    let ltp_scale = SILK_LTP_SCALES_TABLE_Q14[ltp_scale_index as usize] as f32 / 16384.0;

    LtpScaleResult {
        ltp_scale_index,
        ltp_scale,
    }
}
