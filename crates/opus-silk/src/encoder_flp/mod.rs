// Float SILK encoder components — faithful port of silk/float/*.c
//
// Architecture: f32 for all analysis, fixed-point only at NSQ boundary.
// Each function mirrors a specific C function and is tested via FFI.

pub mod dsp;
pub mod wrappers;
pub mod residual_energy;
pub mod process_gains;
pub mod find_lpc;
pub mod find_pred_coefs;
