use crate::tables::CORRECTION;

/// Number of bits to output at a time.
pub const EC_SYM_BITS: u32 = 8;
/// Total number of bits in each of the state registers.
pub const EC_CODE_BITS: u32 = 32;
/// Maximum symbol value.
pub const EC_SYM_MAX: u32 = (1 << EC_SYM_BITS) - 1;
/// Bits to shift by to move a symbol into the high-order position.
pub const EC_CODE_SHIFT: u32 = EC_CODE_BITS - EC_SYM_BITS - 1;
/// Carry bit of the high-order range symbol.
pub const EC_CODE_TOP: u32 = 1 << (EC_CODE_BITS - 1);
/// Low-order bit of the high-order range symbol.
pub const EC_CODE_BOT: u32 = EC_CODE_TOP >> EC_SYM_BITS;
/// Number of bits available for the last, partial symbol in the code field.
pub const EC_CODE_EXTRA: u32 = (EC_CODE_BITS - 2) % EC_SYM_BITS + 1;

/// Window size in bits (always 32 for u32 window).
pub const EC_WINDOW_SIZE: u32 = 32;

/// Number of bits to use for the range-coded part of unsigned integers.
pub const EC_UINT_BITS: u32 = 8;

/// Resolution of fractional-precision bit usage measurements (1/8th bits).
pub const BITRES: u32 = 3;

/// The entropy encoder/decoder context.
#[derive(Clone)]
pub struct EcCtx {
    /// Buffered input/output.
    pub buf: Vec<u8>,
    /// The size of the buffer.
    pub storage: u32,
    /// The offset at which the last byte containing raw bits was read/written.
    pub end_offs: u32,
    /// Bits that will be read from/written at the end.
    pub end_window: u32,
    /// Number of valid bits in end_window.
    pub nend_bits: i32,
    /// The total number of whole bits read/written.
    pub nbits_total: i32,
    /// The offset at which the next range coder byte will be read/written.
    pub offs: u32,
    /// The number of values in the current range.
    pub rng: u32,
    /// In the decoder: the difference between the top of the current range and
    /// the input value, minus one.
    /// In the encoder: the low end of the current range.
    pub val: u32,
    /// In the decoder: the saved normalization factor from ec_decode().
    /// In the encoder: the number of outstanding carry propagating symbols.
    pub ext: u32,
    /// A buffered input/output symbol, awaiting carry propagation.
    pub rem: i32,
    /// Nonzero if an error occurred.
    pub error: i32,
}

/// Lightweight snapshot of EcCtx scalar fields.
/// Matches C semantics where `ec_bak = ec_encoder` copies only the struct
/// scalars and shares the same buffer pointer. No heap allocation.
#[derive(Clone, Copy)]
pub struct EcSnapshot {
    pub storage: u32,
    pub end_offs: u32,
    pub end_window: u32,
    pub nend_bits: i32,
    pub nbits_total: i32,
    pub offs: u32,
    pub rng: u32,
    pub val: u32,
    pub ext: u32,
    pub rem: i32,
    pub error: i32,
}

impl EcSnapshot {
    /// Returns the number of bits "used" at the time of the snapshot.
    #[inline]
    pub fn tell(&self) -> i32 {
        ec_tell_impl(self.nbits_total, self.rng)
    }
}

/// Compute bits used from nbits_total and rng. Shared by EcCtx and EcSnapshot.
#[inline]
fn ec_tell_impl(nbits_total: i32, rng: u32) -> i32 {
    nbits_total - ec_ilog(rng) as i32
}

impl Default for EcCtx {
    fn default() -> Self {
        Self::new()
    }
}

impl EcCtx {
    pub fn new() -> Self {
        EcCtx {
            buf: Vec::new(),
            storage: 0,
            end_offs: 0,
            end_window: 0,
            nend_bits: 0,
            nbits_total: 0,
            offs: 0,
            rng: 0,
            val: 0,
            ext: 0,
            rem: 0,
            error: 0,
        }
    }

    /// Returns the number of bytes in the range coded part of the output.
    #[inline]
    pub fn range_bytes(&self) -> u32 {
        self.offs
    }

    /// Returns a slice of the output buffer.
    #[inline]
    pub fn get_buffer(&self) -> &[u8] {
        &self.buf
    }

    /// Returns whether an error has occurred.
    #[inline]
    pub fn get_error(&self) -> bool {
        self.error != 0
    }

    /// Returns the number of bits "used" by the encoded or decoded symbols so far.
    #[inline]
    pub fn tell(&self) -> i32 {
        ec_tell_impl(self.nbits_total, self.rng)
    }

    /// Returns the number of bits "used" scaled by 2^BITRES (1/8th bit precision).
    pub fn tell_frac(&self) -> u32 {
        let nbits = (self.nbits_total as u32) << BITRES;
        let l = ec_ilog(self.rng);
        let r = self.rng >> (l - 16);
        let mut b = (r >> 12).wrapping_sub(8);
        b = b.wrapping_add(if r > CORRECTION[b as usize] { 1 } else { 0 });
        let l_total = (l << 3) + b;
        nbits.wrapping_sub(l_total)
    }

    /// Save a lightweight snapshot of the encoder/decoder state.
    /// Only copies scalar fields (44 bytes). No heap allocation.
    /// The buffer is shared — the caller must not modify the buffer
    /// beyond the snapshot's `offs` position after restoring.
    ///
    /// This matches C semantics where `ec_bak = ec_encoder` is a
    /// shallow struct copy sharing the same `buf` pointer.
    #[inline]
    pub fn save_state(&self) -> EcSnapshot {
        EcSnapshot {
            storage: self.storage,
            end_offs: self.end_offs,
            end_window: self.end_window,
            nend_bits: self.nend_bits,
            nbits_total: self.nbits_total,
            offs: self.offs,
            rng: self.rng,
            val: self.val,
            ext: self.ext,
            rem: self.rem,
            error: self.error,
        }
    }

    /// Restore a previously saved snapshot.
    /// Rolls back the encoder/decoder scalar state without touching the buffer.
    /// Buffer contents written before the snapshot are preserved; bytes written
    /// after the snapshot position may be overwritten by subsequent encoding.
    #[inline]
    pub fn restore_state(&mut self, saved: &EcSnapshot) {
        self.storage = saved.storage;
        self.end_offs = saved.end_offs;
        self.end_window = saved.end_window;
        self.nend_bits = saved.nend_bits;
        self.nbits_total = saved.nbits_total;
        self.offs = saved.offs;
        self.rng = saved.rng;
        self.val = saved.val;
        self.ext = saved.ext;
        self.rem = saved.rem;
        self.error = saved.error;
    }
}

/// Integer binary logarithm of a 32-bit value.
/// Returns floor(log2(v)) + 1, or 0 if v == 0.
/// Equivalent to the position of the highest set bit + 1.
#[inline]
pub fn ec_ilog(v: u32) -> u32 {
    if v == 0 { 0 } else { 32 - v.leading_zeros() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ec_ilog() {
        assert_eq!(ec_ilog(0), 0);
        assert_eq!(ec_ilog(1), 1);
        assert_eq!(ec_ilog(2), 2);
        assert_eq!(ec_ilog(3), 2);
        assert_eq!(ec_ilog(4), 3);
        assert_eq!(ec_ilog(255), 8);
        assert_eq!(ec_ilog(256), 9);
        assert_eq!(ec_ilog(0xFFFFFFFF), 32);
    }

    #[test]
    fn test_snapshot_roundtrip() {
        let mut ctx = EcCtx::new();
        ctx.offs = 42;
        ctx.rng = 12345;
        ctx.nbits_total = 100;
        ctx.rem = 7;

        let snap = ctx.save_state();
        ctx.offs = 99;
        ctx.rng = 0;
        ctx.nbits_total = 999;
        ctx.rem = -1;

        ctx.restore_state(&snap);
        assert_eq!(ctx.offs, 42);
        assert_eq!(ctx.rng, 12345);
        assert_eq!(ctx.nbits_total, 100);
        assert_eq!(ctx.rem, 7);
    }
}
