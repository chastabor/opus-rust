use crate::ctx::*;

impl EcCtx {
    /// Write a byte to the front of the buffer.
    fn write_byte(&mut self, value: u32) -> i32 {
        if self.offs + self.end_offs >= self.storage {
            return -1;
        }
        self.buf[self.offs as usize] = value as u8;
        self.offs += 1;
        0
    }

    /// Write a byte to the end of the buffer.
    fn write_byte_at_end(&mut self, value: u32) -> i32 {
        if self.offs + self.end_offs >= self.storage {
            return -1;
        }
        self.end_offs += 1;
        self.buf[(self.storage - self.end_offs) as usize] = value as u8;
        0
    }

    /// Output a symbol with carry propagation.
    fn enc_carry_out(&mut self, c: i32) {
        if c as u32 != EC_SYM_MAX {
            let carry = (c >> EC_SYM_BITS as i32) as i32;
            if self.rem >= 0 {
                self.error |= self.write_byte((self.rem + carry) as u32);
            }
            if self.ext > 0 {
                let sym = ((EC_SYM_MAX as i32 + carry) as u32) & EC_SYM_MAX;
                loop {
                    self.error |= self.write_byte(sym);
                    self.ext -= 1;
                    if self.ext == 0 {
                        break;
                    }
                }
            }
            self.rem = c & EC_SYM_MAX as i32;
        } else {
            self.ext += 1;
        }
    }

    /// Normalize the encoder state.
    fn enc_normalize(&mut self) {
        while self.rng <= EC_CODE_BOT {
            self.enc_carry_out((self.val >> EC_CODE_SHIFT) as i32);
            self.val = (self.val << EC_SYM_BITS) & (EC_CODE_TOP - 1);
            self.rng <<= EC_SYM_BITS;
            self.nbits_total += EC_SYM_BITS as i32;
        }
    }

    /// Initialize the encoder with a buffer of the given size.
    pub fn enc_init(size: u32) -> Self {
        EcCtx {
            buf: vec![0u8; size as usize],
            storage: size,
            end_offs: 0,
            end_window: 0,
            nend_bits: 0,
            nbits_total: EC_CODE_BITS as i32 + 1,
            offs: 0,
            rng: EC_CODE_TOP,
            rem: -1,
            val: 0,
            ext: 0,
            error: 0,
        }
    }

    /// Encode a symbol with cumulative frequency range [fl, fh) out of ft.
    pub fn encode(&mut self, fl: u32, fh: u32, ft: u32) {
        let r = self.rng / ft;
        if fl > 0 {
            self.val = self.val.wrapping_add(self.rng.wrapping_sub(r.wrapping_mul(ft - fl)));
            self.rng = r.wrapping_mul(fh - fl);
        } else {
            self.rng -= r.wrapping_mul(ft - fh);
        }
        self.enc_normalize();
    }

    /// Binary version of encode for ft = 1 << bits.
    pub fn encode_bin(&mut self, fl: u32, fh: u32, bits: u32) {
        let r = self.rng >> bits;
        if fl > 0 {
            self.val = self.val.wrapping_add(self.rng.wrapping_sub(r.wrapping_mul((1u32 << bits) - fl)));
            self.rng = r.wrapping_mul(fh - fl);
        } else {
            self.rng -= r.wrapping_mul((1u32 << bits) - fh);
        }
        self.enc_normalize();
    }

    /// Encode a bit with probability 1/(1 << logp) of being one.
    pub fn enc_bit_logp(&mut self, val: bool, logp: u32) {
        let r = self.rng;
        let l = self.val;
        let s = r >> logp;
        let r_new = r - s;
        if val {
            self.val = l.wrapping_add(r_new);
        }
        self.rng = if val { s } else { r_new };
        self.enc_normalize();
    }

    /// Encode a symbol using an inverse CDF table (u8).
    pub fn enc_icdf(&mut self, s: usize, icdf: &[u8], ftb: u32) {
        let r = self.rng >> ftb;
        if s > 0 {
            self.val = self.val.wrapping_add(
                self.rng.wrapping_sub(r.wrapping_mul(icdf[s - 1] as u32)),
            );
            self.rng = r.wrapping_mul(icdf[s - 1] as u32 - icdf[s] as u32);
        } else {
            self.rng -= r.wrapping_mul(icdf[s] as u32);
        }
        self.enc_normalize();
    }

    /// Encode a symbol using an inverse CDF table (u16).
    pub fn enc_icdf16(&mut self, s: usize, icdf: &[u16], ftb: u32) {
        let r = self.rng >> ftb;
        if s > 0 {
            self.val = self.val.wrapping_add(
                self.rng.wrapping_sub(r.wrapping_mul(icdf[s - 1] as u32)),
            );
            self.rng = r.wrapping_mul(icdf[s - 1] as u32 - icdf[s] as u32);
        } else {
            self.rng -= r.wrapping_mul(icdf[s] as u32);
        }
        self.enc_normalize();
    }

    /// Encode a uniformly distributed unsigned integer with ft possible values.
    pub fn enc_uint(&mut self, fl: u32, ft: u32) {
        debug_assert!(ft > 1);
        let ft_minus_1 = ft - 1;
        let ftb = ec_ilog(ft_minus_1);
        if ftb > EC_UINT_BITS {
            let ftb_shift = ftb - EC_UINT_BITS;
            let ft_top = (ft_minus_1 >> ftb_shift) + 1;
            let fl_top = fl >> ftb_shift;
            self.encode(fl_top, fl_top + 1, ft_top);
            self.enc_bits(fl & ((1u32 << ftb_shift) - 1), ftb_shift);
        } else {
            self.encode(fl, fl + 1, ft);
        }
    }

    /// Encode raw bits at the end of the stream.
    pub fn enc_bits(&mut self, fl: u32, bits: u32) {
        let mut window = self.end_window;
        let mut used = self.nend_bits;
        debug_assert!(bits > 0);
        if used as u32 + bits > EC_WINDOW_SIZE {
            loop {
                self.error |= self.write_byte_at_end(window & EC_SYM_MAX);
                window >>= EC_SYM_BITS;
                used -= EC_SYM_BITS as i32;
                if (used as u32) < EC_SYM_BITS {
                    break;
                }
            }
        }
        window |= fl << used as u32;
        used += bits as i32;
        self.end_window = window;
        self.nend_bits = used;
        self.nbits_total += bits as i32;
    }

    /// Patch the first few bits of the output after encoding.
    pub fn enc_patch_initial_bits(&mut self, val: u32, nbits: u32) {
        debug_assert!(nbits <= EC_SYM_BITS);
        let shift = EC_SYM_BITS - nbits;
        let mask = ((1u32 << nbits) - 1) << shift;
        if self.offs > 0 {
            self.buf[0] = ((self.buf[0] as u32 & !mask) | (val << shift)) as u8;
        } else if self.rem >= 0 {
            self.rem = ((self.rem as u32 & !mask) | (val << shift)) as i32;
        } else if self.rng <= EC_CODE_TOP >> nbits {
            self.val = (self.val & !(mask << EC_CODE_SHIFT)) | (val << (EC_CODE_SHIFT + shift));
        } else {
            self.error = -1;
        }
    }

    /// Compact the data in the buffer to the target size.
    /// If `size` is smaller than the data already written (`offs + end_offs`),
    /// clamps to the minimum viable size to avoid corruption.
    pub fn enc_shrink(&mut self, size: u32) {
        let min_size = self.offs + self.end_offs;
        let size = size.max(min_size);
        let src_start = (self.storage - self.end_offs) as usize;
        let dst_start = (size - self.end_offs) as usize;
        let len = self.end_offs as usize;
        if src_start != dst_start && len > 0 {
            self.buf.copy_within(src_start..src_start + len, dst_start);
        }
        self.storage = size;
    }

    /// Finalize the encoding and flush all remaining bits.
    pub fn enc_done(&mut self) {
        let l = EC_CODE_BITS as i32 - ec_ilog(self.rng) as i32;
        let msk = (EC_CODE_TOP - 1) >> l as u32;
        let mut end = (self.val.wrapping_add(msk)) & !msk;
        let mut l = l;
        if (end | msk) >= self.val.wrapping_add(self.rng) {
            l += 1;
            let msk = msk >> 1;
            end = (self.val.wrapping_add(msk)) & !msk;
        }
        while l > 0 {
            self.enc_carry_out((end >> EC_CODE_SHIFT) as i32);
            end = (end << EC_SYM_BITS) & (EC_CODE_TOP - 1);
            l -= EC_SYM_BITS as i32;
        }
        if self.rem >= 0 || self.ext > 0 {
            self.enc_carry_out(0);
        }
        let mut window = self.end_window;
        let mut used = self.nend_bits;
        while used >= EC_SYM_BITS as i32 {
            self.error |= self.write_byte_at_end(window & EC_SYM_MAX);
            window >>= EC_SYM_BITS;
            used -= EC_SYM_BITS as i32;
        }
        if self.error == 0 {
            // Clear excess space
            let start = self.offs as usize;
            let end_region = (self.storage - self.end_offs) as usize;
            for i in start..end_region {
                self.buf[i] = 0;
            }
            if used > 0 {
                if self.end_offs >= self.storage {
                    self.error = -1;
                } else {
                    let neg_l = -l;
                    if self.offs + self.end_offs >= self.storage && neg_l < used as i32 {
                        let window = window & ((1u32 << neg_l as u32) - 1);
                        self.error = -1;
                        self.buf[(self.storage - self.end_offs - 1) as usize] |= window as u8;
                    } else {
                        self.buf[(self.storage - self.end_offs - 1) as usize] |= window as u8;
                    }
                }
            }
        }
    }
}
