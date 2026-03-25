use crate::ctx::*;

impl EcCtx {
    /// Read next byte from the front of the input buffer.
    #[inline]
    fn read_byte(&mut self) -> u32 {
        if self.offs < self.storage {
            let b = self.buf[self.offs as usize] as u32;
            self.offs += 1;
            b
        } else {
            0
        }
    }

    /// Read next byte from the end of the input buffer.
    #[inline]
    fn read_byte_from_end(&mut self) -> u32 {
        if self.end_offs < self.storage {
            self.end_offs += 1;
            self.buf[(self.storage - self.end_offs) as usize] as u32
        } else {
            0
        }
    }

    /// Normalizes the contents of val and rng so that rng lies entirely
    /// in the high-order symbol.
    fn dec_normalize(&mut self) {
        while self.rng <= EC_CODE_BOT {
            self.nbits_total += EC_SYM_BITS as i32;
            self.rng <<= EC_SYM_BITS;
            let sym = self.rem as u32;
            self.rem = self.read_byte() as i32;
            let sym = ((sym << EC_SYM_BITS) | self.rem as u32) >> (EC_SYM_BITS - EC_CODE_EXTRA);
            self.val =
                ((self.val << EC_SYM_BITS).wrapping_add(EC_SYM_MAX & !sym)) & (EC_CODE_TOP - 1);
        }
    }

    /// Initialize the decoder from a byte buffer.
    pub fn dec_init(buf: &[u8]) -> Self {
        let storage = buf.len() as u32;
        let mut ctx = EcCtx {
            buf: buf.to_vec(),
            storage,
            end_offs: 0,
            end_window: 0,
            nend_bits: 0,
            nbits_total: EC_CODE_BITS as i32 + 1
                - ((EC_CODE_BITS - EC_CODE_EXTRA) / EC_SYM_BITS * EC_SYM_BITS) as i32,
            offs: 0,
            rng: 1u32 << EC_CODE_EXTRA,
            val: 0,
            ext: 0,
            rem: 0,
            error: 0,
        };
        ctx.rem = ctx.read_byte() as i32;
        ctx.val = ctx.rng - 1 - ((ctx.rem as u32) >> (EC_SYM_BITS - EC_CODE_EXTRA));
        ctx.dec_normalize();
        ctx
    }

    /// Calculates the cumulative frequency for the next symbol.
    /// This is used together with dec_update() to decode a symbol.
    pub fn decode(&mut self, ft: u32) -> u32 {
        self.ext = self.rng / ft;
        let s = self.val / self.ext;
        ft - (s + 1).min(ft)
    }

    /// Binary version of decode for ft = 1 << bits.
    pub fn decode_bin(&mut self, bits: u32) -> u32 {
        self.ext = self.rng >> bits;
        let s = self.val / self.ext;
        (1u32 << bits) - (s + 1).min(1u32 << bits)
    }

    /// Advance the decoder past a symbol with cumulative frequency range [fl, fh) out of ft.
    pub fn dec_update(&mut self, fl: u32, fh: u32, ft: u32) {
        let s = self.ext.wrapping_mul(ft - fh);
        self.val = self.val.wrapping_sub(s);
        self.rng = if fl > 0 {
            self.ext.wrapping_mul(fh - fl)
        } else {
            self.rng.wrapping_sub(s)
        };
        self.dec_normalize();
    }

    /// Decode a bit with probability 1/(1 << logp) of being one.
    pub fn dec_bit_logp(&mut self, logp: u32) -> bool {
        let r = self.rng;
        let d = self.val;
        let s = r >> logp;
        let ret = d < s;
        if !ret {
            self.val = d - s;
        }
        self.rng = if ret { s } else { r - s };
        self.dec_normalize();
        ret
    }

    /// Decode a symbol using an inverse CDF table (u8).
    pub fn dec_icdf(&mut self, icdf: &[u8], ftb: u32) -> usize {
        let s_init = self.rng;
        let d = self.val;
        let r = s_init >> ftb;
        let mut ret: usize = 0;
        let mut t;
        let mut s = s_init;
        loop {
            t = s;
            s = r.wrapping_mul(icdf[ret] as u32);
            ret += 1;
            if d >= s {
                break;
            }
        }
        ret -= 1;
        self.val = d - s;
        self.rng = t - s;
        self.dec_normalize();
        ret
    }

    /// Decode a symbol using an inverse CDF table (u16).
    pub fn dec_icdf16(&mut self, icdf: &[u16], ftb: u32) -> usize {
        let s_init = self.rng;
        let d = self.val;
        let r = s_init >> ftb;
        let mut ret: usize = 0;
        let mut t;
        let mut s = s_init;
        loop {
            t = s;
            s = r.wrapping_mul(icdf[ret] as u32);
            ret += 1;
            if d >= s {
                break;
            }
        }
        ret -= 1;
        self.val = d - s;
        self.rng = t - s;
        self.dec_normalize();
        ret
    }

    /// Decode a uniformly distributed unsigned integer with ft possible values.
    pub fn dec_uint(&mut self, ft: u32) -> u32 {
        debug_assert!(ft > 1);
        let ft_minus_1 = ft - 1;
        let ftb = ec_ilog(ft_minus_1);
        if ftb > EC_UINT_BITS {
            let ftb_shift = ftb - EC_UINT_BITS;
            let ft_top = (ft_minus_1 >> ftb_shift) as u32 + 1;
            let s = self.decode(ft_top);
            self.dec_update(s, s + 1, ft_top);
            let t = ((s as u32) << ftb_shift) | self.dec_bits(ftb_shift);
            if t <= ft_minus_1 {
                t
            } else {
                self.error = 1;
                ft_minus_1
            }
        } else {
            let s = self.decode(ft);
            self.dec_update(s, s + 1, ft);
            s
        }
    }

    /// Extract raw bits from the end of the stream.
    pub fn dec_bits(&mut self, bits: u32) -> u32 {
        let mut window = self.end_window;
        let mut available = self.nend_bits;
        if (available as u32) < bits {
            loop {
                window |= self.read_byte_from_end() << available as u32;
                available += EC_SYM_BITS as i32;
                if available as u32 > EC_WINDOW_SIZE - EC_SYM_BITS {
                    break;
                }
            }
        }
        let ret = window & ((1u32 << bits) - 1);
        window >>= bits;
        available -= bits as i32;
        self.end_window = window;
        self.nend_bits = available;
        self.nbits_total += bits as i32;
        ret
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dec_init_empty() {
        let ctx = EcCtx::dec_init(&[0u8; 4]);
        assert!(!ctx.get_error());
    }
}
