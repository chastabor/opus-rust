use crate::ctx::EcCtx;

const LAPLACE_LOG_MINP: u32 = 0;
const LAPLACE_MINP: u32 = 1 << LAPLACE_LOG_MINP;
const LAPLACE_NMIN: u32 = 16;

/// Compute frequency for magnitude 1 given frequency for 0 and decay.
fn ec_laplace_get_freq1(fs0: u32, decay: i32) -> u32 {
    let ft = 32768u32 - LAPLACE_MINP * (2 * LAPLACE_NMIN) - fs0;
    ((ft as i64 * (16384 - decay) as i64) >> 15) as u32
}

impl EcCtx {
    /// Encode a Laplace-distributed value.
    /// fs is the probability of 0 (out of 32768), decay controls geometric falloff.
    /// May clamp *value to the representable range.
    pub fn laplace_encode(&mut self, value: &mut i32, fs_init: u32, decay: i32) {
        let val = *value;
        let mut fl: u32 = 0;
        let mut fs = fs_init;

        if val != 0 {
            let s: i32 = if val < 0 { -1 } else { 0 };
            let val_abs = (val + s) ^ s;
            fl = fs;
            fs = ec_laplace_get_freq1(fs, decay);

            let mut i = 1i32;
            while fs > 0 && i < val_abs {
                fs *= 2;
                fl += fs + 2 * LAPLACE_MINP;
                fs = ((fs as i64 * decay as i64) >> 15) as u32;
                i += 1;
            }

            if fs == 0 {
                let ndi_max = ((32768u32.wrapping_sub(fl) + LAPLACE_MINP - 1) >> LAPLACE_LOG_MINP) as i32;
                let ndi_max = (ndi_max - s) >> 1;
                let di = (val_abs - i).min(ndi_max - 1);
                fl += ((2 * di + 1 + s) as u32).wrapping_mul(LAPLACE_MINP);
                fs = LAPLACE_MINP.min(32768u32.wrapping_sub(fl));
                *value = (i + di + s) ^ s;
            } else {
                fs += LAPLACE_MINP;
                // C: fl += fs & ~s
                // s=0 (positive): ~s = 0xFFFFFFFF => fl += fs
                // s=-1 (negative): ~s = 0 => fl += 0
                fl += fs & ((!s) as u32);
            }
        }
        self.encode_bin(fl, fl + fs, 15);
    }

    /// Decode a Laplace-distributed value.
    pub fn laplace_decode(&mut self, fs: u32, decay: i32) -> i32 {
        let mut val: i32 = 0;
        let fm = self.decode_bin(15);
        let mut fl: u32 = 0;
        let mut fs = fs;
        if fm >= fs {
            val = 1;
            fl = fs;
            fs = ec_laplace_get_freq1(fl, decay) + LAPLACE_MINP;
            while fs > LAPLACE_MINP && fm >= fl + 2 * fs {
                fs *= 2;
                fl += fs;
                fs = (((fs - 2 * LAPLACE_MINP) as i64 * decay as i64) >> 15) as u32;
                fs += LAPLACE_MINP;
                val += 1;
            }
            if fs <= LAPLACE_MINP {
                let di = ((fm - fl) >> (LAPLACE_LOG_MINP + 1)) as i32;
                val += di;
                fl += 2 * di as u32 * LAPLACE_MINP;
            }
            if fm < fl + fs {
                val = -val;
            } else {
                fl += fs;
            }
        }
        let fs_clamped = (fl + fs).min(32768);
        self.dec_update(fl, fs_clamped, 32768);
        val
    }

    /// Encode a Laplace-distributed value with p0 parameter.
    pub fn laplace_encode_p0(&mut self, value: i32, p0: u16, decay: u16) {
        let mut sign_icdf = [0u16; 3];
        sign_icdf[0] = 32768 - p0;
        sign_icdf[1] = sign_icdf[0] / 2;
        sign_icdf[2] = 0;
        let s = if value == 0 {
            0
        } else if value > 0 {
            1
        } else {
            2
        };
        self.enc_icdf16(s, &sign_icdf, 15);
        let abs_value = value.unsigned_abs() as i32;
        if abs_value != 0 {
            let mut icdf = [0u16; 8];
            icdf[0] = 7u16.max(decay);
            for i in 1..7 {
                icdf[i] = ((7 - i) as u16).max(((icdf[i - 1] as u32 * decay as u32) >> 15) as u16);
            }
            icdf[7] = 0;
            let mut v = abs_value - 1;
            loop {
                self.enc_icdf16(v.min(7) as usize, &icdf, 15);
                v -= 7;
                if v < 0 {
                    break;
                }
            }
        }
    }

    /// Decode a Laplace-distributed value with p0 parameter.
    pub fn laplace_decode_p0(&mut self, p0: u16, decay: u16) -> i32 {
        let mut sign_icdf = [0u16; 3];
        sign_icdf[0] = 32768 - p0;
        sign_icdf[1] = sign_icdf[0] / 2;
        sign_icdf[2] = 0;
        let mut s = self.dec_icdf16(&sign_icdf, 15) as i32;
        if s == 2 {
            s = -1;
        }
        if s != 0 {
            let mut icdf = [0u16; 8];
            icdf[0] = 7u16.max(decay);
            for i in 1..7 {
                icdf[i] = ((7 - i) as u16).max(((icdf[i - 1] as u32 * decay as u32) >> 15) as u16);
            }
            icdf[7] = 0;
            let mut value: i32 = 1;
            loop {
                let v = self.dec_icdf16(&icdf, 15) as i32;
                value += v;
                if v != 7 {
                    break;
                }
            }
            s * value
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_laplace_p0_roundtrip() {
        let p0: u16 = 16000;
        let decay: u16 = 16000;
        let values = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

        let mut enc = EcCtx::enc_init(10000);
        for &v in &values {
            enc.laplace_encode_p0(v, p0, decay);
        }
        enc.enc_done();

        let buf = enc.buf[..enc.range_bytes() as usize].to_vec();
        let mut dec = EcCtx::dec_init(&buf);
        let mut decoded = Vec::new();
        for _ in 0..values.len() {
            decoded.push(dec.laplace_decode_p0(p0, decay));
        }
        assert_eq!(&values[..], &decoded[..]);
    }

    #[test]
    fn test_laplace_p0_negative_values() {
        let p0: u16 = 16000;
        let decay: u16 = 16000;
        let values = [-3, -1, 0, 1, 3];

        let mut enc = EcCtx::enc_init(10000);
        for &v in &values {
            enc.laplace_encode_p0(v, p0, decay);
        }
        enc.enc_done();

        let buf = enc.buf[..enc.range_bytes() as usize].to_vec();
        let mut dec = EcCtx::dec_init(&buf);
        let mut decoded = Vec::new();
        for _ in 0..values.len() {
            decoded.push(dec.laplace_decode_p0(p0, decay));
        }
        assert_eq!(&values[..], &decoded[..]);
    }

    #[test]
    fn test_laplace_roundtrip() {
        let fs = 15000u32;
        let decay = 11000;
        let test_values: Vec<i32> = vec![0, 1, -1, 2, -2, 5, -5];

        for &original in &test_values {
            let mut enc = EcCtx::enc_init(1000);
            let mut val = original;
            enc.laplace_encode(&mut val, fs, decay);
            enc.enc_done();

            let buf = enc.buf[..enc.range_bytes() as usize].to_vec();
            let mut dec = EcCtx::dec_init(&buf);
            let decoded = dec.laplace_decode(fs, decay);
            assert_eq!(val, decoded, "Failed for original value {original}");
        }
    }
}
