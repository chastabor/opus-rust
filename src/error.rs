use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OpusError {
    BadArg = -1,
    BufferTooSmall = -2,
    InternalError = -3,
    InvalidPacket = -4,
    Unimplemented = -5,
    InvalidState = -6,
    AllocFail = -7,
}

impl fmt::Display for OpusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpusError::BadArg => write!(f, "invalid argument"),
            OpusError::BufferTooSmall => write!(f, "buffer too small"),
            OpusError::InternalError => write!(f, "internal error"),
            OpusError::InvalidPacket => write!(f, "invalid/corrupted packet"),
            OpusError::Unimplemented => write!(f, "unimplemented"),
            OpusError::InvalidState => write!(f, "invalid state"),
            OpusError::AllocFail => write!(f, "allocation failure"),
        }
    }
}

impl std::error::Error for OpusError {}

impl From<i32> for OpusError {
    fn from(code: i32) -> Self {
        match code {
            -1 => OpusError::BadArg,
            -2 => OpusError::BufferTooSmall,
            -3 => OpusError::InternalError,
            -4 => OpusError::InvalidPacket,
            -5 => OpusError::Unimplemented,
            -6 => OpusError::InvalidState,
            -7 => OpusError::AllocFail,
            _ => OpusError::InternalError,
        }
    }
}
