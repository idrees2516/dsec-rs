//! Binary framing for the Aether data plane.
//!
//! Wire format (big-endian, 25-byte header + payload):
//!
//! ```text
//! offset  size  field
//! 0       1     magic      0xD5
//! 1       1     version    1
//! 2       1     flags      bit0 REQUEST, bit1 RESPONSE, bit2 STREAM_DATA,
//!                         bit3 STREAM_FIN, bit4 ERROR
//! 3       2     channel    1=control 2=exec 3=fs 4=http 5=stream
//! 5       8     sandbox id
//! 13      4     request id (correlation key)
//! 17      4     payload length
//! 21      4     CRC32 over header[0..21] ++ payload
//! 25      ...   payload (JSON-encoded message)
//! ```

use std::sync::OnceLock;

use crate::{MAGIC, VERSION};

pub const FLAG_REQUEST: u8 = 1 << 0;
pub const FLAG_RESPONSE: u8 = 1 << 1;
pub const FLAG_STREAM_DATA: u8 = 1 << 2;
pub const FLAG_STREAM_FIN: u8 = 1 << 3;
pub const FLAG_ERROR: u8 = 1 << 4;

pub const HEADER_LEN: usize = 25;

/// Logical channel multiplexed over one Aether connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum Channel {
    Control = 1,
    Exec = 2,
    Fs = 3,
    Http = 4,
    Stream = 5,
}

impl Channel {
    pub fn as_u16(self) -> u16 {
        self as u16
    }
    pub fn from_u16(v: u16) -> Option<Self> {
        match v {
            1 => Some(Channel::Control),
            2 => Some(Channel::Exec),
            3 => Some(Channel::Fs),
            4 => Some(Channel::Http),
            5 => Some(Channel::Stream),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameHeader {
    pub flags: u8,
    pub channel: u16,
    pub sid: u64,
    pub req_id: u32,
    pub payload_len: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    pub header: FrameHeader,
    pub payload: Vec<u8>,
}

impl Frame {
    /// Builds a request frame carrying a JSON payload.
    pub fn request(channel: Channel, sid: u64, req_id: u32, payload: Vec<u8>) -> Self {
        Frame {
            header: FrameHeader {
                flags: FLAG_REQUEST,
                channel: channel.as_u16(),
                sid,
                req_id,
                payload_len: payload.len() as u32,
            },
            payload,
        }
    }

    /// Builds a response frame.
    pub fn response(channel: Channel, sid: u64, req_id: u32, payload: Vec<u8>) -> Self {
        Frame {
            header: FrameHeader {
                flags: FLAG_RESPONSE,
                channel: channel.as_u16(),
                sid,
                req_id,
                payload_len: payload.len() as u32,
            },
            payload,
        }
    }

    /// Builds a server-initiated stream data frame.
    pub fn stream_data(sid: u64, stream_id: u32, payload: Vec<u8>) -> Self {
        Frame {
            header: FrameHeader {
                flags: FLAG_STREAM_DATA,
                channel: Channel::Stream.as_u16(),
                sid,
                req_id: stream_id,
                payload_len: payload.len() as u32,
            },
            payload,
        }
    }

    /// Builds a stream termination frame.
    pub fn stream_fin(sid: u64, stream_id: u32, payload: Vec<u8>) -> Self {
        Frame {
            header: FrameHeader {
                flags: FLAG_STREAM_DATA | FLAG_STREAM_FIN,
                channel: Channel::Stream.as_u16(),
                sid,
                req_id: stream_id,
                payload_len: payload.len() as u32,
            },
            payload,
        }
    }

    pub fn is_request(&self) -> bool {
        self.header.flags & FLAG_REQUEST != 0
    }
    pub fn is_response(&self) -> bool {
        self.header.flags & FLAG_RESPONSE != 0
    }
    pub fn is_stream(&self) -> bool {
        self.header.flags & FLAG_STREAM_DATA != 0
    }

    /// Serializes header bytes (CRC field left zero) for CRC computation.
    fn header_bytes(&self) -> [u8; HEADER_LEN] {
        let mut b = [0u8; HEADER_LEN];
        b[0] = MAGIC;
        b[1] = VERSION;
        b[2] = self.header.flags;
        b[3..5].copy_from_slice(&self.header.channel.to_be_bytes());
        b[5..13].copy_from_slice(&self.header.sid.to_be_bytes());
        b[13..17].copy_from_slice(&self.header.req_id.to_be_bytes());
        b[17..21].copy_from_slice(&self.header.payload_len.to_be_bytes());
        // b[21..25] = crc, filled by encode()
        b
    }

    /// Full wire encoding, header + payload, with CRC trailer.
    pub fn encode(&self) -> Vec<u8> {
        let hdr = self.header_bytes();
        let mut out = Vec::with_capacity(HEADER_LEN + self.payload.len());
        out.extend_from_slice(&hdr);
        out.extend_from_slice(&self.payload);
        let crc = CRC32::compute(&out);
        out[21..25].copy_from_slice(&crc.to_be_bytes());
        out
    }

    /// Decodes a full wire buffer into a frame, verifying magic, version and CRC.
    pub fn decode(buf: &[u8]) -> crate::Result<Self> {
        if buf.len() < HEADER_LEN {
            return Err(crate::Error::Protocol {
                code: crate::message::ErrorCode::Internal as u32,
                message: format!("short frame: {} bytes", buf.len()),
            });
        }
        if buf[0] != MAGIC {
            return Err(crate::Error::Protocol {
                code: crate::message::ErrorCode::Internal as u32,
                message: format!("bad magic 0x{:02x}", buf[0]),
            });
        }
        if buf[1] != VERSION {
            return Err(crate::Error::Protocol {
                code: crate::message::ErrorCode::Internal as u32,
                message: format!("unsupported version {}", buf[1]),
            });
        }
        let payload_len = u32::from_be_bytes([buf[17], buf[18], buf[19], buf[20]]) as usize;
        if buf.len() != HEADER_LEN + payload_len {
            return Err(crate::Error::Protocol {
                code: crate::message::ErrorCode::Internal as u32,
                message: format!(
                    "length mismatch: {} != {} + {}",
                    buf.len(),
                    HEADER_LEN,
                    payload_len
                ),
            });
        }
        let crc = u32::from_be_bytes([buf[21], buf[22], buf[23], buf[24]]);
        let mut crc_input = buf.to_vec();
        crc_input[21..25].copy_from_slice(&0u32.to_be_bytes());
        if CRC32::compute(&crc_input) != crc {
            return Err(crate::Error::Protocol {
                code: crate::message::ErrorCode::Internal as u32,
                message: "CRC32 mismatch".into(),
            });
        }
        Ok(Frame {
            header: FrameHeader {
                flags: buf[2],
                channel: u16::from_be_bytes([buf[3], buf[4]]),
                sid: u64::from_be_bytes([
                    buf[5], buf[6], buf[7], buf[8], buf[9], buf[10], buf[11], buf[12],
                ]),
                req_id: u32::from_be_bytes([buf[13], buf[14], buf[15], buf[16]]),
                payload_len: payload_len as u32,
            },
            payload: buf[HEADER_LEN..].to_vec(),
        })
    }
}

/// CRC-32 (IEEE 802.3, reflected polynomial 0xEDB88320).
pub struct CRC32;

impl CRC32 {
    fn table() -> &'static [u32; 256] {
        static TABLE: OnceLock<[u32; 256]> = OnceLock::new();
        TABLE.get_or_init(|| {
            let mut t = [0u32; 256];
            for i in 0..256u32 {
                let mut c = i;
                for _ in 0..8 {
                    c = if c & 1 != 0 {
                        0xEDB88320 ^ (c >> 1)
                    } else {
                        c >> 1
                    };
                }
                t[i as usize] = c;
            }
            t
        })
    }

    pub fn compute(data: &[u8]) -> u32 {
        let table = Self::table();
        let mut crc = 0xFFFF_FFFFu32;
        for &b in data {
            crc = table[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
        }
        crc ^ 0xFFFF_FFFF
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_known_vectors() {
        assert_eq!(CRC32::compute(b"123456789"), 0xCBF4_3926);
        assert_eq!(CRC32::compute(b""), 0x0000_0000);
        assert_eq!(
            CRC32::compute(b"The quick brown fox jumps over the lazy dog"),
            0x414F_A339
        );
    }

    #[test]
    fn frame_roundtrip() {
        let payload = br#"{"a":1}"#.to_vec();
        let f = Frame::request(Channel::Exec, 42, 7, payload.clone());
        let wire = f.encode();
        assert_eq!(wire.len(), HEADER_LEN + payload.len());
        let back = Frame::decode(&wire).unwrap();
        assert_eq!(back, f);
        assert!(back.is_request());
        assert_eq!(back.header.channel, Channel::Exec.as_u16());
    }

    #[test]
    fn frame_rejects_corruption() {
        let f = Frame::request(Channel::Fs, 1, 1, b"hello".to_vec());
        let mut wire = f.encode();
        let last = wire.len() - 1;
        wire[last] ^= 0xFF;
        assert!(Frame::decode(&wire).is_err());
        let mut bad_magic = f.encode();
        bad_magic[0] = 0x00;
        assert!(Frame::decode(&bad_magic).is_err());
    }

    #[test]
    fn stream_frames() {
        let f = Frame::stream_data(9, 3, b"chunk".to_vec());
        assert!(f.is_stream() && !f.is_request());
        let fin = Frame::stream_fin(9, 3, Vec::new());
        assert!(fin.is_stream());
        assert_eq!(fin.header.flags & FLAG_STREAM_FIN, FLAG_STREAM_FIN);
    }
}
