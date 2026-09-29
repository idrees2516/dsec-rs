//! Async frame codec over any `AsyncRead`/`AsyncWrite` pair.
//!
//! Reads are CRC-verified and length-guarded; a frame exceeding
//! [`MAX_PAYLOAD`] aborts the connection, which is the standard defense
//! against hostile or corrupted peers.

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use crate::frame::{Frame, HEADER_LEN};
use crate::{Error, Result, MAX_PAYLOAD};

/// Writes a full frame (encode + flush).
pub async fn write_frame<W: AsyncWrite + Unpin>(w: &mut W, frame: &Frame) -> Result<()> {
    let wire = frame.encode();
    w.write_all(&wire).await?;
    w.flush().await?;
    Ok(())
}

/// Reads exactly one frame; returns `None` on clean EOF at a frame boundary.
pub async fn read_frame<R: AsyncRead + Unpin>(r: &mut R) -> Result<Option<Frame>> {
    let mut header = [0u8; HEADER_LEN];
    let mut filled = 0usize;
    while filled < HEADER_LEN {
        let n = r.read(&mut header[filled..]).await?;
        if n == 0 {
            if filled == 0 {
                return Ok(None); // clean EOF between frames
            }
            return Err(Error::Protocol {
                code: 9,
                message: "EOF mid-header".into(),
            });
        }
        filled += n;
    }
    let payload_len = u32::from_be_bytes([header[17], header[18], header[19], header[20]]) as usize;
    if payload_len > MAX_PAYLOAD {
        return Err(Error::Protocol {
            code: 9,
            message: format!("payload {} exceeds limit {}", payload_len, MAX_PAYLOAD),
        });
    }
    let mut payload = vec![0u8; payload_len];
    if payload_len > 0 {
        r.read_exact(&mut payload).await?;
    }
    let mut wire = header.to_vec();
    wire.extend_from_slice(&payload);
    Ok(Some(Frame::decode(&wire)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::Channel;
    use tokio::io::duplex;

    #[tokio::test]
    async fn duplex_roundtrip() {
        let (mut a, mut b) = duplex(64);
        let f = Frame::request(Channel::Exec, 77, 3, br#"{"op":"ping"}"#.to_vec());
        write_frame(&mut a, &f).await.unwrap();
        let back = read_frame(&mut b).await.unwrap().unwrap();
        assert_eq!(back, f);
    }

    #[tokio::test]
    async fn many_frames_in_order() {
        let (mut a, mut b) = duplex(4096);
        let client = tokio::spawn(async move {
            for i in 0..100u32 {
                let f = Frame::request(
                    Channel::Fs,
                    i as u64,
                    i,
                    format!("{{\"i\":{}}}", i).into_bytes(),
                );
                write_frame(&mut a, &f).await.unwrap();
            }
        });
        for i in 0..100u32 {
            let f = read_frame(&mut b).await.unwrap().unwrap();
            assert_eq!(f.header.sid, i as u64);
            assert_eq!(f.header.req_id, i);
        }
        client.await.unwrap();
    }

    #[tokio::test]
    async fn clean_eof() {
        let (a, mut b) = duplex(64);
        drop(a);
        assert!(read_frame(&mut b).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn oversize_payload_rejected() {
        // Craft a header claiming a huge payload.
        let mut hdr = [0u8; HEADER_LEN];
        hdr[0] = crate::MAGIC;
        hdr[1] = crate::VERSION;
        hdr[17..21].copy_from_slice(&0xFFFF_FFFFu32.to_be_bytes());
        let (mut a, mut b) = duplex(64);
        a.write_all(&hdr).await.unwrap();
        a.flush().await.unwrap();
        assert!(read_frame(&mut b).await.is_err());
    }
}
