//! Message framing and encode/decode — `PROTOCOL.md` §3 and §4.
//!
//! Frame: `VER(1) | TYPE(1) | LENGTH(u16 BE) | PAYLOAD`.
//! Within a payload: fixed-size fields raw, variable fields as `u16 len | bytes`.
//! All integers big-endian.

use std::io::{self, Read, Write};

use crate::error::WireError;

pub const PROTOCOL_VERSION: u8 = 0x01;
pub const MAX_PAYLOAD: usize = 8192;

pub const PROTOCOL_LABEL: &[u8] = b"PQC-VPN-HANDSHAKE-v1";

// -------------------------------------------------------------------------
// message type
// -------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MsgType {
    ClientHello,
    ServerHello,
    ClientFinish,
    ServerFinish,
    RekeyRequest,
    Error,
}

impl MsgType {
    fn to_u8(self) -> u8 {
        match self {
            MsgType::ClientHello => 0x01,
            MsgType::ServerHello => 0x02,
            MsgType::ClientFinish => 0x03,
            MsgType::ServerFinish => 0x04,
            MsgType::RekeyRequest => 0x05,
            MsgType::Error => 0xEE,
        }
    }
    fn from_u8(b: u8) -> Result<Self, WireError> {
        Ok(match b {
            0x01 => MsgType::ClientHello,
            0x02 => MsgType::ServerHello,
            0x03 => MsgType::ClientFinish,
            0x04 => MsgType::ServerFinish,
            0x05 => MsgType::RekeyRequest,
            0xEE => MsgType::Error,
            other => return Err(WireError::UnknownMsgType(other)),
        })
    }
}

// -------------------------------------------------------------------------
// algorithm code (PROTOCOL.md §4.1, matches contracts/algo_registry.json)
// -------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlgoCode {
    MlKem512,
    MlKem768,
    MlKem1024,
}

impl AlgoCode {
    pub fn to_u8(self) -> u8 {
        match self {
            AlgoCode::MlKem512 => 0x00,
            AlgoCode::MlKem768 => 0x01,
            AlgoCode::MlKem1024 => 0x02,
        }
    }
    pub fn from_u8(b: u8) -> Result<Self, WireError> {
        Ok(match b {
            0x00 => AlgoCode::MlKem512,
            0x01 => AlgoCode::MlKem768,
            0x02 => AlgoCode::MlKem1024,
            _ => return Err(WireError::Invalid("algorithm code")),
        })
    }
    /// Name as it appears in the signed transcript — must byte-match the
    /// `algorithm_name` `core::crypto::hybrid_kem::build_handshake_transcript_*`
    /// uses.
    pub fn transcript_name(self) -> &'static [u8] {
        match self {
            AlgoCode::MlKem512 => b"ML-KEM-512",
            AlgoCode::MlKem768 => b"ML-KEM-768",
            AlgoCode::MlKem1024 => b"ML-KEM-1024",
        }
    }
    /// Human-readable name for logging.
    pub fn name(self) -> &'static str {
        match self {
            AlgoCode::MlKem512 => "ML-KEM-512",
            AlgoCode::MlKem768 => "ML-KEM-768",
            AlgoCode::MlKem1024 => "ML-KEM-1024",
        }
    }
    /// Encoded ML-KEM encapsulation-key length (client → server).
    pub fn mlkem_pub_len(self) -> usize {
        match self {
            AlgoCode::MlKem512 => 800,
            AlgoCode::MlKem768 => 1184,
            AlgoCode::MlKem1024 => 1568,
        }
    }
    /// Encoded ML-KEM ciphertext length (server → client).
    pub fn ciphertext_len(self) -> usize {
        match self {
            AlgoCode::MlKem512 => 768,
            AlgoCode::MlKem768 => 1088,
            AlgoCode::MlKem1024 => 1568,
        }
    }
}

/// ML-DSA-65 signature length (fixed, FIPS 204).
pub const SIGNATURE_LEN: usize = 3309;

// -------------------------------------------------------------------------
// buffer helpers
// -------------------------------------------------------------------------

struct Writer(Vec<u8>);

impl Writer {
    fn new() -> Self {
        Writer(Vec::new())
    }
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn bytes(&mut self, v: &[u8]) {
        self.0.extend_from_slice(v);
    }
    /// `u16` big-endian length prefix, then bytes.
    fn lv(&mut self, v: &[u8]) {
        debug_assert!(v.len() <= u16::MAX as usize);
        self.0.extend_from_slice(&(v.len() as u16).to_be_bytes());
        self.0.extend_from_slice(v);
    }
    fn into_vec(self) -> Vec<u8> {
        self.0
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }
    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], WireError> {
        if self.remaining() < n {
            return Err(WireError::Truncated);
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn take_u8(&mut self) -> Result<u8, WireError> {
        Ok(self.take(1)?[0])
    }
    fn take_u16(&mut self) -> Result<u16, WireError> {
        let b = self.take(2)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }
    fn take_arr<const N: usize>(&mut self, what: &'static str) -> Result<[u8; N], WireError> {
        let s = self.take(N).map_err(|_| WireError::Invalid(what))?;
        let mut a = [0u8; N];
        a.copy_from_slice(s);
        Ok(a)
    }
    /// `u16` length prefix, then that many bytes.
    fn take_lv(&mut self) -> Result<&'a [u8], WireError> {
        let n = self.take_u16()? as usize;
        if n > MAX_PAYLOAD {
            return Err(WireError::LengthOutOfRange);
        }
        self.take(n)
    }
    fn expect_end(&self) -> Result<(), WireError> {
        if self.remaining() == 0 {
            Ok(())
        } else {
            Err(WireError::Invalid("trailing bytes"))
        }
    }
}

// -------------------------------------------------------------------------
// messages
// -------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClientHello {
    pub algo: AlgoCode,
    pub client_nonce: [u8; 32],
    pub client_wg_pubkey: [u8; 32],
    pub client_x25519_pub: [u8; 32],
    pub client_mlkem_pub: Vec<u8>,
}

impl ClientHello {
    fn encode_into(&self, w: &mut Writer) {
        w.u8(self.algo.to_u8());
        w.bytes(&self.client_nonce);
        w.bytes(&self.client_wg_pubkey);
        w.lv(&self.client_x25519_pub);
        w.lv(&self.client_mlkem_pub);
    }
    fn decode_from(r: &mut Reader) -> Result<Self, WireError> {
        let algo = AlgoCode::from_u8(r.take_u8()?)?;
        let client_nonce = r.take_arr::<32>("client_nonce")?;
        let client_wg_pubkey = r.take_arr::<32>("client_wg_pubkey")?;
        let x = r.take_lv()?;
        if x.len() != 32 {
            return Err(WireError::Invalid("client_x25519_pub length"));
        }
        let mut client_x25519_pub = [0u8; 32];
        client_x25519_pub.copy_from_slice(x);
        let mk = r.take_lv()?;
        if mk.len() != algo.mlkem_pub_len() {
            return Err(WireError::Invalid("client_mlkem_pub length"));
        }
        Ok(ClientHello {
            algo,
            client_nonce,
            client_wg_pubkey,
            client_x25519_pub,
            client_mlkem_pub: mk.to_vec(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerHello {
    pub algo: AlgoCode,
    pub session_id: [u8; 16],
    pub server_nonce: [u8; 32],
    pub server_x25519_pub: [u8; 32],
    pub mlkem_ciphertext: Vec<u8>,
    pub signature: Vec<u8>,
}

impl ServerHello {
    fn encode_into(&self, w: &mut Writer) {
        w.u8(self.algo.to_u8());
        w.bytes(&self.session_id);
        w.bytes(&self.server_nonce);
        w.lv(&self.server_x25519_pub);
        w.lv(&self.mlkem_ciphertext);
        w.lv(&self.signature);
    }
    fn decode_from(r: &mut Reader) -> Result<Self, WireError> {
        let algo = AlgoCode::from_u8(r.take_u8()?)?;
        let session_id = r.take_arr::<16>("session_id")?;
        let server_nonce = r.take_arr::<32>("server_nonce")?;
        let x = r.take_lv()?;
        if x.len() != 32 {
            return Err(WireError::Invalid("server_x25519_pub length"));
        }
        let mut server_x25519_pub = [0u8; 32];
        server_x25519_pub.copy_from_slice(x);
        let ct = r.take_lv()?;
        if ct.len() != algo.ciphertext_len() {
            return Err(WireError::Invalid("mlkem_ciphertext length"));
        }
        let sig = r.take_lv()?;
        if sig.len() != SIGNATURE_LEN {
            return Err(WireError::Invalid("signature length"));
        }
        Ok(ServerHello {
            algo,
            session_id,
            server_nonce,
            server_x25519_pub,
            mlkem_ciphertext: ct.to_vec(),
            signature: sig.to_vec(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClientFinish {
    pub session_id: [u8; 16],
    pub client_tag: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerFinish {
    pub session_id: [u8; 16],
    pub server_tag: [u8; 32],
    /// WireGuard tunnel parameters the client needs to bring the tunnel up
    /// (PROTOCOL.md §4.5). The PSK itself is not sent — both sides derived it.
    pub server_wg_pubkey: [u8; 32],
    /// IPv4 address the server assigned this peer inside the tunnel.
    pub assigned_ip: [u8; 4],
    /// UDP port the server's WireGuard listens on.
    pub wg_port: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RekeyRequest {
    pub session_id: [u8; 16],
    pub hello: ClientHello,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorMsg {
    pub code: u8,
    pub message: String,
}

// -------------------------------------------------------------------------
// Message enum + frame I/O
// -------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Message {
    ClientHello(ClientHello),
    ServerHello(ServerHello),
    ClientFinish(ClientFinish),
    ServerFinish(ServerFinish),
    RekeyRequest(RekeyRequest),
    Error(ErrorMsg),
}

impl Message {
    fn msg_type(&self) -> MsgType {
        match self {
            Message::ClientHello(_) => MsgType::ClientHello,
            Message::ServerHello(_) => MsgType::ServerHello,
            Message::ClientFinish(_) => MsgType::ClientFinish,
            Message::ServerFinish(_) => MsgType::ServerFinish,
            Message::RekeyRequest(_) => MsgType::RekeyRequest,
            Message::Error(_) => MsgType::Error,
        }
    }

    /// Payload bytes only (no frame header).
    pub fn encode_payload(&self) -> Vec<u8> {
        let mut w = Writer::new();
        match self {
            Message::ClientHello(m) => m.encode_into(&mut w),
            Message::ServerHello(m) => m.encode_into(&mut w),
            Message::ClientFinish(m) => {
                w.bytes(&m.session_id);
                w.bytes(&m.client_tag);
            }
            Message::ServerFinish(m) => {
                w.bytes(&m.session_id);
                w.bytes(&m.server_tag);
                w.bytes(&m.server_wg_pubkey);
                w.bytes(&m.assigned_ip);
                w.bytes(&m.wg_port.to_be_bytes());
            }
            Message::RekeyRequest(m) => {
                w.bytes(&m.session_id);
                m.hello.encode_into(&mut w);
            }
            Message::Error(m) => {
                w.u8(m.code);
                w.lv(m.message.as_bytes());
            }
        }
        w.into_vec()
    }

    fn decode_payload(ty: MsgType, payload: &[u8]) -> Result<Self, WireError> {
        let mut r = Reader::new(payload);
        let msg = match ty {
            MsgType::ClientHello => Message::ClientHello(ClientHello::decode_from(&mut r)?),
            MsgType::ServerHello => Message::ServerHello(ServerHello::decode_from(&mut r)?),
            MsgType::ClientFinish => Message::ClientFinish(ClientFinish {
                session_id: r.take_arr::<16>("session_id")?,
                client_tag: r.take_arr::<32>("client_tag")?,
            }),
            MsgType::ServerFinish => Message::ServerFinish(ServerFinish {
                session_id: r.take_arr::<16>("session_id")?,
                server_tag: r.take_arr::<32>("server_tag")?,
                server_wg_pubkey: r.take_arr::<32>("server_wg_pubkey")?,
                assigned_ip: r.take_arr::<4>("assigned_ip")?,
                wg_port: r.take_u16()?,
            }),
            MsgType::RekeyRequest => {
                let session_id = r.take_arr::<16>("session_id")?;
                let hello = ClientHello::decode_from(&mut r)?;
                Message::RekeyRequest(RekeyRequest { session_id, hello })
            }
            MsgType::Error => {
                let code = r.take_u8()?;
                let message = String::from_utf8_lossy(r.take_lv()?).into_owned();
                Message::Error(ErrorMsg { code, message })
            }
        };
        r.expect_end()?;
        Ok(msg)
    }

    /// Write a full frame (`VER | TYPE | LEN | PAYLOAD`).
    pub fn write<W: Write>(&self, w: &mut W) -> io::Result<()> {
        let payload = self.encode_payload();
        if payload.len() > MAX_PAYLOAD {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "payload exceeds MAX_PAYLOAD",
            ));
        }
        let mut frame = Vec::with_capacity(4 + payload.len());
        frame.push(PROTOCOL_VERSION);
        frame.push(self.msg_type().to_u8());
        frame.extend_from_slice(&(payload.len() as u16).to_be_bytes());
        frame.extend_from_slice(&payload);
        w.write_all(&frame)
    }

    /// Read one full frame. `io::Error` for transport problems, `WireError`
    /// (wrapped) for malformed content.
    pub fn read<R: Read>(r: &mut R) -> Result<Message, FrameError> {
        let mut header = [0u8; 4];
        r.read_exact(&mut header)?;
        if header[0] != PROTOCOL_VERSION {
            return Err(WireError::UnsupportedVersion(header[0]).into());
        }
        let ty = MsgType::from_u8(header[1])?;
        let len = u16::from_be_bytes([header[2], header[3]]) as usize;
        if len > MAX_PAYLOAD {
            return Err(WireError::LengthOutOfRange.into());
        }
        let mut payload = vec![0u8; len];
        r.read_exact(&mut payload)?;
        Ok(Message::decode_payload(ty, &payload)?)
    }
}

/// Either a transport error or a malformed-message error.
#[derive(Debug)]
pub enum FrameError {
    Io(io::Error),
    Wire(WireError),
}

impl std::fmt::Display for FrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FrameError::Io(e) => write!(f, "transport error: {e}"),
            FrameError::Wire(e) => write!(f, "{e}"),
        }
    }
}
impl std::error::Error for FrameError {}
impl From<io::Error> for FrameError {
    fn from(e: io::Error) -> Self {
        FrameError::Io(e)
    }
}
impl From<WireError> for FrameError {
    fn from(e: WireError) -> Self {
        FrameError::Wire(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_client_hello() -> ClientHello {
        ClientHello {
            algo: AlgoCode::MlKem768,
            client_nonce: [1u8; 32],
            client_wg_pubkey: [2u8; 32],
            client_x25519_pub: [3u8; 32],
            client_mlkem_pub: vec![4u8; 1184],
        }
    }

    #[test]
    fn client_hello_round_trips() {
        let m = Message::ClientHello(sample_client_hello());
        let mut buf = Vec::new();
        m.write(&mut buf).unwrap();
        // header + 1 + 32 + 32 + (2+32) + (2+1184)
        assert_eq!(buf.len(), 4 + 1 + 32 + 32 + 34 + 1186);
        assert_eq!(buf[0], PROTOCOL_VERSION);
        assert_eq!(buf[1], 0x01);
        let back = Message::read(&mut buf.as_slice()).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn server_hello_round_trips() {
        let m = Message::ServerHello(ServerHello {
            algo: AlgoCode::MlKem1024,
            session_id: [9u8; 16],
            server_nonce: [8u8; 32],
            server_x25519_pub: [7u8; 32],
            mlkem_ciphertext: vec![6u8; 1568],
            signature: vec![5u8; SIGNATURE_LEN],
        });
        let mut buf = Vec::new();
        m.write(&mut buf).unwrap();
        let back = Message::read(&mut buf.as_slice()).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn finish_and_rekey_round_trip() {
        for m in [
            Message::ClientFinish(ClientFinish {
                session_id: [1u8; 16],
                client_tag: [2u8; 32],
            }),
            Message::ServerFinish(ServerFinish {
                session_id: [3u8; 16],
                server_tag: [4u8; 32],
                server_wg_pubkey: [5u8; 32],
                assigned_ip: [10, 8, 0, 2],
                wg_port: 51820,
            }),
            Message::RekeyRequest(RekeyRequest {
                session_id: [5u8; 16],
                hello: sample_client_hello(),
            }),
            Message::Error(ErrorMsg {
                code: 0x05,
                message: "rekey algorithm mismatch".into(),
            }),
        ] {
            let mut buf = Vec::new();
            m.write(&mut buf).unwrap();
            assert_eq!(Message::read(&mut buf.as_slice()).unwrap(), m);
        }
    }

    #[test]
    fn rejects_bad_version() {
        let bytes = [0x02u8, 0x01, 0x00, 0x00];
        match Message::read(&mut bytes.as_slice()) {
            Err(FrameError::Wire(WireError::UnsupportedVersion(0x02))) => {}
            other => panic!("expected UnsupportedVersion, got {other:?}"),
        }
    }

    #[test]
    fn rejects_unknown_type() {
        let bytes = [PROTOCOL_VERSION, 0x77, 0x00, 0x00];
        match Message::read(&mut bytes.as_slice()) {
            Err(FrameError::Wire(WireError::UnknownMsgType(0x77))) => {}
            other => panic!("expected UnknownMsgType, got {other:?}"),
        }
    }

    #[test]
    fn rejects_wrong_mlkem_pub_length() {
        let mut ch = sample_client_hello();
        ch.client_mlkem_pub = vec![0u8; 800]; // 512-sized body but algo says 768
        let mut buf = Vec::new();
        Message::ClientHello(ch).write(&mut buf).unwrap();
        assert!(matches!(
            Message::read(&mut buf.as_slice()),
            Err(FrameError::Wire(WireError::Invalid(_)))
        ));
    }

    #[test]
    fn algo_code_ordering_supports_no_downgrade_check() {
        assert!(AlgoCode::MlKem512 < AlgoCode::MlKem768);
        assert!(AlgoCode::MlKem768 < AlgoCode::MlKem1024);
    }
}
