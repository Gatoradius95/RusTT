//! Particle template list (`.PTL`) parsing — the per-level particle/emitter
//! definition table that drives ambient dust, fire, smoke, and other
//! environmental particles in the Nu2 engine.
//!
//! Layout was recovered from the level files themselves (all the shipped
//! level `.PTL`s are version 40) plus the PC executable's field-read order in
//! `FUN_0066a190` (`LEGOStarWarsSaga.exe`). Note that the executable's
//! `NuPTLLoadFromFile` reads a *u16* version/count header and reads only the
//! first 983 bytes of each 989-byte record — the level files use a u32
//! header and carry a 6-byte zero tail per record, so the file format
//! (documented below) is what matters for parsing.
//!
//! ## File layout (version 40)
//!
//! Records are 989 bytes: 983 bytes of fields + 6 zero pad. The file is
//! dense in the executable's *read order* — the loader stores fields at
//! struct offsets that do not equal their file positions (e.g. the head
//! u16s land at struct `0x16`/`0x14`), so offsets below are file-relative
//! and the field names mirror the loader's struct.
//!
//! ```text
//! u32 version (40)
//! u32 record count N
//! N × records, 989 bytes each (= 983 fields + 6 pad):
//!    0x000  name[16]
//!    0x010  u16            (loader struct 0x16 — "frames")
//!    0x012  u16            (loader struct 0x14 — "rate")
//!    0x014  f32 ×5   base scale / gravity group
//!    0x028  u8 ×5    flags/types (struct 0x2c, 0x2d, 0x12, 0x2e, 0x412)
//!    0x02d  f32 ×6   (struct 0x30..0x44)
//!    0x045  f32 ×4   (struct 0x48..0x54)
//!    0x055  f32 ×3   (struct 0x58)
//!    0x061  f32 ×3   (struct 0x64)
//!    0x06d  f32 ×14  (struct 0x70..0xa4)
//!    0x0a5  u16; u8; u8
//!    0x0a9  f32 ×5
//!    0x0bd  8 × (f32 + 4×u8)   spawn/emitter slots
//!    0x13d  8 × 2×f32
//!    0x14d  4 × f32
//!    0x18d  8 × 2×f32
//!    0x1cd  8 × 2×f32
//!    0x1d5  2 × f32
//!    0x215  8 × 2×f32
//!    0x255  8 × 2×f32
//!    0x295  8 × 2×f32
//!    0x2a5  4 × f32
//!    0x2e5  8 × 2×f32
//!    0x2e9  u8 ×4
//!    0x2f5  3 × f32; f32; 3 × f32
//!    0x305  3 × (8 × 2×f32)
//!    0x3c5  u32 spawn-list count (0 in level files)
//!    0x3c9  u8; f32; u8; f32; f32   tail (struct 0x410/0x414/0x411/0x41c/0x418)
//!    0x3d7  u8[6] pad
//! u32 section-2 count M
//! M × 77-byte emitter records:
//!    0x00  f32 ×3
//!    0x0c  u16 ×4
//!    0x14  u16
//!    0x16  f32
//!    0x1a  name[16]
//!    0x2a  u16, u16, f32
//!    0x34  u16, u16, f32
//!    0x3e  f32
//!    0x42  u16; 0x44 u16
//!    0x46  u8 ×3
//!    0x49  u16, u16
//!    0x4d  u8[4] reserved
//! ```

use std::fmt;

/// Errors produced while parsing a `.PTL` file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PtlError {
    /// Not a particle template file.
    NotPtl,
    /// Version is not 40 (only the level-file version is decoded).
    UnsupportedVersion(u32),
    /// Data ended before the declared structure was consumed.
    Truncated,
    /// Bytes remain after the declared structure.
    Trailing,
}

impl fmt::Display for PtlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PtlError::NotPtl => write!(f, "not a PTL file"),
            PtlError::UnsupportedVersion(v) => {
                write!(f, "unsupported PTL version {v} (only 40 is decoded)")
            }
            PtlError::Truncated => write!(f, "truncated PTL data"),
            PtlError::Trailing => write!(f, "trailing bytes after PTL structure"),
        }
    }
}

impl std::error::Error for PtlError {}

/// A parsed `.PTL` file: the particle table plus the trailing emitter table.
#[derive(Debug, Clone)]
pub struct PtlFile {
    pub version: u32,
    pub particles: Vec<PtlParticle>,
    pub emitters: Vec<PtlEmitter>,
}

/// One 989-byte particle template record.
///
/// Field names carry the file offset where the semantics are not yet known.
#[derive(Debug, Clone)]
pub struct PtlParticle {
    pub name: String,
    /// File `0x10` — base frame rate.
    pub frames: u16,
    /// File `0x12`.
    pub rate: u16,
    /// File `0x18..0x2c` — base scale / gravity group.
    pub scale: [f32; 5],
    /// File `0x2c..0x30` — flags/types (`0x2e == 7` forces `0x2f1 = 2`).
    pub flags: [u8; 4],
    /// File `0x30`.
    pub type_byte: u8,
    /// File `0x31..0x59` (struct `0x30..0x54`).
    pub f10: [f32; 10],
    /// File `0x59..0x65` (struct `0x58`).
    pub f3: [f32; 3],
    /// File `0x65..0x71` (struct `0x64`).
    pub f3b: [f32; 3],
    /// File `0x71..0xa9` (struct `0x70`).
    pub f14: [f32; 14],
    /// File `0xa9..0xad` — u16 + 2×u8.
    pub u_a8: u16,
    pub u_aa: u8,
    pub u_ab: u8,
    /// File `0xad..0xc1`.
    pub f5: [f32; 5],
    /// File `0xc1..0x101` — 8 emitter/spawn slots; the f32 of each slot.
    pub slots_c0: Vec<f32>,
    /// File `0x101..0x141` — 8 vec2 pairs.
    pub f100: [[f32; 2]; 8],
    /// File `0x141..0x149`.
    pub f140: [f32; 2],
    /// File `0x149..0x151`.
    pub f148: [f32; 2],
    /// File `0x151..0x191`.
    pub f150: [[f32; 2]; 8],
    /// File `0x191..0x1d1`.
    pub f190: [[f32; 2]; 8],
    /// File `0x1d1..0x1d9`.
    pub f1d0: [f32; 2],
    /// File `0x1d9..0x219`.
    pub f1d8: [[f32; 2]; 8],
    /// File `0x219..0x259`.
    pub f218: [[f32; 2]; 8],
    /// File `0x259..0x299`.
    pub f258: [[f32; 2]; 8],
    /// File `0x299..0x2a9`.
    pub f298: [f32; 4],
    /// File `0x2b1..0x2f1` — 8 vec2 pairs.
    pub f2b0: [[f32; 2]; 8],
    /// File `0x2f1..0x2f5` — struct `0x2f0..0x2f3`.
    pub u2f0: [u8; 4],
    /// File `0x2f9..0x301` (struct `0x2f8`).
    pub f2f8: [f32; 3],
    /// File `0x301..0x305` (struct `0x2f4`).
    pub f2f4: f32,
    /// File `0x305..0x311` (struct `0x304`).
    pub f304: [f32; 3],
    /// File `0x311..0x3c5` — three 8×vec2 groups (struct `0x310`, `0x350`, `0x390`).
    pub f310: [[f32; 2]; 8],
    pub f350: [[f32; 2]; 8],
    pub f390: [[f32; 2]; 8],
    /// File `0x3e0` — sub-list of referenced particle names.
    pub spawn_count: u32,
    pub spawns: Vec<PtlSpawnRef>,
    /// Tail fields (struct `0x410`, `0x411`).
    pub tail: [u8; 2],
    /// Tail fields (struct `0x414`, `0x41c`, `0x418`).
    pub tail_f32: [f32; 3],
}

/// One sub-list entry (`name[16] + u16 + u16`).
#[derive(Debug, Clone)]
pub struct PtlSpawnRef {
    pub name: String,
    pub a: u32,
    pub b: u32,
}

/// One 77-byte section-2 emitter record.
#[derive(Debug, Clone)]
pub struct PtlEmitter {
    pub pos: [f32; 3],
    pub u12: [u16; 4],
    pub u20: u16,
    pub f22: f32,
    pub name: String,
    pub u42: u16,
    pub u44: u16,
    pub f46: f32,
    pub u50: u16,
    pub u52: u16,
    pub f54: f32,
    pub f58: f32,
    pub u62: u16,
    pub u64: u16,
    pub u66: [u8; 3],
    pub u69: u16,
    pub u71: u16,
    pub reserved: [u8; 4],
}

const RECORD_LEN: usize = 989;

/// Parse a version-40 level `.PTL` file.
pub fn parse(data: &[u8]) -> Result<PtlFile, PtlError> {
    if data.len() < 8 {
        return Err(PtlError::NotPtl);
    }
    let version = u32::from_le_bytes(data[0..4].try_into().unwrap());
    if version != 40 {
        return Err(PtlError::UnsupportedVersion(version));
    }
    let count = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;

    let mut r = Reader {
        data,
        pos: 8,
        end: data.len(),
    };

    let mut particles = Vec::with_capacity(count);
    for _ in 0..count {
        let record_start = r.pos;
        particles.push(parse_particle(&mut r)?);
        // Level records are exactly 989 bytes; align on the boundary
        // regardless of how many bytes the field reads consumed.
        r.pos = record_start + RECORD_LEN;
    }

    // Section 2: u32 emitter count + 77-byte records.
    let n2 = r.u32()? as usize;
    let mut emitters = Vec::with_capacity(n2);
    for _ in 0..n2 {
        emitters.push(parse_emitter(&mut r)?);
    }
    if r.pos != r.end {
        return Err(PtlError::Trailing);
    }

    Ok(PtlFile {
        version,
        particles,
        emitters,
    })
}

fn parse_particle(r: &mut Reader<'_>) -> Result<PtlParticle, PtlError> {
    let name = r.cstr(16)?;
    let frames = r.u16()?; // 0x16
    let rate = r.u16()?; // 0x14
    let scale = r.f32s5()?; // 0x18..0x28
    let flags = r.u8s4()?; // 0x2c..0x30
    let type_byte = r.u8()?; // 0x30
    let f10 = r.f32s10()?; // 0x31..0x59
    let f3 = r.f32s3()?; // 0x59..0x65
    let f3b = r.f32s3()?; // 0x65..0x71
    let f14 = r.f32s14()?; // 0x71..0xa9
    let u_a8 = r.u16()?; // 0xa9
    let u_aa = r.u8()?; // 0xab
    let u_ab = r.u8()?; // 0xac
    let f5 = r.f32s5()?; // 0xad..0xc1
    let mut slots_c0 = Vec::with_capacity(8);
    for _ in 0..8 {
        slots_c0.push(r.f32()?);
        r.skip(4)?; // 4×u8
    }
    let f100 = r.f32s8x2()?; // 0x101..0x141
    let f140 = r.f32s2()?;
    let f148 = r.f32s2()?;
    let f150 = r.f32s8x2()?;
    let f190 = r.f32s8x2()?;
    let f1d0 = r.f32s2()?;
    let f1d8 = r.f32s8x2()?;
    let f218 = r.f32s8x2()?;
    let f258 = r.f32s8x2()?;
    let f298 = r.f32s4()?;
    let f2b0 = r.f32s8x2()?;
    let u2f0 = r.u8s4()?;
    let f2f8 = r.f32s3()?;
    let f2f4 = r.f32()?;
    let f304 = r.f32s3()?;
    let f310 = r.f32s8x2()?;
    let f350 = r.f32s8x2()?;
    let f390 = r.f32s8x2()?;
    let spawn_count = r.u32()?; // 0x3e0
    let mut spawns = Vec::with_capacity(spawn_count as usize);
    for _ in 0..spawn_count {
        spawns.push(PtlSpawnRef {
            name: r.cstr(16)?,
            a: r.u32()?,
            b: r.u32()?,
        });
    }
    // Tail (struct 0x410, 0x414, 0x411, 0x41c, 0x418).
    let tail = [r.u8()?, r.u8()?];
    let tail_f32 = [r.f32()?, r.f32()?, r.f32()?];

    Ok(PtlParticle {
        name,
        frames,
        rate,
        scale,
        flags,
        type_byte,
        f10,
        f3,
        f3b,
        f14,
        u_a8,
        u_aa,
        u_ab,
        f5,
        slots_c0,
        f100,
        f140,
        f148,
        f150,
        f190,
        f1d0,
        f1d8,
        f218,
        f258,
        f298,
        f2b0,
        u2f0,
        f2f8,
        f2f4,
        f304,
        f310,
        f350,
        f390,
        spawn_count,
        spawns,
        tail,
        tail_f32,
    })
}

fn parse_emitter(r: &mut Reader<'_>) -> Result<PtlEmitter, PtlError> {
    let pos = r.f32s3()?;
    let u12 = r.u16s4()?;
    let u20 = r.u16()?;
    let f22 = r.f32()?;
    let name = r.cstr(16)?;
    let u42 = r.u16()?;
    let u44 = r.u16()?;
    let f46 = r.f32()?;
    let u50 = r.u16()?;
    let u52 = r.u16()?;
    let f54 = r.f32()?;
    let f58 = r.f32()?;
    let u62 = r.u16()?;
    let u64 = r.u16()?;
    let u66 = r.u8s3()?;
    let u69 = r.u16()?;
    let u71 = r.u16()?;
    let reserved = r.u8s4()?;
    Ok(PtlEmitter {
        pos,
        u12,
        u20,
        f22,
        name,
        u42,
        u44,
        f46,
        u50,
        u52,
        f54,
        f58,
        u62,
        u64,
        u66,
        u69,
        u71,
        reserved,
    })
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
    end: usize,
}

impl<'a> Reader<'a> {
    fn need(&self, n: usize) -> Result<(), PtlError> {
        if self.pos + n > self.end {
            return Err(PtlError::Truncated);
        }
        Ok(())
    }
    fn u8(&mut self) -> Result<u8, PtlError> {
        self.need(1)?;
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }
    fn u8s4(&mut self) -> Result<[u8; 4], PtlError> {
        self.need(4)?;
        let v = self.data[self.pos..self.pos + 4].try_into().unwrap();
        self.pos += 4;
        Ok(v)
    }
    fn u8s3(&mut self) -> Result<[u8; 3], PtlError> {
        self.need(3)?;
        let v = self.data[self.pos..self.pos + 3].try_into().unwrap();
        self.pos += 3;
        Ok(v)
    }
    fn u16(&mut self) -> Result<u16, PtlError> {
        self.need(2)?;
        let v = u16::from_le_bytes(self.data[self.pos..self.pos + 2].try_into().unwrap());
        self.pos += 2;
        Ok(v)
    }
    fn u16s4(&mut self) -> Result<[u16; 4], PtlError> {
        self.need(8)?;
        let mut out = [0u16; 4];
        for (i, v) in out.iter_mut().enumerate() {
            *v = u16::from_le_bytes(self.data[self.pos + 2 * i..self.pos + 2 * i + 2].try_into().unwrap());
        }
        self.pos += 8;
        Ok(out)
    }
    fn u32(&mut self) -> Result<u32, PtlError> {
        self.need(4)?;
        let v = u32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }
    fn f32(&mut self) -> Result<f32, PtlError> {
        self.need(4)?;
        let v = f32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }
    fn f32s2(&mut self) -> Result<[f32; 2], PtlError> {
        Ok([self.f32()?, self.f32()?])
    }
    fn f32s3(&mut self) -> Result<[f32; 3], PtlError> {
        Ok([self.f32()?, self.f32()?, self.f32()?])
    }
    fn f32s4(&mut self) -> Result<[f32; 4], PtlError> {
        Ok([self.f32()?, self.f32()?, self.f32()?, self.f32()?])
    }
    fn f32s5(&mut self) -> Result<[f32; 5], PtlError> {
        Ok([self.f32()?, self.f32()?, self.f32()?, self.f32()?, self.f32()?])
    }
    fn f32s10(&mut self) -> Result<[f32; 10], PtlError> {
        let mut out = [0f32; 10];
        for v in out.iter_mut() {
            *v = self.f32()?;
        }
        Ok(out)
    }
    fn f32s14(&mut self) -> Result<[f32; 14], PtlError> {
        let mut out = [0f32; 14];
        for v in out.iter_mut() {
            *v = self.f32()?;
        }
        Ok(out)
    }
    fn f32s8x2(&mut self) -> Result<[[f32; 2]; 8], PtlError> {
        let mut out = [[0f32; 2]; 8];
        for pair in out.iter_mut() {
            pair[0] = self.f32()?;
            pair[1] = self.f32()?;
        }
        Ok(out)
    }
    fn cstr(&mut self, n: usize) -> Result<String, PtlError> {
        self.need(n)?;
        let raw = &self.data[self.pos..self.pos + n];
        self.pos += n;
        let len = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
        Ok(String::from_utf8_lossy(&raw[..len]).into_owned())
    }
    fn skip(&mut self, n: usize) -> Result<(), PtlError> {
        self.need(n)?;
        self.pos += n;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(path: &str) -> Vec<u8> {
        std::fs::read(path).expect(path)
    }

    #[test]
    fn parses_map_ptl() {
        let data = load("backup/LEVELS/MAP/MAP/MAP.PTL");
        let f = parse(&data).unwrap();
        assert_eq!(f.version, 40);
        assert_eq!(f.particles.len(), 44);
        assert_eq!(f.emitters.len(), 88);
        // Size arithmetic: 8 + 44×989 + 4 + 88×77 = 50304.
        assert_eq!(data.len(), 50304);

        let p = &f.particles[0];
        assert_eq!(p.name, "New98099");
        assert_eq!(p.frames, 108);
        assert_eq!(p.rate, 76);
        // Scale group: the file starts with a zero float, then real values.
        assert_eq!(
            p.scale.map(|v| v.to_bits()),
            [0, 0x3D950007, 0x3D8E7FFC, 0x3EF24AA1, 0x4003B953]
        );
        assert_eq!(p.flags, [0, 0, 0, 2]);
        assert_eq!(p.type_byte, 0);
        assert_eq!(
            p.f10.map(|v| v.to_bits()),
            [0x471C4000, 0, 0x41C80000, 0, 0, 0x3F000000, 0x3ED0FEA0, 0, 0, 0]
        );
        assert_eq!(p.spawn_count, 0);
        assert_eq!(p.spawns.len(), 0);
        assert_eq!(p.tail, [0, 0]);
        assert_eq!(p.tail_f32, [0.0, 0.0, 0.0]);

        // Section 2 first record (current MAP.PTL fixture).
        let e = &f.emitters[0];
        assert_eq!(e.name, "New98");
        assert_eq!(e.pos.map(|v| v.to_bits()), [0xC1F2D608, 0x3F1BC963, 0xC258D41B]);
        assert_eq!(e.u12, [0, 0, 0xFF6E, 0xBECA]);
        assert_eq!(e.u20, 0);
        assert_eq!(f.emitters[1].name, "New98099");
    }

    #[test]
    fn parses_dooku_b_ptl() {
        let data = load("backup/LEVELS/EPISODE_II/DOOKU/DOOKU_B/DOOKU_B.PTL");
        let f = parse(&data).unwrap();
        assert_eq!(f.version, 40);
        assert_eq!(f.particles.len(), 2);
        assert_eq!(f.emitters.len(), 18);
        // 8 + 2×989 + 4 + 18×77 = 3376.
        assert_eq!(data.len(), 3376);
        assert_eq!(f.emitters[0].name, "DUST1");
        assert_eq!(
            f.emitters[0].pos.map(|v| v.to_bits()),
            [0xBF6E2CF0, 0x3F04BB80, 0xBEF110FF]
        );
        assert_eq!(f.emitters[0].u12, [0, 0, 0xFFA0, 0xFFB5]);
    }

    #[test]
    fn rejects_other_versions() {
        // RETAKE_INTRO2/3 are version 39/33 leftovers (not v40).
        let v39 = load("backup/LEVELS/EPISODE_I/RETAKEPALACE/RETAKE_INTRO/RETAKE_INTRO2.PTL");
        assert!(matches!(parse(&v39), Err(PtlError::UnsupportedVersion(39))));
        let v33 = load("backup/LEVELS/EPISODE_I/RETAKEPALACE/RETAKE_INTRO/RETAKE_INTRO3.PTL");
        assert!(matches!(parse(&v33), Err(PtlError::UnsupportedVersion(33))));
    }
}
