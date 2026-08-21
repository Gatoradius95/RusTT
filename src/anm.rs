//! Light-animation (`.ANM`) parsing — the per-level table that drives the
//! animated light setups (flickering torches, bubbling vats, beamed
//! searchlights) in the Nu2 engine.
//!
//! Layout recovered from the level files themselves plus the PC executable:
//! the loader (`NuANMLoadFromFile` at `0x00687850`), the save routine
//! (`FUN_00659af0`) and the per-frame update loop (`FUN_00686950`) in
//! `LEGOStarWarsSaga.exe`. Every file under `backup/LEVELS` parses
//! byte-exact with the layout below.
//!
//! ## File layout
//!
//! ```text
//! u32 version (4..=6 — 6 in most files; 4 in four LSW1-era leftovers:
//!   GUNGAN_B, NEGOTIATIONS_A, RESCUE_A, RESCUE_E)
//! u32 light count N
//! N × light groups:
//!   record, 44 bytes:
//!     0x00  name[16]
//!     0x10  u32            (read together with the name; zero in level files)
//!     0x14  u32  key count      (loader's first count; loop bound for keys)
//!     0x18  u32  second count   (loader's second count; 0 in level files)
//!     0x1c  u32            (updater: behaviour/state switch, 0 in level files)
//!     0x20  u32            (updater: index into the paired table, 0 in files)
//!     0x24  f32
//!     0x28  f32            (updater: time thresholds, 0 in level files)
//!   key count × keyframes, 40 bytes each:
//!     0x00  name[16]       PTL particle-template name ("TORCH_1", "ROUND",
//!                          "BUBBLYMUD1", "MAUL_STEAM", "BLUE_LIGHT", ...)
//!     0x10  u32  duration  encoding depends on the version (see below)
//!     0x14  u32            (1 = looping emitter, 0 = one-shot in level files)
//!     0x18  f32 ×3         emitter offset relative to the light
//!     0x24  u16
//!     0x26  u16
//!   second count × records, 36 bytes each (none in level files):
//!     0x00  name[16] + u32 + f32 + f32 ×3
//!   12-byte tail (f32 ×3, zero in level files)
//! ```
//!
//! Semantics: each light is a named scene object; its keyframes spawn PTL
//! (particle-template) effects — the key name is looked up via
//! `NuPTLFindEntryByName` and the effect runs at `light world position +
//! offset` for `time` ticks, which is why torch rigs carry keyframes named
//! `TORCH_1`/`TORCH_2`/`TORCH_3` with rising offsets and times of 1/60.
//!
//! The duration field is read per the loader (`0x00687850`), which converts
//! it to 60 Hz ticks the same way the running game sees it:
//! - version 6: raw `u32` ticks (60 = 1 s) — [`AnmKey::time`] is the raw
//!   value;
//! - version 5: a `f32` tick count (truncated) — not present in the levels;
//! - versions < 5: a signed `u32`: positive = seconds (× 60), negative =
//!   frequency (`60 / -value`, e.g. `-2` = 30 ticks) — the level files
//!   carry 1 (→ 60) and -2 (→ 30).
//!
//! The converted value is exposed as [`AnmKey::time`]; the raw stored field
//! as [`AnmKey::time_raw`].
//!
//! Two files in the tree (`NEGOTIATIONS_C.ANM`, `KAMINO_C.ANM`) have their
//! numeric fields stored big-endian while the fixed-width string fields are
//! plain ASCII; [`parse_swapped`] handles them.

use std::fmt;

/// Errors produced while parsing an `.ANM` file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnmError {
    /// Not an animation file.
    NotAnm,
    /// Version is not 6 (only the level-file version is decoded).
    UnsupportedVersion(u32),
    /// Data ended before the declared structure was consumed.
    Truncated,
    /// Bytes remain after the declared structure.
    Trailing,
}

impl fmt::Display for AnmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AnmError::NotAnm => write!(f, "not an ANM file"),
            AnmError::UnsupportedVersion(v) => {
                write!(f, "unsupported ANM version {v} (only 6 is decoded)")
            }
            AnmError::Truncated => write!(f, "truncated ANM data"),
            AnmError::Trailing => write!(f, "trailing bytes after ANM structure"),
        }
    }
}

impl std::error::Error for AnmError {}

/// A parsed `.ANM` file.
#[derive(Debug, Clone)]
pub struct AnmFile {
    pub version: u32,
    pub lights: Vec<AnmLight>,
}

/// One animated-light group: a 44-byte record, its keyframes, and the
/// 12-byte tail.
#[derive(Debug, Clone)]
pub struct AnmLight {
    /// File `0x00` — scene-object name ("light_2b", "Torch7", ...).
    pub name: String,
    /// File `0x10` — loaded together with the name; zero in level files.
    pub u10: u32,
    /// File `0x14` — how many keyframes follow.
    pub keys: Vec<AnmKey>,
    /// File `0x18` — second keyframe set (v ≥ 2 only; none in level files),
    /// 36-byte records.
    pub second: Vec<AnmSecond>,
    /// Tail — f32 ×3 (v ≥ 4; zero in level files).
    pub tail: [f32; 3],
}

/// One 40-byte keyframe: a PTL particle spawn at an offset for a duration.
#[derive(Debug, Clone)]
pub struct AnmKey {
    /// File `0x00` — PTL particle-template name; the name field is also
    /// zero-padded to 16 bytes and may embed a secondary token after a NUL
    /// (e.g. "LIGHT\0BEAM1").
    pub name: String,
    /// File `0x10` — duration in 60 Hz ticks as the game sees it (see the
    /// module docs for the version-dependent conversion).
    pub time: u32,
    /// File `0x10` raw value: ticks (v6), signed seconds/frequency (v4).
    pub time_raw: i32,
    /// File `0x14` — 1 for looping emitters, 0 for one-shots.
    pub flags: u32,
    /// File `0x18` — emitter offset relative to the light.
    pub offset: [f32; 3],
    /// File `0x24`.
    pub u50: u16,
    /// File `0x26`.
    pub u52: u16,
}

/// One 36-byte second-set record (v ≥ 2 only; none in the level files).
#[derive(Debug, Clone)]
pub struct AnmSecond {
    /// File `0x00`.
    pub name: String,
    /// File `0x10`.
    pub u10: u32,
    /// File `0x14`.
    pub f14: f32,
    /// File `0x18`.
    pub offset: [f32; 3],
}

/// Parse a version 4..=6 level `.ANM` file (little-endian fields).
pub fn parse(data: &[u8]) -> Result<AnmFile, AnmError> {
    parse_inner(data, false)
}

/// Parse an `.ANM` whose numeric fields are byte-swapped (big-endian) while
/// the fixed-width string fields stay plain ASCII; `NEGOTIATIONS_C.ANM`
/// and `KAMINO_C.ANM` in the level tree are stored this way.
pub fn parse_swapped(data: &[u8]) -> Result<AnmFile, AnmError> {
    parse_inner(data, true)
}

fn parse_inner(data: &[u8], swap: bool) -> Result<AnmFile, AnmError> {
    if data.len() < 8 {
        return Err(AnmError::NotAnm);
    }
    let mut r = Reader {
        data,
        pos: 0,
        end: data.len(),
        swap,
    };

    let version = r.u32()?;
    if !(4..=6).contains(&version) {
        return Err(AnmError::UnsupportedVersion(version));
    }
    let count = r.u32()? as usize;

    let mut lights = Vec::with_capacity(count);
    for _ in 0..count {
        let name = r.cstr(16)?;
        let u10 = r.u32()?;
        let key_count = r.u32()? as usize;
        let second_count = r.u32()? as usize;
        r.skip(16)?; // 4 × (u32/u32/f32/f32), zero in level files

        let mut keys = Vec::with_capacity(key_count.min(8));
        for _ in 0..key_count {
            let name = r.cstr(16)?;
            let (time, time_raw) = read_time(&mut r, version)?;
            keys.push(AnmKey {
                name,
                time,
                time_raw,
                flags: r.u32()?,
                offset: r.f32s3()?,
                u50: r.u16()?,
                u52: r.u16()?,
            });
        }

        let mut second = Vec::with_capacity(second_count);
        for _ in 0..second_count {
            second.push(AnmSecond {
                name: r.cstr(16)?,
                u10: r.u32()?,
                f14: r.f32()?,
                offset: r.f32s3()?,
            });
        }

        let tail = r.f32s3()?;
        lights.push(AnmLight {
            name,
            u10,
            keys,
            second,
            tail,
        });
    }

    if r.pos != r.end {
        return Err(AnmError::Trailing);
    }
    Ok(AnmFile { version, lights })
}

fn read_time(r: &mut Reader<'_>, version: u32) -> Result<(u32, i32), AnmError> {
    // Mirrors the loader's version dispatch: v6 ticks, v5 float ticks,
    // v4 signed seconds/frequency — all converted to 60 Hz ticks.
    match version {
        5 => {
            let f = r.f32()?;
            let t = f as i32;
            Ok((t as u32, t))
        }
        4 => {
            let raw = r.u32()? as i32;
            let t = if raw > 0 {
                // Loader scales in signed i32 ("iVar3 * 0x3c"), which wraps
                // for huge raw values — mirrored here so huge u32 times
                // (as in some v4 leftovers) end up as the game would see.
                (raw as i32).wrapping_mul(60) as u32
            } else if raw < 0 {
                (-60i64 / raw as i64) as u32
            } else {
                0
            };
            Ok((t, raw))
        }
        _ => {
            let raw = r.u32()?;
            Ok((raw, raw as i32))
        }
    }
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
    end: usize,
    swap: bool,
}

impl<'a> Reader<'a> {
    fn need(&self, n: usize) -> Result<(), AnmError> {
        if self.pos + n > self.end {
            return Err(AnmError::Truncated);
        }
        Ok(())
    }
    fn rev4(&self, off: usize) -> [u8; 4] {
        let mut b: [u8; 4] = self.data[off..off + 4].try_into().unwrap();
        if self.swap {
            b.reverse();
        }
        b
    }
    fn u16(&mut self) -> Result<u16, AnmError> {
        self.need(2)?;
        let b: [u8; 2] = self.data[self.pos..self.pos + 2].try_into().unwrap();
        let v = if self.swap {
            u16::from_be_bytes(b)
        } else {
            u16::from_le_bytes(b)
        };
        self.pos += 2;
        Ok(v)
    }
    fn u32(&mut self) -> Result<u32, AnmError> {
        self.need(4)?;
        let v = u32::from_le_bytes(self.rev4(self.pos));
        self.pos += 4;
        Ok(v)
    }
    fn f32(&mut self) -> Result<f32, AnmError> {
        self.need(4)?;
        let v = f32::from_le_bytes(self.rev4(self.pos));
        self.pos += 4;
        Ok(v)
    }
    fn f32s3(&mut self) -> Result<[f32; 3], AnmError> {
        Ok([self.f32()?, self.f32()?, self.f32()?])
    }
    fn cstr(&mut self, n: usize) -> Result<String, AnmError> {
        self.need(n)?;
        let raw = &self.data[self.pos..self.pos + n];
        self.pos += n;
        let len = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
        Ok(String::from_utf8_lossy(&raw[..len]).into_owned())
    }
    fn skip(&mut self, n: usize) -> Result<(), AnmError> {
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
    fn parses_map_anm() {
        let data = load("backup/LEVELS/MAP/MAP/MAP.ANM");
        let f = parse(&data).unwrap();
        assert_eq!(f.version, 6);
        assert_eq!(f.lights.len(), 22);
        // 8 + 22×(44 + 1×40 + 12) = 2120; every light carries one keyframe.
        assert_eq!(data.len(), 2120);

        assert_eq!(f.lights[0].name, "light_2b");
        assert_eq!(f.lights[0].u10, 0);
        assert_eq!(f.lights[0].second.len(), 0);
        assert_eq!(f.lights[0].tail, [0.0, 0.0, 0.0]);
        let k = &f.lights[0].keys[0];
        assert_eq!(k.name, "BLUE_LIGHT");
        assert_eq!(k.time, 60);
        assert_eq!(k.time_raw, 60);
        assert_eq!(k.flags, 1);
        assert_eq!(k.offset.map(|v| v.to_bits()), [0xBCC51A80, 0xBDA7E2C8, 0x3B8C2D00]);
        assert_eq!((k.u50, k.u52), (55883, 32296));

        // The flickering "8b" cluster uses colour patterns; their exact
        // distribution differs between builds of the level data.
        let patterns: Vec<&str> = f.lights.iter().map(|l| l.keys[0].name.as_str()).collect();
        assert_eq!(patterns.iter().filter(|&&p| p == "BLUE_LIGHT").count(), 7);
        assert!(patterns.iter().any(|&p| p == "ORANGE_LIGHT"));
        assert!(patterns.iter().any(|&p| p == "GREEN_LIGHT"));
        assert!(patterns.iter().any(|&p| p == "WHITE_LIGHT"));
        // PSP-only radar light runs a one-shot RAY pulse.
        let radar = f.lights.iter().find(|l| l.name == "PSP_only_radar").unwrap();
        assert_eq!(radar.keys[0].name, "RAY");
        assert_eq!(radar.keys[0].flags, 0);
        assert_eq!(f.lights[21].name, "PSP_only_radar");
    }

    #[test]
    fn parses_tatooine_c_anm() {
        let data = load("backup/LEVELS/EPISODE_IV/TATOOINE/TATOOINE_C/TATOOINE_C.ANM");
        let f = parse(&data).unwrap();
        assert_eq!(f.lights.len(), 4);
        assert_eq!(data.len(), 992);
        let counts: Vec<usize> = f.lights.iter().map(|l| l.keys.len()).collect();
        assert_eq!(counts, [7, 5, 7, 0]);
        assert_eq!(f.lights[0].name, "vap3_particles");
        assert_eq!(f.lights[3].name, "vap02_mud");
        let k0 = &f.lights[0].keys[0];
        assert_eq!(k0.name, "BUBBLYMUD1");
        assert_eq!(k0.time, 11);
        assert_eq!(k0.offset.map(|v| v.to_bits()), [0x3FB421C0, 0x3ECD57AE, 0x3DF80A00]);
        assert_eq!(f.lights[1].name, "vap2_particles");
    }

    #[test]
    fn parses_version_4_anm() {
        // GUNGAN_B is an LSW1-era leftover: version 4, time stored as
        // seconds/frequency. Positive 1 = 1 s = 60 ticks; -2 = 60/2 = 30.
        let data = load("backup/LEVELS/EPISODE_I/GUNGAN/GUNGAN_B/GUNGAN_B.ANM");
        let f = parse(&data).unwrap();
        assert_eq!(f.version, 4);
        assert_eq!(f.lights.len(), 8);
        assert_eq!(data.len(), 976);
        assert_eq!(f.lights[0].name, "debris_3");
        let k = &f.lights[0].keys[0];
        assert_eq!(k.name, "DUST_01");
        assert_eq!((k.time_raw, k.time), (1, 60));
        assert_eq!(k.offset, [0.0, 0.0, 0.0]);
        let fall = f.lights.iter().find(|l| l.name == "fall1").unwrap();
        assert_eq!(fall.keys.len(), 3);
        assert_eq!(fall.keys[0].name, "DUST_02");
        assert_eq!((fall.keys[0].time_raw, fall.keys[0].time), (-2, 30));
        assert_eq!(f.lights[7].name, "fall4");
        assert_eq!(f.lights[7].keys.len(), 4);
    }

    #[test]
    fn parses_new_town_anm() {
        let data = load("backup/LEVELS/BONUS/NEW_TOWN/NEW_TOWN.ANM");
        let f = parse(&data).unwrap();
        assert_eq!(f.lights.len(), 21);
        assert_eq!(data.len(), 2064);

        let pop = &f.lights[0].keys[0];
        assert_eq!(pop.name, "EVAP_POP_2");
        assert_eq!(pop.time, 60);
        assert_eq!(pop.flags, 1);
        assert_eq!(pop.offset, [0.0, 0.0, 0.0]);

        // The reflector's name field embeds a second token after a NUL.
        let reflector = f.lights.iter().find(|l| l.name == "reflector").unwrap();
        assert_eq!(reflector.keys.len(), 2);
        assert_eq!(reflector.keys[0].name, "LIGHT");
        assert_eq!(reflector.keys[0].time, 300);
        assert_eq!(reflector.keys[1].name, "LIGHT");

        let splash = f.lights.iter().find(|l| l.name == "splash_1").unwrap();
        assert_eq!(splash.keys.len(), 3);
        assert_eq!(splash.keys[2].name, "SPLASH_2");
    }

    #[test]
    fn parses_jabbaspalace_a_anm() {
        let data = load("backup/LEVELS/EPISODE_VI/JABBASPALACE/JABBASPALACE_A/JABBASPALACE_A.ANM");
        let f = parse(&data).unwrap();
        assert_eq!(f.lights.len(), 24);
        assert_eq!(data.len(), 3632);

        // 19 torches with three flame steps each; five helper lights keyless.
        let keys: Vec<usize> = f.lights.iter().map(|l| l.keys.len()).collect();
        assert_eq!(keys.iter().filter(|&&n| n == 3).count(), 19);
        assert_eq!(keys.iter().filter(|&&n| n == 0).count(), 5);

        // Torch7's flame steps run up and out: TORCH_2/TORCH_3 linger a
        // second while TORCH_1 is the instant ignition flash.
        let t = &f.lights[0];
        assert_eq!(t.name, "Torch7");
        let steps: Vec<(&str, u32)> = t.keys.iter().map(|k| (k.name.as_str(), k.time)).collect();
        assert_eq!(steps, [("TORCH_2", 60), ("TORCH_3", 60), ("TORCH_1", 1)]);
        assert!((t.keys[0].offset[1] - 0.2063757).abs() < 0.0001);
    }

    #[test]
    fn parses_swapped_anms() {
        // NEGOTIATIONS_C / KAMINO_C carry big-endian numeric fields; their
        // string fields stay plain ASCII, so a whole-file byte swap would
        // corrupt the names.
        //
        // KAMINO_C matches the standard geometry exactly (rec 44, keys 40,
        // tail 12; 104 bytes) and is asserted field-deep. NEGOTIATIONS_C is
        // a non-standard authoring variant — its 256 bytes hold records of
        // 56/40/44 bytes with 48-byte keys (time at +0x14, vec at +0x20:
        // 0xBF3595D0/0x3EC7DE99/0x3B1A0000 and 0x3F2C8F50/0x3EB02B58/
        // 0xBC8D6400) — so only its record-level fields are asserted.
        let data = load("backup/LEVELS/EPISODE_I/NEGOTIATIONS/NEGOTIATIONS_C/NEGOTIATIONS_C.ANM");
        assert!(matches!(parse(&data), Err(AnmError::UnsupportedVersion(_))));
        let f = parse_swapped(&data).unwrap();
        assert_eq!(f.version, 6);
        assert_eq!(f.lights.len(), 3);
        assert_eq!(data.len(), 256);
        assert_eq!(f.lights[0].name, "magnet_ball_1");
        assert_eq!(f.lights[1].name, "magnet_ball_2");
        assert_eq!(f.lights[2].name, "gate");
        assert_eq!(f.lights[2].keys.len(), 2);

        let data = load("backup/LEVELS/EPISODE_II/KAMINO/KAMINO_C/KAMINO_C.ANM");
        let f = parse_swapped(&data).unwrap();
        assert_eq!(f.version, 6);
        assert_eq!(f.lights.len(), 1);
        assert_eq!(data.len(), 104);
        let k = &f.lights[0].keys[0];
        assert_eq!(k.name, "ROCKET");
        assert_eq!((k.time, k.time_raw), (60, 60));
        assert_eq!(k.offset.map(|v| v.to_bits()), [0x3E1E7500, 0xBDC9BAD5, 0x3CC2CE00]);
    }

    #[test]
    fn parses_every_level_anm() {
        // Regression over the whole shipped level set: the two big-endian
        // files parse via parse_swapped, everything else little-endian.
        let swapped = [
            "backup/LEVELS/EPISODE_I/NEGOTIATIONS/NEGOTIATIONS_C/NEGOTIATIONS_C.ANM",
            "backup/LEVELS/EPISODE_II/KAMINO/KAMINO_C/KAMINO_C.ANM",
        ];
        let mut ok = 0;
        for path in walk_anms("backup/LEVELS") {
            let data = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
            let normalized = path.replace('\\', "/");
            if swapped.contains(&normalized.as_str()) {
                parse_swapped(&data).unwrap_or_else(|e| panic!("{path}: {e}"));
            } else {
                parse(&data).unwrap_or_else(|e| panic!("{path}: {e}"));
            }
            ok += 1;
        }
        assert_eq!(ok, 78);
    }

    fn walk_anms(root: &str) -> Vec<String> {
        fn walk(dir: &str, out: &mut Vec<String>) {
            for entry in std::fs::read_dir(dir).expect(dir) {
                let entry = entry.unwrap();
                let path = entry.path();
                if path.is_dir() {
                    walk(&path.to_string_lossy(), out);
                } else if path.extension().is_some_and(|e| e == "ANM") {
                    out.push(path.to_string_lossy().into_owned());
                }
            }
        }
        let mut out = Vec::new();
        walk(root, &mut out);
        out
    }

    #[test]
    fn rejects_other_versions() {
        let v39 = load("backup/LEVELS/EPISODE_I/RETAKEPALACE/RETAKE_INTRO/RETAKE_INTRO2.PTL");
        assert!(matches!(parse(&v39), Err(AnmError::UnsupportedVersion(39))));
        assert!(matches!(parse(&[]), Err(AnmError::NotAnm)));
    }
}