//! Parser for level AI scripts (`.AI2`), mirroring BrickBench's
//! `AI2Loader.java` byte-for-byte (version-aware).
//!
//! The file begins with a version + path count, then a set of navigation
//! paths, then triggers, then (when paths exist) AI locators, locator sets
//! and creature spawns. Positions are raw game/world coordinates.

use anyhow::{ensure, Context, Result};
use glam::Vec3;

pub struct Ai2 {
    pub version: u32,
    pub paths: Vec<AiPath>,
    pub triggers: Vec<Trigger>,
    pub locators: Vec<Locator>,
    pub locator_sets: Vec<LocatorSet>,
    pub creatures: Vec<Creature>,
}

pub struct AiPath {
    pub name: String,
    pub connections: Vec<AiConnection>,
    pub points: Vec<PathPoint>,
}

pub struct AiConnection {
    pub to: u8,
    pub field_0x11: u8,
    pub sub0: i32,
    pub sub4: i32,
    pub sub12: u16,
    pub sub14: u16,
    pub sub18: f32,
    pub sub1c: f32,
}

pub struct PathPoint {
    pub name: String,
    pub pos: Vec3,
    pub xz_size: f32,
    pub min_y: f32,
    pub max_y: f32,
    pub connections: Vec<u16>,
    pub special_name: String,
    pub special_pos: Option<Vec3>,
}

pub struct Trigger {
    pub name: String,
    pub pos: Vec3,
    pub half_size: Vec3,
    pub angle: f32,
    pub offset: usize,
}

pub struct Locator {
    pub name: String,
    pub pos: Vec3,
    pub angle: f32,
    pub offset: usize,
}

pub struct LocatorSet {
    pub name: String,
    pub locators: Vec<usize>,
}

pub struct Creature {
    pub name: String,
    pub script: String,
    pub char_type: String,
    pub pos: Vec3,
    pub angle: f32,
    pub offset: usize,
}

/// Map a stored short angle to degrees (`shortAngleToFloat`).
fn angle(u: u16) -> f32 {
    (u as i16) as f32 * 180.0 / 32767.0
}

/// Read `n` bytes as a name string, truncated at the first NUL and trimmed.
pub(crate) fn read_name(data: &[u8], o: usize, n: usize) -> String {
    let end = data[o..o + n]
        .iter()
        .position(|&b| b == 0)
        .map(|p| o + p)
        .unwrap_or(o + n);
    String::from_utf8_lossy(&data[o..end])
        .trim()
        .to_string()
}

pub(crate) struct R<'a> {
    d: &'a [u8],
    p: usize,
}

impl<'a> R<'a> {
    pub(crate) fn new(d: &'a [u8]) -> Self {
        Self { d, p: 0 }
    }

    pub(crate) fn pos(&self) -> usize {
        self.p
    }

    pub(crate) fn skip(&mut self, n: usize) -> Result<()> {
        ensure!(self.p + n <= self.d.len(), "read past end at {:#x}", self.p);
        self.p += n;
        Ok(())
    }

    pub(crate) fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        ensure!(self.p + n <= self.d.len(), "read past end at {:#x}", self.p);
        let s = &self.d[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }

    pub(crate) fn u8(&mut self) -> Result<u8> {
        Ok(self.bytes(1)?[0])
    }

    pub(crate) fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.bytes(2)?.try_into().unwrap()))
    }

    pub(crate) fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }

    pub(crate) fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }

    pub(crate) fn vec3(&mut self) -> Result<Vec3> {
        Ok(Vec3::new(self.f32()?, self.f32()?, self.f32()?))
    }

    /// A fixed-width name string (`name_len` bytes).
    fn name(&mut self, name_len: usize) -> Result<String> {
        let o = self.p;
        self.skip(name_len)?;
        Ok(read_name(self.d, o, name_len))
    }
}

pub fn parse(data: &[u8]) -> Result<Ai2> {
    let mut r = R::new(data);
    let version = r.u32()?;
    let path_count = r.u32()?;

    let mut paths = Vec::with_capacity(path_count as usize);
    for _ in 0..path_count {
        let name = r.name(16)?;
        let sub_count2 = r.u8()?;
        r.u8()?; // field0x15

        let cnx_count = if version == 1 {
            r.u8()? as usize
        } else {
            r.u16()? as usize
        };
        let mut connections = Vec::with_capacity(cnx_count);
        for _ in 0..cnx_count {
            let to = r.u8()?;
            let field_0x11 = r.u8()?;
            let (sub0, sub4) = if version < 0xc {
                if version < 9 {
                    (r.u8()? as i32, r.u8()? as i32)
                } else {
                    (r.u16()? as i32, r.u16()? as i32)
                }
            } else {
                (r.u32()? as i32, r.u32()? as i32)
            };
            connections.push(AiConnection {
                to,
                field_0x11,
                sub0,
                sub4,
                sub12: r.u16()?,
                sub14: r.u16()?,
                sub18: r.f32()?,
                sub1c: r.f32()?,
            });
        }

        if version == 1 {
            r.skip(1)?;
        }

        let mut points = Vec::with_capacity(sub_count2 as usize);
        for _ in 0..sub_count2 {
            let num_to_read = r.u32()? as usize;
            let point_name = if num_to_read != 0 {
                read_name(r.d, r.pos(), num_to_read)
            } else {
                String::new()
            };
            r.skip(num_to_read)?;
            let pos = r.vec3()?;
            let xz_size = r.f32()?;
            let (min_y, max_y) = if version < 8 {
                (pos.y - 0.2, pos.y + 0.2)
            } else {
                (r.f32()?, r.f32()?)
            };
            let subsub_count = r.u8()?;
            r.skip(3)?; // byte, byte, flag
            r.u16()?; // unknown short
            r.u8()?; // unknown (v<0x13 reads it oddly; position unaffected)

            let special_len = r.u8()? as usize;
            let (special_name, special_pos) = if special_len != 0 {
                let sname = read_name(r.d, r.pos(), special_len);
                r.skip(special_len)?;
                let spos = r.vec3()?;
                (sname, Some(spos))
            } else {
                (String::new(), None)
            };

            let mut conns = Vec::with_capacity(subsub_count as usize);
            for _ in 0..subsub_count {
                conns.push(r.u16()?);
            }
            if subsub_count & 1 != 0 {
                r.u16()?; // odd padding
            }
            if 4 < version {
                r.u16()?;
                r.u16()?;
            }
            points.push(PathPoint {
                name: point_name,
                pos,
                xz_size,
                min_y,
                max_y,
                connections: conns,
                special_name,
                special_pos,
            });
        }

        for _ in 0..sub_count2 {
            r.skip(sub_count2 as usize)?;
        }

        if 4 < version {
            let sub_object3_count = r.u8()?;
            for _ in 0..sub_object3_count {
                let size2 = r.u8()? as usize;
                if size2 != 0 {
                    r.skip(size2)?;
                    let buffer3_size = r.u8()? as usize;
                    let buffer4_size = r.u8()? as usize;
                    r.skip(2)?;
                    if sub_count2 != 0 && buffer3_size != 0 {
                        r.skip(sub_count2 as usize)?;
                        r.skip(buffer3_size)?;
                        for _ in 0..buffer3_size {
                            r.skip(buffer3_size)?;
                        }
                        if buffer4_size != 0 {
                            r.skip(buffer4_size)?;
                        }
                    }
                }
                let another_count = r.u8()?;
                for _ in 0..another_count {
                    let byte_var = r.u8()? as usize;
                    r.skip(byte_var)?;
                }
            }
        }

        if 0x12 < version {
            let sub_object4_count = r.u8()?;
            for _ in 0..sub_object4_count {
                r.u8()?;
                r.u16()?;
            }
        }

        paths.push(AiPath {
            name,
            connections,
            points,
        });
    }

    if 0x12 < version {
        let obj4_count = r.u16()? as usize;
        for _ in 0..obj4_count {
            let unk = r.u8()? as usize;
            if unk != 0 {
                r.skip(unk)?;
            }
        }
    }

    let mut triggers = Vec::new();
    let num_triggers = r.u32()? as usize;
    for _ in 0..num_triggers {
        let offset = r.pos();
        let name = r.name(16)?;
        let pos = r.vec3()?;
        let half_size = r.vec3()?;
        let ang = r.u16()?;
        r.skip(2)?;
        triggers.push(Trigger {
            name,
            pos,
            half_size,
            angle: angle(ang),
            offset,
        });
    }

    let mut locators = Vec::new();
    let mut locator_sets = Vec::new();
    let mut creatures = Vec::new();

    if path_count != 0 {
        if version >= 6 {
            let locator_count = r.u32()? as usize;
            for _ in 0..locator_count {
                let offset = r.pos();
                let lname = r.name(16)?;
                let pos = r.vec3()?;
                let ang = r.u16()?;
                r.skip(4)?; // byte pathIDMaybe, byte unk, u16 connectionsMaybe
                r.f32()?;
                r.f32()?;
                if version >= 15 {
                    r.u32()?;
                }
                locators.push(Locator {
                    name: lname,
                    pos,
                    angle: angle(ang),
                    offset,
                });
            }

            if version >= 18 {
                let locator_set_count = r.u32()? as usize;
                for _ in 0..locator_set_count {
                    let set_name = r.name(16)?;
                    let locator_count = r.u32()? as usize;
                    let mut set = Vec::with_capacity(locator_count);
                    for _ in 0..locator_count {
                        set.push(r.u8()? as usize);
                    }
                    locator_sets.push(LocatorSet {
                        name: set_name,
                        locators: set,
                    });
                }
            }
        }

        let num_creatures = r.u32()? as usize;
        for _ in 0..num_creatures {
            let offset = r.pos();
            let cname = r.name(16)?;
            let script = r.name(16)?;
            let text_size = if version >= 14 { 0x20 } else { 0x10 };
            let char_type = r.name(text_size)?;
            let pos = r.vec3()?;
            let start_angle = r.u16()?;
            if version >= 16 {
                r.u8()?;
            }
            r.skip(2)?;
            r.u32()?;
            r.f32()?;
            r.f32()?;
            r.u32()?;
            r.skip(2)?;
            r.u16()?;
            if version >= 3 {
                r.f32()?;
                r.f32()?;
                r.f32()?;
                r.f32()?;
            }
            if version >= 4 && r.u32()? != 0 {
                r.skip(16)?; // trigger1 ref
            }
            if version >= 6 && r.u32()? != 0 {
                r.skip(16)?; // locator1 ref
            }
            if version >= 17 && r.u32()? != 0 {
                r.skip(16)?; // locator2 ref
            }
            if version >= 8 {
                r.skip(3)?;
                let load_2nd_trigger = r.u8()?;
                r.f32()?;
                r.f32()?;
                if version >= 10 {
                    r.f32()?;
                }
                if load_2nd_trigger == 1 {
                    r.skip(16)?; // trigger2 ref
                }
                if version >= 11 {
                    r.f32()?;
                    r.f32()?;
                    r.f32()?;
                    r.f32()?;
                    r.u32()?;
                }
            }
            creatures.push(Creature {
                name: cname,
                script,
                char_type,
                pos,
                angle: angle(start_angle),
                offset,
            });
        }
    }

    Ok(Ai2 {
        version,
        paths,
        triggers,
        locators,
        locator_sets,
        creatures,
    })
}

/// Parse the file at `path`, returning a context-wrapped error on failure.
pub fn parse_file(path: &str) -> Result<Ai2> {
    let data = std::fs::read(path).with_context(|| format!("reading {path}"))?;
    parse(&data).with_context(|| format!("parsing {path}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_ai2_parses_locators_and_creatures() {
        let path = "backup/LEVELS/MAP/MAP/AI/MAP.AI2";
        let data = match std::fs::read(path) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping: {path} not present");
                return;
            }
        };
        let ai = parse(&data).expect("MAP.AI2 should parse");

        assert_eq!(ai.version, 20, "hub AI2 version");
        assert!(!ai.paths.is_empty(), "hub has navigation paths");
        assert!(!ai.locators.is_empty(), "hub has AI locators");

        let names: Vec<&str> = ai.locators.iter().map(|l| l.name.as_str()).collect();
        for want in ["BARMAN_1", "SERVEPLAYER", "JABBA_1", "JABBA_2", "JABBA_3"] {
            assert!(names.contains(&want), "missing locator {want}: {names:?}");
        }

        let creatures: Vec<&str> = ai.creatures.iter().map(|c| c.name.as_str()).collect();
        for want in ["mapcar_1", "mapcar_2"] {
            assert!(
                creatures.contains(&want),
                "missing creature {want}: {creatures:?}"
            );
        }

        // The cantina main-room trigger is the natural player spawn point.
        let main = ai
            .triggers
            .iter()
            .find(|t| t.name == "MAINROOM")
            .expect("MAINROOM trigger");
        assert!(main.pos.distance(Vec3::new(-27.49, -0.29, -52.72)) < 0.5);
    }
}
