//! Level light-list (`.RTL`) parsing and per-mesh light-set computation.
//!
//! The original game stores its lights in a small `.RTL` file that sits
//! beside the map's `.GSC` (e.g. `MAP.RTL` next to `MAP_PC.GSC`); the file
//! format follows BrickBench `RTLLoader`: an `int` version header followed by
//! fixed 0x8C-byte light records (slot count depends on the version). At
//! scene load the game bakes the lights near each mesh/part into pixel-shader
//! constants (`lightColor0..2`, scene ambient, ...) that the uber shader
//! consumes as Lambert diffuse + Phong specular — this module reproduces that
//! per-mesh computation so map geometry (walls, floors) gets real shading
//! instead of a flat hardcoded rig.

/// Light kinds, mapped from the RTL record's type byte (BrickBench
/// `RTLLight.LightType`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RtlLightKind {
    /// Spherical light: `distance` = radius of full strength, `falloff` =
    /// outer edge (linear ramp between).
    Point,
    /// Directional light: `dir` is the world-space direction.
    Directional,
    /// Ambient fill: contributes to `scene_ambient` instead of a light slot.
    Ambient,
}

/// One parsed light from an `.RTL` file. Colors are linear RGB.
#[derive(Copy, Clone, Debug)]
pub struct RtlLight {
    pub pos: [f32; 3],
    pub dir: [f32; 3],
    pub color: [f32; 3],
    pub kind: RtlLightKind,
    /// Radius of full strength (point lights / ambient).
    pub distance: f32,
    /// Outer falloff edge; lights beyond it do not contribute.
    pub falloff: f32,
    /// Intensity factor applied to the color contribution.
    pub multiplier: f32,
}

/// The per-mesh light block fed to the uber shader, mirroring the
/// `MeshLights` WGSL uniform (7 vec4s = 112 bytes): `scene_ambient` +
/// `light_color[3]` + `light_pos[3]` (xyz = direction, w = intensity).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightSet {
    pub scene_ambient: [f32; 4],
    pub light_color: [[f32; 4]; 3],
    pub light_pos: [[f32; 4]; 3],
}

/// The classic hardcoded rig used before RTL support: a warm key, cool fill
/// and rim kick with a dim ambient. Used for characters and for maps with no
/// sibling `.RTL` (their previous look is preserved).
impl Default for LightSet {
    fn default() -> Self {
        let norm = |v: [f32; 3]| {
            let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
            [v[0] / len, v[1] / len, v[2] / len]
        };
        Self {
            scene_ambient: [0.10, 0.11, 0.13, 1.0],
            light_color: [
                [1.00, 0.96, 0.88, 1.0],
                [0.35, 0.40, 0.55, 1.0],
                [0.30, 0.28, 0.25, 1.0],
            ],
            light_pos: [
                [
                    norm([0.4, 1.0, 0.3])[0],
                    norm([0.4, 1.0, 0.3])[1],
                    norm([0.4, 1.0, 0.3])[2],
                    1.0,
                ],
                [
                    norm([-0.6, -0.2, 0.4])[0],
                    norm([-0.6, -0.2, 0.4])[1],
                    norm([-0.6, -0.2, 0.4])[2],
                    0.5,
                ],
                [
                    norm([0.0, -0.9, 0.35])[0],
                    norm([0.0, -0.9, 0.35])[1],
                    norm([0.0, -0.9, 0.35])[2],
                    0.35,
                ],
            ],
        }
    }
}

/// Parse an `.RTL` light list (BrickBench `RTLLoader` layout). Lights with an
/// INVALID (0) type are dropped.
pub fn parse(data: &[u8]) -> Vec<RtlLight> {
    if data.len() < 4 {
        return Vec::new();
    }
    let version = i32::from_le_bytes(data[0..4].try_into().unwrap());
    let slot_count = match version {
        0 => 0x80,
        1 => 0,
        2 => 0x40,
        3 => 0x40,
        _ => 0x80,
    };
    let mut out = Vec::new();
    for i in 0..slot_count {
        let o = 4 + i * 0x8c;
        if o + 0x74 > data.len() {
            break;
        }
        let f = |off: usize| f32::from_le_bytes(data[o + off..o + off + 4].try_into().unwrap());
        let t = u16::from_le_bytes(data[o + 0x58..o + 0x5a].try_into().unwrap());
        let kind = match t {
            0 => continue, // INVALID
            1 => RtlLightKind::Ambient,
            2 | 3 | 6 | 7 | 8 => RtlLightKind::Point,
            4 | 5 => RtlLightKind::Directional, // DIRECTIONAL / CAMDIR
            _ => continue,
        };
        out.push(RtlLight {
            pos: [f(0x00), f(0x04), f(0x08)],
            dir: [f(0x0c), f(0x10), f(0x14)],
            color: [f(0x24), f(0x28), f(0x2c)],
            kind,
            distance: f(0x3c),
            falloff: f(0x40),
            multiplier: f(0x6c),
        });
    }
    out
}

/// Resolve the sibling `.RTL` paths for a map `.GSC` path, most specific
/// first.
///
/// The level is shipped as `MAP_PC.GSC` but its light list is written
/// without the platform tag (`MAP.RTL`), so a naive `replace(".GSC",
/// ".RTL")` (or `with_extension("RTL")`) misses the actual file and the map
/// silently falls back to the default rig. This tries the exact sibling
/// first, then the same name with any trailing `_PC`/`_WD`/`_GL` platform
/// tag stripped.
pub fn sibling_rtl_candidates(map_path: &str) -> Vec<std::path::PathBuf> {
    let p = std::path::Path::new(map_path);
    let mut out = vec![p.with_extension("RTL")];
    if let Some(stem) = p.file_stem().and_then(|s| s.to_str()) {
        for tag in ["_PC", "_WD", "_GL"] {
            if let Some(base) = stem.strip_suffix(tag) {
                out.push(p.with_file_name(format!("{base}.RTL")));
                break;
            }
        }
    }
    out
}

/// Compute the per-mesh light set for a mesh centered at `center`, the way
/// the original bakes per-part light constants at scene load (BrickBench
/// `GSCMesh.updateLights`): the strongest three non-ambient lights within
/// falloff become `light_color/light_pos`, ambient lights sum into
/// `scene_ambient`. Directional lights are always included; point lights
/// ramp linearly from full strength at `distance` to zero at `falloff`.
///
/// An empty light list falls back to [`LightSet::default`].
pub fn compute_light_set(lights: &[RtlLight], center: [f32; 3]) -> LightSet {
    if lights.is_empty() {
        return LightSet::default();
    }

    let mut ambient = [0.0f32; 3];
    let mut cols = [[0.0f32; 4]; 3];
    let mut poss = [[0.0f32; 4]; 3];

    // Per-light intensity, folded with the kind for the final slot writes.
    struct Cand {
        intensity: f32,
        dir: [f32; 3],
        color: [f32; 3],
        ambient: bool,
    }
    let mut cands: Vec<Cand> = Vec::new();
    for l in lights {
        let (intensity, dir) = match l.kind {
            RtlLightKind::Directional => (l.multiplier, l.dir),
            RtlLightKind::Point | RtlLightKind::Ambient => {
                let d = ((l.pos[0] - center[0]).powi(2)
                    + (l.pos[1] - center[1]).powi(2)
                    + (l.pos[2] - center[2]).powi(2))
                .sqrt();
                if d >= l.falloff {
                    continue;
                }
                let ramp = if l.falloff <= l.distance {
                    1.0
                } else if d <= l.distance {
                    1.0
                } else {
                    (l.falloff - d) / (l.falloff - l.distance)
                };
                let dir = [
                    l.pos[0] - center[0],
                    l.pos[1] - center[1],
                    l.pos[2] - center[2],
                ];
                let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt().max(1e-6);
                (l.multiplier * ramp, [dir[0] / len, dir[1] / len, dir[2] / len])
            }
        };
        if intensity <= 0.0 {
            continue;
        }
        cands.push(Cand {
            intensity,
            dir,
            color: l.color,
            ambient: l.kind == RtlLightKind::Ambient,
        });
    }

    // Ambient lights: strongest three summed into scene_ambient.
    let mut amb: Vec<&Cand> = cands.iter().filter(|c| c.ambient).collect();
    amb.sort_by(|a, b| b.intensity.total_cmp(&a.intensity));
    for c in amb.iter().take(3) {
        for k in 0..3 {
            ambient[k] += c.color[k] * c.intensity;
        }
    }

    // Non-ambient: strongest three into the light slots.
    let mut lit: Vec<&Cand> = cands.iter().filter(|c| !c.ambient).collect();
    lit.sort_by(|a, b| b.intensity.total_cmp(&a.intensity));
    for (i, c) in lit.iter().take(3).enumerate() {
        cols[i] = [c.color[0], c.color[1], c.color[2], 1.0];
        poss[i] = [c.dir[0], c.dir[1], c.dir[2], c.intensity];
    }

    LightSet {
        scene_ambient: [ambient[0], ambient[1], ambient[2], 1.0],
        light_color: cols,
        light_pos: poss,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_map_rtl_lights() {
        // MAP.RTL: version 5, three lights (1 AMBIENT + 2 CAMDIR).
        let mut data = Vec::new();
        data.extend_from_slice(&5i32.to_le_bytes());
        let mut rec = vec![0u8; 0x8c];
        let f = |v: f32| v.to_le_bytes();
        rec[0x00..0x0c].copy_from_slice(&[f(-25.28), f(0.31), f(-71.20)].concat());
        rec[0x0c..0x18].copy_from_slice(&[f(0.51), f(0.74), f(-0.44)].concat());
        rec[0x24..0x30].copy_from_slice(&[f(0.8), f(0.8), f(0.8)].concat());
        rec[0x3c..0x44].copy_from_slice(&[f(67.75), f(67.75)].concat());
        rec[0x58..0x5a].copy_from_slice(&1u16.to_le_bytes());
        rec[0x6c..0x70].copy_from_slice(&f(0.5));
        data.extend_from_slice(&rec);

        let mut rec2 = vec![0u8; 0x8c];
        rec2[0x0c..0x18].copy_from_slice(&[f(0.51), f(0.74), f(-0.44)].concat());
        rec2[0x24..0x30].copy_from_slice(&[f(0.28), f(0.33), f(0.40)].concat());
        rec2[0x58..0x5a].copy_from_slice(&5u16.to_le_bytes());
        rec2[0x6c..0x70].copy_from_slice(&f(1.0));
        data.extend_from_slice(&rec2);

        let lights = parse(&data);
        assert_eq!(lights.len(), 2);
        assert_eq!(lights[0].kind, RtlLightKind::Ambient);
        assert_eq!(lights[1].kind, RtlLightKind::Directional);

        // Mesh at the ambient light's center: full-strength ambient 0.8*0.5,
        // the CAMDIR light always included.
        let set = compute_light_set(&lights, lights[0].pos);
        assert!((set.scene_ambient[0] - 0.4).abs() < 1e-4);
        assert!((set.light_pos[0][3] - 1.0).abs() < 1e-4);
        assert!((set.light_color[0][0] - 0.28).abs() < 1e-4);
    }

    #[test]
    fn point_light_falls_off() {
        let mut rec = vec![0u8; 0x8c];
        let f = |v: f32| v.to_le_bytes();
        rec[0x00..0x0c].copy_from_slice(&[f(0.0), f(0.0), f(0.0)].concat());
        rec[0x24..0x30].copy_from_slice(&[f(1.0), f(1.0), f(1.0)].concat());
        rec[0x3c..0x44].copy_from_slice(&[f(2.0), f(6.0)].concat());
        rec[0x58..0x5a].copy_from_slice(&2u16.to_le_bytes()); // POINT
        rec[0x6c..0x70].copy_from_slice(&f(1.0));
        let mut data = 0i32.to_le_bytes().to_vec();
        data.extend_from_slice(&rec);
        let lights = parse(&data);
        assert_eq!(lights.len(), 1);

        let near = compute_light_set(&lights, [0.0, 0.0, 1.0]); // d=1 <= radius
        assert!((near.light_pos[0][3] - 1.0).abs() < 1e-4);
        let mid = compute_light_set(&lights, [0.0, 0.0, 4.0]); // d=4: half ramp
        assert!((mid.light_pos[0][3] - 0.5).abs() < 1e-4);
        let far = compute_light_set(&lights, [0.0, 0.0, 7.0]); // d=7 > falloff
        assert_eq!(far.light_pos[0][3], 0.0);
    }
}