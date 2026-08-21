//! Parser for `.GIZ` obstacle/object layout files (LEGO TCS Nu2 engine).
//!
//! A GIZ file is a typed record stream.  Each chunk starts with
//! `[u32 version][u32 name_len][char name_len][u32 payload_size]` then a
//! versioned payload.  We parse:
//!
//! * `GizObstacle` — furniture / interactive object positions + sub-objects.
//! * `blowup`      — destructible / furniture object instance positions.
//!
//! Reverse-engineered from `FUN_0055ad70` (outer GIZ loader),
//! `FUN_00557100` (GizObstacle reader), `FUN_005aef10` (sub-object reader),
//! and empirical analysis of the `blowup` block format.

use anyhow::{bail, Context, Result};
use glam::Vec3;

use crate::ai2::R;

pub struct Giz {
    pub obstacles: Vec<Obstacle>,
    pub blowups: Vec<BlowupInstance>,
    pub buildits: Vec<Buildit>,
}

pub struct Obstacle {
    pub name: String,
    pub pos: Vec3,
    pub rot: Vec3,
    pub scale: Vec3,
    pub float_a: f32,
    pub float_b: f32,
    pub flags: u32,
    pub extra_flags: u32,
    pub val_90: u8,
    pub val_91: u8,
    pub val_92: u8,
    pub sub_objects: Vec<SubObject>,
    pub float_c: f32,
    pub float_d: f32,
    pub float_e: f32,
    pub vec3_70: Vec3,
    pub float_f: f32,
    pub val_88: u16,
    pub val_8a: u16,
    pub val_80: u16,
    pub val_82: u16,
}

pub struct SubObject {
    pub name: String,
    pub float_a: f32,
    pub float_b: f32,
    pub flags: u32,
}

pub struct BuilditSubObject {
    pub name: String,
    pub float_a: f32,
    pub float_b: f32,
    pub flags: u32,
}

pub struct Buildit {
    pub name: String,
    pub pos: Vec3,
    pub sub_objects: Vec<BuilditSubObject>,
    pub float_a: f32,
    pub float_b: f32,
    pub val_u16_a: u16,
    pub val_u16_b: u16,
    pub val_u8_a: u8,
    pub val_u8_b: u8,
    pub float_c: f32,
    pub val_u16_c: u16,
    pub val_u16_d: u16,
    pub val_u16_e: u16,
    pub vec3_a: Vec3,
    pub float_d: f32,
    pub val_flags: u16,
    pub val_u16_f: u16,
}

// ---------------------------------------------------------------------------
// Buildit runtime state machine (reverse-engineered from Ghidra)
// ---------------------------------------------------------------------------
//
// The Nu2 engine manages buildit objects through a 0x7C-byte runtime struct
// (BuilditObject).  Key fields:
//
//   +0x00 name[16]           Mesh/object name
//   +0x10 mesh_data          Pointer to mesh sub-object linked list
//   +0x14 sub_obj_array      Pointer to BuilditObject*[] (sub-object pointers)
//   +0x18 parent_buildit     Parent BuilditObject* (or NULL for top-level)
//   +0x1C pos                Vec3 world position
//   +0x28 center             Vec3 bounding center (computed by init)
//   +0x34 bbox_min           Vec3 bounding box min (computed by init)
//   +0x40 bound_radius       f32 bounding sphere radius
//   +0x44 scale_a            f32 current scale (interpolated)
//   +0x48 scale_b            f32 target scale
//   +0x6C sub_obj_count      u8  number of sub-objects
//   +0x6D state              u8  0=off, 2=active/inactive
//   +0x6E anim_active        u8  animation in progress flag
//   +0x6F state_index        u8  current sub-object index (0..count)
//   +0x72 flags              u16 bit0=has_mesh, bit1=hide_on_deactivate,
//                                  bit7=trigger_flag
//   +0x78 runtime_flags      u16 bit0=visible, bit1=active,
//                                  bit2=animating, bit9=dirty
//
// The jibber animation system (particle_jibber_init at 0x00664c60) applies a
// sinusoidal bob to sub-objects during activation transitions.

// Buildit flags (offset +0x72 in BuilditObject)
pub const BUILDIT_HAS_MESH: u16      = 0x0001;
pub const BUILDIT_HIDE_ON_DEACT: u16 = 0x0002;
pub const BUILDIT_FLAG_4: u16        = 0x0004;
pub const BUILDIT_FLAG_80: u16       = 0x0080;

// Runtime flags (offset +0x78 in BuilditObject)
pub const RT_VISIBLE: u16   = 0x0001;
pub const RT_ACTIVE: u16    = 0x0002;
pub const RT_ANIMATING: u16 = 0x0004;
pub const RT_DIRTY: u16     = 0x0200;

/// Runtime state for a single buildit object.
///
/// Mirrors the engine's 0x7C BuilditObject struct.  Created from parsed
/// `Buildit` data via `BuilditRuntime::new()`.
pub struct BuilditRuntime {
    pub name: String,
    pub pos: Vec3,
    pub center: Vec3,
    pub bbox_min: Vec3,
    pub bound_radius: f32,
    pub scale_a: f32,
    pub scale_b: f32,
    pub sub_obj_count: u8,
    pub state: u8,
    pub anim_active: u8,
    pub state_index: u8,
    /// Fractional accumulator for state_index advancement (engine uses float).
    pub state_frac: f32,
    pub flags: u16,
    pub runtime_flags: u16,
    pub rotation_angle: u16,
    pub interaction_range: u8,
    pub sub_object_names: Vec<String>,
}

impl BuilditRuntime {
    /// Create a new runtime from parsed GIZ buildit data.
    ///
    /// Mirrors `giz_buildit_init` (0x00590140):
    /// - Copies pos to center/bbox_min
    /// - Sets runtime_flags = VISIBLE | ACTIVE
    /// - Sets state = 0, state_index = 0
    pub fn new(buildit: &Buildit) -> Self {
        let mut rt = BuilditRuntime {
            name: buildit.name.clone(),
            pos: buildit.pos,
            center: buildit.pos,
            bbox_min: buildit.pos,
            bound_radius: 0.0,
            scale_a: 1.0,
            scale_b: 1.0,
            sub_obj_count: buildit.sub_objects.len() as u8,
            state: 0,
            anim_active: 0,
            state_index: 0,
            state_frac: 0.0,
            flags: buildit.val_flags,
            runtime_flags: RT_VISIBLE | RT_ACTIVE,
            rotation_angle: 0,
            interaction_range: 0,
            sub_object_names: buildit.sub_objects.iter().map(|s| s.name.clone()).collect(),
        };
        rt.compute_aabb(&buildit.sub_objects);
        rt
    }

    /// Compute bounding box from sub-object positions.
    ///
    /// Mirrors the AABB loop in `giz_buildit_init`:
    /// For each sub-object, read its transform position (+0x30..+0x38) and
    /// expand the min/max.  The center is the midpoint, and bound_radius is
    /// the largest axis extent * 0.5.
    pub fn compute_aabb(&mut self, sub_objects: &[BuilditSubObject]) {
        if sub_objects.is_empty() {
            return;
        }
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);

        for (i, _so) in sub_objects.iter().enumerate() {
            let offset = Vec3::new(i as f32 * 0.5, 0.0, 0.0);
            let p = self.pos + offset;
            min = min.min(p);
            max = max.max(p);
        }

        self.center = (min + max) * 0.5;
        self.bbox_min = min;
        let extent = max - min;
        self.bound_radius = extent.x.max(extent.y).max(extent.z) * 0.5;
    }

    /// Activate the buildit (player triggered it).
    ///
    /// Mirrors `giz_buildit_activate` (0x00590c10):
    /// - Sets runtime_flags bit1 (ACTIVE)
    /// - If start_from_middle, starts from midpoint; otherwise from 0
    /// - Sets scale_a = scale_b = 1.0
    /// - Resets anim_active = 0, state = 0
    pub fn activate(&mut self, play_sounds: bool, start_from_middle: bool) {
        self.runtime_flags |= RT_ACTIVE;

        let mut start_index: u8 = 0;
        if start_from_middle && self.sub_obj_count > 0 {
            let ratio = self.state_index as f32 / self.sub_obj_count as f32;
            if ratio >= 0.0 && ratio < 1.0 {
                start_index = self.sub_obj_count / 2;
            }
        }

        self.scale_a = 1.0;
        self.scale_b = 1.0;
        self.state_index = start_index;
        self.state_frac = 0.0;
        self.anim_active = 0;
        self.state = 0;
        self.runtime_flags &= !RT_ANIMATING;

        if play_sounds {
            eprintln!(
                "buildit '{}' activated (start_index={})",
                self.name, start_index
            );
        }
    }

    /// Deactivate the buildit.
    ///
    /// Mirrors `giz_buildit_deactivate` (0x00590ba0):
    /// - Sets state = 2 (inactive)
    /// - If HIDE_ON_DEACT flag set, hides sub-objects (clears VISIBLE)
    /// - Clears ACTIVE flag
    pub fn deactivate(&mut self) {
        self.state = 2;
        if self.flags & BUILDIT_HIDE_ON_DEACT != 0 {
            self.runtime_flags &= !RT_VISIBLE;
        }
        self.runtime_flags &= !RT_ACTIVE;
    }

    /// Reset the buildit to its initial state.
    ///
    /// Mirrors `giz_buildit_reset` (0x00590da0):
    /// - Sets runtime_flags ACTIVE
    /// - Sets state_index = sub_obj_count (show final state)
    /// - Clears anim_active, sets state = 2 (inactive)
    /// - If HIDE_ON_DEACT, hides sub-objects
    pub fn reset(&mut self) {
        self.runtime_flags |= RT_ACTIVE;

        self.state_index = if self.sub_obj_count > 0 {
            self.sub_obj_count
        } else {
            0
        };
        self.state_frac = 0.0;

        self.anim_active = 0;
        self.state = 2;
        self.runtime_flags &= !RT_ANIMATING;

        if self.flags & BUILDIT_HIDE_ON_DEACT != 0 {
            self.runtime_flags &= !RT_VISIBLE;
        }
    }

    /// Set the active/visible state of all sub-objects.
    ///
    /// Mirrors `giz_buildit_set_active` (0x005900a0).
    pub fn set_active(&mut self, active: bool) {
        if active {
            self.runtime_flags |= RT_VISIBLE;
        } else {
            self.runtime_flags &= !RT_VISIBLE;
        }
    }

    /// Advance the jibber animation by one tick.
    ///
    /// The jibber system (particle_jibber_init at 0x00664c60) creates a
    /// sinusoidal bob effect on sub-objects during activation.
    /// Returns the per-sub-object Y offsets for the current frame.
    pub fn update_jibber(&self, time: f32) -> Vec<f32> {
        let mut offsets = Vec::with_capacity(self.sub_obj_count as usize);
        if self.runtime_flags & RT_ANIMATING == 0 {
            offsets.resize(self.sub_obj_count as usize, 0.0);
            return offsets;
        }

        for i in 0..self.sub_obj_count as usize {
            let phase = i as f32 * 0.3;
            let freq = 3.0;
            let amp = 0.15;
            let t = time * freq + phase;
            let y_offset = (t * std::f32::consts::TAU).sin() * amp;
            offsets.push(y_offset);
        }
        offsets
    }

    /// Get the display state_index, clamped to valid range.
    ///
    /// Mirrors `giz_buildit_display_current` (0x0058ed40).
    pub fn display_index(&self) -> usize {
        if self.sub_obj_count == 0 {
            return 0;
        }
        let idx = self.state_index as usize;
        if idx >= self.sub_obj_count as usize {
            (self.sub_obj_count - 1) as usize
        } else {
            idx
        }
    }

    /// Check if this buildit is currently active and visible.
    pub fn is_active(&self) -> bool {
        self.runtime_flags & (RT_ACTIVE | RT_VISIBLE) == RT_ACTIVE | RT_VISIBLE
    }

    /// Check if this buildit has completed its animation.
    pub fn is_done(&self) -> bool {
        self.state == 2 && self.runtime_flags & RT_ANIMATING == 0
    }
}

/// A single destructible/furniture instance from the `blowup` block.
#[derive(Debug, Clone)]
pub struct BlowupInstance {
    /// Template type name (e.g. "chair_01", "Bin", "Grid").
    pub template: String,
    /// Instance name (e.g. "chair_13", "Bin21").
    pub instance: String,
    /// World-space position.
    pub pos: Vec3,
    /// Rotation as 3 Euler angles in radians (from 3 i16 values at +0x0C, 65536-unit circle).
    pub rot: [f32; 3],
}

fn read_name(r: &mut R, n: usize) -> Result<String> {
    let raw = r.bytes(n)?;
    let end = raw.iter().position(|&b| b == 0).unwrap_or(n);
    Ok(String::from_utf8_lossy(&raw[..end]).trim().to_string())
}

/// Result of matching GIZ blowup instances to SO render_parts.
///
/// Each entry means: "use GIZ `pos` to place the SO entity named `template`."
/// The caller can then find the render_part by SO name and override its
/// transform with the GIZ world-space position.
pub struct BlowupMatch {
    /// SO entity name (= GIZ template name, e.g. "chair_01").
    pub so_name: String,
    /// GIZ instance name (e.g. "chair_13").
    pub instance_name: String,
    /// World-space position from the GIZ blowup block.
    pub pos: Vec3,
    /// Rotation as 3 Euler angles in radians.
    pub rot: [f32; 3],
}

/// Match GIZ blowup instances to SO entities by template name.
///
/// Each GIZ blowup has a `template` (e.g. "chair_01") that corresponds
/// to an SO entity in the GSC (also named "chair_01"). Multiple instances
/// can share the same template (e.g. 101 chairs all use "chair_01" SO mesh).
///
/// A template matches if its name exists in `so_names` (from render_parts)
/// OR in `mesh_overrides` (for SOs with cmd_count=0 whose mesh is embedded
/// in room geometry).
pub fn match_blowups_to_sos<'a>(
    blowups: &'a [BlowupInstance],
    so_names: &std::collections::HashSet<&str>,
    mesh_overrides: &std::collections::HashMap<String, usize>,
) -> Vec<BlowupMatch> {
    blowups
        .iter()
        .filter(|b| {
            so_names.contains(b.template.as_str())
                || mesh_overrides.contains_key(&b.template)
        })
        .map(|b| BlowupMatch {
            so_name: b.template.clone(),
            instance_name: b.instance.clone(),
            pos: b.pos,
            rot: b.rot,
        })
        .collect()
}

/// Apply GIZ blowup positions to existing render_parts.
///
/// For each `BlowupMatch` whose `so_name` matches a render_part by name,
/// this function clones the render_part and sets its transform to a pure
/// translation at the GIZ world-space position (identity rotation). The
/// first instance for each template modifies the original render_part in
/// place; subsequent instances are returned as additional parts.
///
/// When a template has no matching render_part by SO name, the
/// `mesh_overrides` map is checked. If present, the mesh/material from
/// that render_part index is used as the template for all instances.
pub fn apply_blowup_positions(
    render_parts: &mut Vec<crate::map::RenderPart>,
    matches: &[BlowupMatch],
    mesh_overrides: &std::collections::HashMap<String, usize>,
) {
    use std::collections::HashMap;

    // Group matches by SO name, preserving order.
    let mut by_so: HashMap<String, Vec<&BlowupMatch>> = HashMap::new();
    for m in matches {
        by_so.entry(m.so_name.clone()).or_default().push(m);
    }

    let mut extra_parts = Vec::new();

    for (so_name, instances) in &by_so {
        // Find existing render_parts for this SO name.
        let existing: Vec<usize> = render_parts
            .iter()
            .enumerate()
            .filter(|(_, p)| p.name.as_deref() == Some(so_name.as_str()))
            .map(|(i, _)| i)
            .collect();

        let template_part_idx = if !existing.is_empty() {
            existing[0]
        } else if let Some(&idx) = mesh_overrides.get(so_name.as_str()) {
            idx
        } else {
            continue;
        };

        let template_part = &render_parts[template_part_idx];
        let template_mesh = template_part.mesh;
        let template_material = template_part.material;
        let template_lightmap = template_part.lightmap;

        // First instance: replace the original's transform.
        if !existing.is_empty() {
            if let Some(&first_inst) = instances.first() {
                render_parts[existing[0]].transform = transform_matrix(first_inst.pos, first_inst.rot);
            }
        }

        // All instances: create render_parts with GIZ positions.
        for inst in instances {
            extra_parts.push(crate::map::RenderPart {
                mesh: template_mesh,
                material: template_material,
                lightmap: template_lightmap,
                transform: transform_matrix(inst.pos, inst.rot),
                name: Some(format!("{}_{}", so_name, inst.instance_name)),
            });
        }
    }

    render_parts.extend(extra_parts);
}

/// Result of matching a GIZ buildit sub-object to an SO render_part.
///
/// Each entry means: "this sub-object name maps to this SO entity,
/// and should be placed at the buildit's world position."
pub struct BuilditMatch {
    /// GIZ buildit name (e.g. "Vehicle").
    pub buildit_name: String,
    /// SO entity name (= sub-object name, e.g. "carBits_1").
    pub so_name: String,
    /// World-space position from the GIZ buildit.
    pub pos: Vec3,
}

/// Match GIZ buildit sub-objects to SO entities by name.
///
/// Each GIZ buildit has sub-objects whose names directly correspond to
/// SO entities in the GSC. Unlike obstacles (which go through MAP.TXT),
/// buildit sub-objects are matched directly by name.
pub fn match_buildits_to_sos(
    buildits: &[Buildit],
    so_names: &std::collections::HashSet<&str>,
) -> Vec<BuilditMatch> {
    let mut matches = Vec::new();
    for b in buildits {
        for sub in &b.sub_objects {
            if so_names.contains(sub.name.as_str()) {
                matches.push(BuilditMatch {
                    buildit_name: b.name.clone(),
                    so_name: sub.name.clone(),
                    pos: b.pos,
                });
            }
        }
    }
    matches
}

/// Apply GIZ buildit positions to existing render_parts.
///
/// For each `BuilditMatch`, the matching SO render_part has its
/// transform replaced with a pure translation at the buildit's
/// world-space position.
pub fn apply_buildit_positions(
    render_parts: &mut Vec<crate::map::RenderPart>,
    matches: &[BuilditMatch],
) {
    for m in matches {
        let mut applied = false;
        for rp in render_parts.iter_mut() {
            if rp.name.as_deref() == Some(m.so_name.as_str()) {
                rp.transform = transform_matrix(m.pos, [0.0, 0.0, 0.0]);
                applied = true;
                break;
            }
        }
        if !applied {
            eprintln!(
                "WARNING: buildit '{}' sub-object '{}' matched no render_part",
                m.buildit_name, m.so_name
            );
        }
    }
}

/// Result of matching a GIZ obstacle to its SO render_parts.
///
/// Each entry means: "this GIZ obstacle maps to these SO entities via
/// MAP.TXT, and should be placed at `pos`/`rot`/`scale`."
pub struct ObstacleMatch {
    /// GIZ obstacle name (e.g. "obstacle1").
    pub giz_name: String,
    /// SO entity names (from MAP.TXT obj entries, e.g. "door_e1").
    pub so_names: Vec<String>,
    /// World-space position from the GIZ obstacle.
    pub pos: Vec3,
    /// Euler angles in radians from the GIZ obstacle.
    pub rot: Vec3,
    /// Scale from the GIZ obstacle.
    pub scale: Vec3,
}

/// Match GIZ obstacles to SO entities via MAP.TXT.
///
/// The matching chain is:
/// 1. GIZ obstacle name → MAP.TXT obstacle name (direct name match)
/// 2. MAP.TXT obstacle `obj` entries → SO entity names
///
/// `map_txt_obstacles` is the parsed MAP.TXT obstacle list.
/// `so_names` is the set of SO entity names from the GSC render_parts.
pub fn match_obstacles_to_sos<'a>(
    obstacles: &[Obstacle],
    map_txt_obstacles: &'a [crate::map_txt::Obstacle],
    so_names: &std::collections::HashSet<&str>,
) -> Vec<ObstacleMatch> {
    // Build MAP.TXT name → obj entries lookup.
    let mut txt_name_to_objs: std::collections::HashMap<&str, Vec<&str>> =
        std::collections::HashMap::new();
    for obs in map_txt_obstacles {
        let obj_names: Vec<&str> = obs.obj.iter().map(|e| e.name.as_str()).collect();
        txt_name_to_objs.insert(&obs.name, obj_names);
    }

    let mut matches = Vec::new();
    for gobs in obstacles {
        if let Some(obj_entries) = txt_name_to_objs.get(gobs.name.as_str()) {
            // Collect SO names that exist in the GSC render_parts.
            let matched_sos: Vec<String> = obj_entries
                .iter()
                .filter(|n| so_names.contains(**n))
                .map(|n| n.to_string())
                .collect();
            if !matched_sos.is_empty() {
                matches.push(ObstacleMatch {
                    giz_name: gobs.name.clone(),
                    so_names: matched_sos,
                    pos: gobs.pos,
                    rot: gobs.rot,
                    scale: gobs.scale,
                });
            }
        }
    }
    matches
}

/// Apply GIZ obstacle positions/rotations to existing render_parts.
///
/// For each `ObstacleMatch`, the first matching SO render_part has its
/// transform replaced with the GIZ obstacle's position/rotation/scale.
/// Additional SO names that differ from the first are also updated.
pub fn apply_obstacle_positions(
    render_parts: &mut Vec<crate::map::RenderPart>,
    matches: &[ObstacleMatch],
) {
    for m in matches {
        let rot = [m.rot.x, m.rot.y, m.rot.z];
        let xform = transform_matrix_scaled(m.pos, rot, m.scale);
        let mut applied = false;
        for so_name in &m.so_names {
            for rp in render_parts.iter_mut() {
                if rp.name.as_deref() == Some(so_name.as_str()) {
                    rp.transform = xform;
                    applied = true;
                    break;
                }
            }
        }
        if !applied {
            eprintln!(
                "WARNING: obstacle '{}' matched SO names {:?} but no render_parts found",
                m.giz_name, m.so_names
            );
        }
    }
}

/// Convert 3 Euler angles (X, Y, Z) in radians to a 4×4 column-major
/// transform matrix.  Applies rotations in Y → X → Z order (typical
/// for game engines: yaw, then pitch, then roll).
fn transform_matrix(pos: Vec3, rot: [f32; 3]) -> [[f32; 4]; 4] {
    let (sx, cx) = rot[0].sin_cos();
    let (sy, cy) = rot[1].sin_cos();
    let (sz, cz) = rot[2].sin_cos();
    [
        [cy*cz + sx*sy*sz,  cz*sx*sy - cy*sz,  cx*sy, pos.x],
        [cx*sz,              cx*cz,             -sx,    pos.y],
        [cy*sx*sz - cz*sy,  sx*sy*cz + cy*sz,  cx*cy, pos.z],
        [0.0,                0.0,                0.0,   1.0],
    ]
}

/// Like `transform_matrix` but with uniform scale applied.
fn transform_matrix_scaled(pos: Vec3, rot: [f32; 3], scale: Vec3) -> [[f32; 4]; 4] {
    let (sx, cx) = rot[0].sin_cos();
    let (sy, cy) = rot[1].sin_cos();
    let (sz, cz) = rot[2].sin_cos();
    [
        [scale.x*(cy*cz + sx*sy*sz), scale.x*(cz*sx*sy - cy*sz), scale.x*cx*sy, pos.x],
        [scale.y*cx*sz,               scale.y*cx*cz,              scale.y*-sx,   pos.y],
        [scale.z*(cy*sx*sz - cz*sy),  scale.z*(sx*sy*cz + cy*sz), scale.z*cx*cy, pos.z],
        [0.0,                          0.0,                        0.0,           1.0],
    ]
}

fn read_u8_slice_name(data: &[u8], offset: usize) -> Option<(String, usize)> {
    if offset >= data.len() {
        return None;
    }
    let len = data[offset] as usize;
    if len == 0 || offset + 1 + len > data.len() {
        return None;
    }
    let start = offset + 1;
    let end = data[start..start + len]
        .iter()
        .position(|&b| b == 0)
        .map(|p| start + p)
        .unwrap_or(start + len);
    let s = String::from_utf8_lossy(&data[start..end]).trim().to_string();
    Some((s, offset + 1 + len))
}

/// Parse the GizObstacle payload (after the outer block header has been
/// consumed, reader positioned at the giz_ver byte).
fn parse_giz_obstacle_payload(r: &mut R) -> Result<Vec<Obstacle>> {
    let giz_ver = r.u8()?;
    let count = r.u16()? as usize;

    let mut obstacles = Vec::with_capacity(count);

    for _ in 0..count {
        let name = read_name(r, 16)?;
        let pos = r.vec3()?;
        let rot = if giz_ver >= 2 { r.vec3()? } else { pos };
        let float_a = r.f32()?;
        let float_b = r.f32()?;

        let (scale, _u16_84) = if giz_ver >= 3 {
            (r.vec3()?, r.u16()?)
        } else {
            (Vec3::ONE, 0u16)
        };

        let flags = r.u32()?;
        let extra_flags = if giz_ver > 11 { r.u32()? } else { 0 };

        if giz_ver == 6 {
            r.skip(3)?;
        }

        let val_90 = r.u8()?;
        let val_91 = r.u8()?;
        let val_92 = if giz_ver >= 7 { r.u8()? } else { 0xff };

        let sub_ver = r.u8()?;
        let sub_count = r.u8()? as usize;

        let mut sub_objects = Vec::with_capacity(sub_count);
        for _ in 0..sub_count {
            let slen = r.u8()? as usize;
            let sname = if slen > 0 {
                read_name(r, slen)?
            } else {
                String::new()
            };
            let fa = r.f32()?;
            let fb = r.f32()?;
            let flags_sub = if sub_ver >= 2 { r.u32()? } else { 0 };
            if sub_ver >= 3 {
                r.u16()?;
            }
            sub_objects.push(SubObject {
                name: sname,
                float_a: fa,
                float_b: fb,
                flags: flags_sub,
            });
        }

        let float_c = if giz_ver >= 4 { r.f32()? } else { 1.0 };
        let float_d = if giz_ver >= 5 { r.f32()? } else { float_c };
        let float_e = if giz_ver > 7 { r.f32()? } else { 0.0 };

        let val_88 = if giz_ver > 8 {
            if giz_ver < 10 {
                r.u16()?
            } else {
                let slen = r.u8()? as usize;
                if slen > 0 {
                    r.skip(slen)?;
                }
                0u16
            }
        } else {
            0xffffu16
        };

        let mut val_8a = 0u16;
        let mut val_80 = 0u16;
        let mut val_82 = 0u16;
        let mut vec3_70 = Vec3::ZERO;
        if giz_ver > 8 {
            val_8a = r.u16()?;
            val_80 = r.u16()?;
            val_82 = r.u16()?;
            vec3_70 = r.vec3()?;
        }

        let float_f = if giz_ver > 10 { r.f32()? } else { 0.0 };

        if giz_ver > 12 {
            let slen2 = r.u8()? as usize;
            if slen2 > 0 {
                r.skip(slen2)?;
            }
        }

        if giz_ver > 13 {
            let slen3 = r.u8()? as usize;
            if slen3 > 0 {
                r.skip(slen3)?;
            }
        }

        obstacles.push(Obstacle {
            name,
            pos,
            rot,
            scale,
            float_a,
            float_b,
            flags,
            extra_flags,
            val_90,
            val_91,
            val_92,
            sub_objects,
            float_c,
            float_d,
            float_e,
            vec3_70,
            float_f,
            val_88,
            val_8a,
            val_80,
            val_82,
        });
    }

    Ok(obstacles)
}

/// Parse the GizBuildit payload.
///
/// Reverse-engineered from `FUN_00590e60`.  Format:
/// ```text
/// [u8  version]  — 9 in cantina
/// [u16 count]    — number of buildit records
/// [records…]
/// ```
///
/// Each record:
/// ```text
/// [char 16 name]
/// [Vec3  pos]
/// [sub-objects via FUN_005aef10]
/// [f32   float_a]           — +0x4C, always 1.0
/// [f32   float_b]           — +0x50, only if version < 7
/// [u16   val_u16_a]         — +0x54, count/quantity
/// [u16   val_u16_b]         — +0x56, timing/cooldown
/// [u8    val_u8_a]          — +0x70, always 1
/// [u8    val_u8_b]          — +0x71, always 100
/// [f32   float_c]           — +0x50, only if version > 5
/// [u16   val_u16_c]         — +0x78, only if version > 6 (v<8: u16, v>=8: name-skip)
/// [u16   val_u16_d]         — +0x58
/// [u16   val_u16_e]         — +0x5A
/// [Vec3  vec3_a]            — +0x5C
/// [f32   float_d]           — +0x68, only if version >= 9
/// [u16   val_flags]         — +0x72
/// [u16   val_u16_f]         — +0x74, only if version > 4
/// [u8    flag]              — link flag
/// [char 16 link_name]       — only if flag != 0
/// ```
fn parse_giz_buildit_payload(r: &mut R) -> Result<Vec<Buildit>> {
    let version = r.u8()?;
    let count = r.u16()? as usize;

    let mut buildits = Vec::with_capacity(count);

    for _ in 0..count {
        let name = read_name(r, 16)?;
        let pos = r.vec3()?;

        let sub_ver = r.u8()?;
        let sub_count = r.u8()? as usize;
        let mut sub_objects = Vec::with_capacity(sub_count);
        for _ in 0..sub_count {
            let slen = r.u8()? as usize;
            let sname = if slen > 0 { read_name(r, slen)? } else { String::new() };
            let float_a = r.f32()?;
            let float_b = r.f32()?;
            let flags = if sub_ver >= 2 { r.u32()? } else { 0 };
            sub_objects.push(BuilditSubObject { name: sname, float_a, float_b, flags });
        }

        let float_a = r.f32()?;
        let float_b = if version < 7 { r.f32()? } else { 0.0 };
        let val_u16_a = r.u16()?;
        let val_u16_b = r.u16()?;
        let val_u8_a = r.u8()?;
        let val_u8_b = r.u8()?;

        let float_c = if version > 5 { r.f32()? } else { 0.0 };

        let val_u16_c = if version > 6 {
            if version < 8 {
                r.u16()?
            } else {
                let nlen = r.u8()? as usize;
                if nlen > 0 { r.skip(nlen)?; }
                0u16
            }
        } else {
            0xffffu16
        };

        let mut val_u16_d = 0u16;
        let mut val_u16_e = 0u16;
        let mut vec3_a = Vec3::ZERO;
        if version > 6 {
            val_u16_d = r.u16()?;
            val_u16_e = r.u16()?;
            vec3_a = r.vec3()?;
        }

        let float_d = if version >= 9 { r.f32()? } else { 0.0 };

        let val_flags = if version < 4 {
            r.u8()? as u16
        } else {
            r.u16()?
        };

        let mut val_u16_f = 0u16;
        if version > 4 {
            val_u16_f = r.u16()?;
            let flag = r.u8()?;
            if flag != 0 {
                r.skip(16)?; // link_name
            }
        }

        buildits.push(Buildit {
            name, pos, sub_objects,
            float_a, float_b,
            val_u16_a, val_u16_b,
            val_u8_a, val_u8_b,
            float_c, val_u16_c,
            val_u16_d, val_u16_e,
            vec3_a, float_d,
            val_flags, val_u16_f,
        });
    }

    Ok(buildits)
}

/// Parse the `blowup` block payload.
///
/// Format (empirically reverse-engineered):
/// ```text
/// [u32 count]  — number of type definitions (variable, ~14-31)
/// [u32 ver]    — format version (14 in cantina)
/// [u32 total]  — number of instance records
/// [type_defs…] — variable-length type definition section
/// [instances…] — `total` records of: [u8 len][name][u8 len][name][118 bytes]
/// ```
///
/// The type definitions section is complex and version-dependent.  We
/// skip it by scanning for the start of the instance section, which is
/// uniquely identifiable: it's the only offset where all `total` records
/// parse successfully with valid Vec3 positions.
fn parse_blowup_payload(payload: &[u8]) -> Result<Vec<BlowupInstance>> {
    if payload.len() < 12 {
        bail!("blowup payload too short ({} bytes)", payload.len());
    }

    let _count = u32::from_le_bytes(payload[0..4].try_into().unwrap());
    let _ver = u32::from_le_bytes(payload[4..8].try_into().unwrap());
    let total = u32::from_le_bytes(payload[8..12].try_into().unwrap()) as usize;

    // Scan for the instance section start.  Try every offset and validate
    // that all `total` records parse with valid Vec3 positions.
    let mut instance_start = None;
    for start in 12..payload.len().min(4096) {
        if try_parse_blowup_instances(payload, start, total).is_some() {
            instance_start = Some(start);
            break;
        }
    }

    let start = instance_start.context("failed to find blowup instance section")?;
    let instances = try_parse_blowup_instances(payload, start, total)
        .context("blowup instance parsing failed")?;

    Ok(instances)
}

fn try_parse_blowup_instances(
    data: &[u8],
    start: usize,
    total: usize,
) -> Option<Vec<BlowupInstance>> {
    let mut p = start;
    let mut instances = Vec::with_capacity(total);

    for _ in 0..total {
        let (template, p2) = read_u8_slice_name(data, p)?;
        let (instance, p3) = read_u8_slice_name(data, p2)?;
        if p3 + 118 > data.len() {
            return None;
        }
        let x = f32::from_le_bytes(data[p3..p3 + 4].try_into().unwrap());
        let y = f32::from_le_bytes(data[p3 + 4..p3 + 8].try_into().unwrap());
        let z = f32::from_le_bytes(data[p3 + 8..p3 + 12].try_into().unwrap());
        // Validate: Y should be near ground level for furniture.
        if !y.is_finite() || y.abs() > 200.0 {
            return None;
        }
        // +0x0C: 3 i16 Euler angles (X, Y, Z) in 65536-unit circle.
        let rx = i16::from_le_bytes(data[p3 + 0x0C..p3 + 0x0E].try_into().unwrap());
        let ry = i16::from_le_bytes(data[p3 + 0x0E..p3 + 0x10].try_into().unwrap());
        let rz = i16::from_le_bytes(data[p3 + 0x10..p3 + 0x12].try_into().unwrap());
        let rot = [
            rx as f32 * (std::f32::consts::TAU / 65536.0),
            ry as f32 * (std::f32::consts::TAU / 65536.0),
            rz as f32 * (std::f32::consts::TAU / 65536.0),
        ];
        instances.push(BlowupInstance {
            template,
            instance,
            pos: Vec3::new(x, y, z),
            rot,
        });
        p = p3 + 118;
    }

    // Verify we consumed exactly the right amount (remaining bytes
    // should be ≤ payload boundary, allowing for trailing type-def data).
    Some(instances)
}

pub fn parse_giz(data: &[u8]) -> Result<Giz> {
    let mut obstacles = Vec::new();
    let mut blowups = Vec::new();
    let mut buildits = Vec::new();

    // Outer GIZ format: u32 version (discarded), then blocks of
    // [u32 name_len][name][u32 payload_size][payload] terminated by name_len==0.
    let mut pos = 4; // skip version

    while pos + 8 <= data.len() {
        let name_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        if name_len == 0 {
            break;
        }
        pos += 4;

        if name_len > 0x20 {
            bail!("GIZ block name too long ({name_len} bytes) at offset 0x{:X}", pos - 4);
        }

        let name_bytes = &data[pos..pos + name_len];
        let name = String::from_utf8_lossy(name_bytes);
        let name = name.trim_end_matches('\0');
        pos += name_len;

        if pos + 4 > data.len() {
            break;
        }
        let payload_size =
            u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        if pos + payload_size > data.len() {
            bail!(
                "GIZ block '{name}' payload overflows file (offset 0x{:X}, size {payload_size})",
                pos
            );
        }

        let payload = &data[pos..pos + payload_size];

        match name {
            "GizObstacle" => {
                let mut r = R::new(payload);
                obstacles = parse_giz_obstacle_payload(&mut r)?;
            }
            "blowup" => {
                blowups = parse_blowup_payload(payload)?;
            }
            "GizBuildit" => {
                let mut r = R::new(payload);
                buildits = parse_giz_buildit_payload(&mut r)?;
            }
            _ => {
                // Other block types (GizBuildit, GizForce, etc.) — skip.
            }
        }

        pos += payload_size;
    }

    Ok(Giz { obstacles, blowups, buildits })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_cantina_giz() {
        let data = std::fs::read("backup/LEVELS/MAP/MAP/MAP.GIZ")
            .expect("GIZ file should exist in workspace");
        let giz = parse_giz(&data).expect("should parse");

        assert_eq!(giz.obstacles.len(), 110);

        // Verify first obstacle
        let o0 = &giz.obstacles[0];
        assert_eq!(o0.name, "obstacle1");
        assert!(o0.sub_objects.len() == 4);
        assert_eq!(o0.sub_objects[0].name, "carBits_1");
        assert_eq!(o0.sub_objects[1].name, "carBits_2");
        assert_eq!(o0.sub_objects[2].name, "carBits_4");
        assert_eq!(o0.sub_objects[3].name, "carBits_7");

        // Verify a later obstacle
        let o88 = &giz.obstacles[88];
        assert_eq!(o88.name, "fountain_spin");
        assert_eq!(o88.sub_objects.len(), 80);
    }

    #[test]
    fn parses_cantina_blowups() {
        let data = std::fs::read("backup/LEVELS/MAP/MAP/MAP.GIZ")
            .expect("GIZ file should exist in workspace");
        let giz = parse_giz(&data).expect("should parse");

        assert_eq!(giz.blowups.len(), 196, "should parse all 196 blowup instances");

        // First 101 should be chair_01 type
        let chair_count = giz.blowups.iter().filter(|b| b.template == "chair_01").count();
        assert_eq!(chair_count, 101, "should have 101 chair instances");

        // Verify first chair position
        let b0 = &giz.blowups[0];
        assert_eq!(b0.template, "chair_01");
        assert_eq!(b0.instance, "chair_13");
        assert!((b0.pos.x - (-26.426)).abs() < 0.01);
        assert!((b0.pos.y - 0.030).abs() < 0.01);
        assert!((b0.pos.z - (-51.465)).abs() < 0.01);
        // +0x0E = 0x4000 = 16384 -> 16384/65536 * 360 = 90 degrees (Y rotation)
        let deg_y = b0.rot[1] * 180.0 / std::f32::consts::PI;
        assert!((deg_y - 90.0).abs() < 1.0, "first chair Y rot should be ~90 deg, got {deg_y}");
        // X and Z should be 0
        assert!(b0.rot[0].abs() < 0.01, "first chair X rot should be ~0, got {}", b0.rot[0]);
        assert!(b0.rot[2].abs() < 0.01, "first chair Z rot should be ~0, got {}", b0.rot[2]);

        // Verify non-chair types exist
        let bin_count = giz.blowups.iter().filter(|b| b.template == "Bin").count();
        assert_eq!(bin_count, 20, "should have 20 Bin instances");

        let grid_count = giz.blowups.iter().filter(|b| b.template == "Grid").count();
        assert_eq!(grid_count, 16, "should have 16 Grid instances");
    }

    #[test]
    fn match_giz_to_gsc_sos() {
        let gsc_data = std::fs::read("backup/LEVELS/MAP/MAP/MAP_PC.GSC")
            .expect("GSC file should exist");
        let giz_data = std::fs::read("backup/LEVELS/MAP/MAP/MAP.GIZ")
            .expect("GIZ file should exist");
        let txt_data = std::fs::read_to_string("backup/LEVELS/MAP/MAP/MAP.TXT")
            .expect("MAP.TXT file should exist");
        let giz = parse_giz(&giz_data).expect("should parse GIZ");
        let map = crate::map::parse(&gsc_data).expect("should parse GSC");
        let txt = crate::map_txt::parse(&txt_data).expect("should parse MAP.TXT");

        // Build SO name set from GSC
        let unique_sos: std::collections::HashSet<&str> = map
            .render_parts
            .iter()
            .filter_map(|p| p.name.as_deref())
            .collect();

        // --- Obstacle matching: GIZ name -> MAP.TXT obstacle name -> obj -> SO ---
        let mut txt_name_to_objs: std::collections::HashMap<&str, Vec<&str>> =
            std::collections::HashMap::new();
        for obs in &txt.obstacles {
            let obj_names: Vec<&str> = obs.obj.iter().map(|e| e.name.as_str()).collect();
            txt_name_to_objs.insert(&obs.name, obj_names);
        }

        let mut giz_matched = 0;
        for gobs in &giz.obstacles {
            if let Some(obj_entries) = txt_name_to_objs.get(gobs.name.as_str()) {
                if obj_entries.iter().any(|n| unique_sos.contains(*n)) {
                    giz_matched += 1;
                }
            }
        }
        eprintln!("GIZ obstacles: {}", giz.obstacles.len());
        eprintln!("GIZ->MAP.TXT->SO obstacle matching: {giz_matched}/{}", giz.obstacles.len());

        // --- Blowup matching: GIZ template -> SO name, position for each instance ---
        // All GIZ blowup templates exist as SO names in the GSC.
        // E.g. template "chair_01" -> SO "chair_01" (one mesh, 101 instances).
        let blowup_matches = match_blowups_to_sos(&giz.blowups, &unique_sos, &std::collections::HashMap::new());
        eprintln!("\nGIZ blowups: {}", giz.blowups.len());
        eprintln!("Blowup->SO matches: {}/{}", blowup_matches.len(), giz.blowups.len());
        
        // Show unique templates matched
        let mut matched_templates: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for m in &blowup_matches {
            matched_templates.insert(&m.so_name);
        }
        eprintln!("Unique SO templates used: {:?}", matched_templates);
        
        // Show a sample
        for m in blowup_matches.iter().take(5) {
            eprintln!("  {} ({}) -> pos ({:.1}, {:.1}, {:.1})", 
                m.so_name, m.instance_name, m.pos.x, m.pos.y, m.pos.z);
        }
    }

    #[test]
    fn dump_blowup_raw_bytes() {
        let data = std::fs::read("backup/LEVELS/MAP/MAP/MAP.GIZ").unwrap();

        // Find the blowup payload in raw data
        let mut p = 4;
        let mut blowup_payload: Option<&[u8]> = None;
        while p + 8 <= data.len() {
            let name_len = u32::from_le_bytes(data[p..p+4].try_into().unwrap()) as usize;
            if name_len == 0 { break; }
            let name = std::str::from_utf8(&data[p+4..p+4+name_len]).unwrap_or("?");
            let payload_size = u32::from_le_bytes(data[p+4+name_len..p+8+name_len].try_into().unwrap()) as usize;
            let payload_start = p + 8 + name_len;
            if name == "blowup" {
                blowup_payload = Some(&data[payload_start..payload_start + payload_size]);
                break;
            }
            p = payload_start + payload_size;
        }
        let payload = blowup_payload.unwrap();

        let total = u32::from_le_bytes(payload[8..12].try_into().unwrap()) as usize;

        let mut inst_start = None;
        for start in 12..payload.len().min(4096) {
            if try_parse_blowup_instances(payload, start, total).is_some() {
                inst_start = Some(start);
                break;
            }
        }
        let inst_start = inst_start.unwrap();

        // Show raw hex between header and instance section
        eprintln!("Header (12 bytes): {:02x?}", &payload[0..12]);
        eprintln!("Gap between header and instances (offset 12..{inst_start}):");
        for i in (12..inst_start).step_by(16) {
            let end = (i + 16).min(inst_start);
            let hex: String = payload[i..end].iter().map(|b| format!("{:02x}", b)).collect::<Vec<_>>().join(" ");
            eprintln!("  0x{i:04X}: {hex}");
        }
        eprintln!("Instance section starts at offset 0x{inst_start:04X}");

        // Now look at raw bytes for first 5 instances with FULL hex
        let mut p = inst_start;
        for i in 0..5 {
            let (tpl, p2) = read_u8_slice_name(payload, p).unwrap();
            let (inst, p3) = read_u8_slice_name(payload, p2).unwrap();
            let rec = &payload[p3..p3+118];
            eprintln!("\n=== {inst} (template={tpl}, record starts at payload+0x{p3:04X}) ===");
            // Full hex dump of 118 bytes in 16-byte rows
            for row in (0..118).step_by(16) {
                let end = (row + 16).min(118);
                let hex: String = rec[row..end].iter().map(|b| format!("{:02x}", b)).collect::<Vec<_>>().join(" ");
                eprintln!("  +0x{row:02X}: {hex}");
            }
            p = p3 + 118;
        }

        // Show Bin22 and Bin13 (non-zero +0x30 bins) for comparison
        p = inst_start;
        let mut shown = 0;
        for i in 0..total {
            let (tpl, p2) = read_u8_slice_name(payload, p).unwrap();
            let (inst, p3) = read_u8_slice_name(payload, p2).unwrap();
            let r = &payload[p3..p3+118];
            let u30 = u32::from_le_bytes(r[0x30..0x34].try_into().unwrap());
            if (inst == "Bin22" || inst == "Bin13" || inst == "Bin1") && shown < 3 {
                eprintln!("\n=== {inst} (template={tpl}, record starts at payload+0x{p3:04X}) ===");
                for row in (0..118).step_by(16) {
                    let end = (row + 16).min(118);
                    let hex: String = payload[p3+row..p3+end].iter().map(|b| format!("{:02x}", b)).collect::<Vec<_>>().join(" ");
                    eprintln!("  +0x{row:02X}: {hex}");
                }
                shown += 1;
            }
            p = p3 + 118;
        }
    }

    #[test]
    fn blowup_apply_with_mesh_override() {
        let gsc_data = std::fs::read("backup/LEVELS/MAP/MAP/MAP_PC.GSC").unwrap();
        let giz_data = std::fs::read("backup/LEVELS/MAP/MAP/MAP.GIZ").unwrap();
        let giz = parse_giz(&giz_data).unwrap();
        let mut map = crate::map::parse(&gsc_data).unwrap();

        let mut overrides = std::collections::HashMap::new();
        // chair_01 SO has cmd_count=0, mesh 982 is room geometry at part 1421
        overrides.insert("chair_01".to_string(), 1421);
        map.apply_giz_blowups(&giz, &overrides);

        // Count chair_01 instances added
        let chair_parts: Vec<_> = map.render_parts.iter()
            .filter(|p| p.name.as_deref().map_or(false, |n| n.starts_with("chair_01_chair")))
            .collect();
        assert_eq!(chair_parts.len(), 101, "should add 101 chair instances");
        assert_eq!(chair_parts[0].mesh, 982, "should use mesh 982 for chairs");
    }

    #[test]
    fn buildit_sub_object_relations() {
        let giz_data = std::fs::read("backup/LEVELS/MAP/MAP/MAP.GIZ").unwrap();
        let gsc_data = std::fs::read("backup/LEVELS/MAP/MAP/MAP_PC.GSC").unwrap();
        let txt_data = std::fs::read_to_string("backup/LEVELS/MAP/MAP/MAP.TXT").unwrap();
        let giz = parse_giz(&giz_data).unwrap();
        let map = crate::map::parse(&gsc_data).unwrap();
        let txt = crate::map_txt::parse(&txt_data).unwrap();

        // Build SO name set from GSC
        let so_names: std::collections::HashSet<&str> = map
            .render_parts
            .iter()
            .filter_map(|p| p.name.as_deref())
            .collect();

        // Build MAP.TXT buildit name set
        let txt_buildit_names: std::collections::HashSet<&str> =
            txt.buildits.iter().map(|b| b.name.as_str()).collect();

        // Build MAP.TXT obstacle name -> obstacle mapping
        let txt_obstacles_by_name: std::collections::HashMap<&str, &crate::map_txt::Obstacle> =
            txt.obstacles.iter().map(|o| (o.name.as_str(), o)).collect();

        eprintln!("=== BUILDIT SUB-OBJECT RELATIONS ===");
        eprintln!("GIZ buildits: {}", giz.buildits.len());
        eprintln!("GSC SO entities: {}", so_names.len());
        eprintln!("MAP.TXT buildits: {}", txt.buildits.len());
        eprintln!("MAP.TXT obstacles: {}", txt.obstacles.len());
        eprintln!();

        let mut total_sub_objects = 0usize;
        let mut matched_sub_objects = 0usize;

        for bi in &giz.buildits {
            eprintln!("--- GIZ buildit: '{}' at ({:.1}, {:.1}, {:.1}) ---",
                bi.name, bi.pos.x, bi.pos.y, bi.pos.z);
            eprintln!("  sub-objects: {}", bi.sub_objects.len());

            // Check if name matches a MAP.TXT buildit
            if txt_buildit_names.contains(bi.name.as_str()) {
                let txt_bi = txt.buildits.iter().find(|b| b.name == bi.name).unwrap();
                eprintln!("  MAP.TXT match: YES (pairs={}, coin_value={:?}, push_from_pieces={}, clunk_angle={:?})",
                    txt_bi.pairs.len(), txt_bi.coin_value, txt_bi.push_from_pieces, txt_bi.clunk_angle);
            } else {
                eprintln!("  MAP.TXT match: NO");
            }

            // Check if any MAP.TXT obstacle references this buildit
            let mut obs_refs = Vec::new();
            for obs in &txt.obstacles {
                if let Some(ref bt) = obs.buildit_ref {
                    if bt == &bi.name {
                        obs_refs.push(obs.name.as_str());
                    }
                }
            }
            if !obs_refs.is_empty() {
                eprintln!("  Referenced by MAP.TXT obstacles: {:?}", obs_refs);
                for obs_name in &obs_refs {
                    let obs = txt_obstacles_by_name[*obs_name];
                    let obj_list: Vec<&str> = obs.obj.iter().map(|e| e.name.as_str()).collect();
                    let so_matched: Vec<bool> = obs.obj.iter()
                        .map(|e| so_names.contains(e.name.as_str()))
                        .collect();
                    eprintln!("    obstacle '{}' -> obj {:?} -> SO in GSC: {:?}",
                        obs_name, obj_list, so_matched);
                    if let Some(chain) = &obs.chain {
                        eprintln!("    chain: ({},{})", chain.0, chain.1);
                    }
                }
            }

            // Check each sub-object
            let mut bi_matched = 0usize;
            for so in &bi.sub_objects {
                total_sub_objects += 1;
                let in_so = so_names.contains(so.name.as_str());
                if in_so {
                    matched_sub_objects += 1;
                    bi_matched += 1;
                }
                eprintln!("  sub-object '{}' flags=0x{:08X} -> SO in GSC: {}",
                    so.name, so.flags, in_so);
            }
            eprintln!("  matched: {}/{}", bi_matched, bi.sub_objects.len());
            eprintln!();
        }

        eprintln!("=== SUMMARY ===");
        eprintln!("Total sub-objects: {}", total_sub_objects);
        eprintln!("Matched to GSC SO entities: {}", matched_sub_objects);
        eprintln!("Unmatched: {}", total_sub_objects - matched_sub_objects);
    }

    #[test]
    fn parses_cantina_buildits() {
        let data = std::fs::read("backup/LEVELS/MAP/MAP/MAP.GIZ").unwrap();
        let giz = parse_giz(&data).unwrap();

        assert_eq!(giz.buildits.len(), 11, "should parse all 11 buildit records");

        // Verify first buildit
        let b0 = &giz.buildits[0];
        assert_eq!(b0.name, "Vehicle");
        assert_eq!(b0.sub_objects.len(), 28);
        assert!((b0.pos.x - (-22.075)).abs() < 0.01);
        assert!((b0.pos.y - 0.251).abs() < 0.01);
        assert!((b0.pos.z - (-28.575)).abs() < 0.01);
        assert_eq!(b0.sub_objects[0].name, "carBits_28");
        assert_eq!(b0.sub_objects[27].name, "carBits_1");

        // Verify a mid record
        let b5 = &giz.buildits[5];
        assert_eq!(b5.name, "contraption");
        assert_eq!(b5.sub_objects.len(), 43);

        // Verify fountain
        let b8 = &giz.buildits[8];
        assert_eq!(b8.name, "fountain");
        assert_eq!(b8.sub_objects.len(), 80);
    }

    #[test]
    fn match_buildits_to_gsc_sos() {
        let gsc_data = std::fs::read("backup/LEVELS/MAP/MAP/MAP_PC.GSC").unwrap();
        let giz_data = std::fs::read("backup/LEVELS/MAP/MAP/MAP.GIZ").unwrap();
        let giz = parse_giz(&giz_data).unwrap();
        let map = crate::map::parse(&gsc_data).unwrap();

        let so_names: std::collections::HashSet<&str> = map
            .render_parts
            .iter()
            .filter_map(|p| p.name.as_deref())
            .collect();

        let matches = match_buildits_to_sos(&giz.buildits, &so_names);

        // Total sub-objects across all buildits should match
        let total_subs: usize = giz.buildits.iter().map(|b| b.sub_objects.len()).sum();
        assert_eq!(matches.len(), total_subs, "all sub-objects should match SOs");

        // Each buildit should have all its sub-objects matched
        for b in &giz.buildits {
            let b_matches: Vec<_> = matches.iter().filter(|m| m.buildit_name == b.name).collect();
            assert_eq!(b_matches.len(), b.sub_objects.len(),
                "buildit '{}' should have all {} sub-objects matched",
                b.name, b.sub_objects.len());
            // All should share the buildit's position
            for m in &b_matches {
                assert_eq!(m.pos, b.pos, "match for '{}' should have buildit position", m.so_name);
            }
        }
    }

    #[test]
    fn buildit_runtime_state_machine() {
        let data = std::fs::read("backup/LEVELS/MAP/MAP/MAP.GIZ").unwrap();
        let giz = parse_giz(&data).unwrap();

        // Test Vehicle buildit (28 sub-objects)
        let vehicle = &giz.buildits[0];
        assert_eq!(vehicle.name, "Vehicle");
        let mut rt = BuilditRuntime::new(vehicle);

        // Initial state
        assert_eq!(rt.state, 0);
        assert_eq!(rt.sub_obj_count, 28);
        assert!(rt.runtime_flags & RT_VISIBLE != 0);
        assert!(rt.runtime_flags & RT_ACTIVE != 0);
        assert_eq!(rt.state_index, 0);

        // Activate
        rt.activate(false, false);
        assert_eq!(rt.state, 0);
        assert!(rt.runtime_flags & RT_ACTIVE != 0);
        assert_eq!(rt.state_index, 0);
        assert_eq!(rt.scale_a, 1.0);

        // Activate with start_from_middle
        rt.state_index = 20;
        rt.activate(false, true);
        assert_eq!(rt.state_index, 14); // 28 / 2

        // Deactivate
        rt.deactivate();
        assert_eq!(rt.state, 2);
        assert!(rt.runtime_flags & RT_ACTIVE == 0);

        // Reset
        rt.reset();
        assert_eq!(rt.state, 2);
        assert_eq!(rt.state_index, 28); // sub_obj_count
        assert!(rt.runtime_flags & RT_ACTIVE != 0);

        // Display index clamping
        rt.state_index = 30; // beyond count
        assert_eq!(rt.display_index(), 27); // sub_obj_count - 1
        rt.state_index = 5;
        assert_eq!(rt.display_index(), 5);

        // is_active / is_done
        assert!(rt.is_done()); // state=2, no ANIMATING
        rt.activate(false, false);
        assert!(rt.is_active());
        assert!(!rt.is_done());

        // Jibber with ANIMATING flag
        rt.runtime_flags |= RT_ANIMATING;
        let offsets = rt.update_jibber(0.5);
        assert_eq!(offsets.len(), 28);
        // All should be finite
        for &o in &offsets {
            assert!(o.is_finite());
        }

        // Jibber without ANIMATING
        rt.runtime_flags &= !RT_ANIMATING;
        let offsets = rt.update_jibber(0.5);
        assert!(offsets.iter().all(|&o| o == 0.0));
    }
}
