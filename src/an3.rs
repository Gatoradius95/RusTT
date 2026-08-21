//! AN3 (ANI4/ANI3/ANI6/ANI8) skeletal animation format.
//!
//! Layout and math follow the community "TT Games Animation Addon"
//! (`parse_an3`/`apply_animation` in an3_reader.py) and EasyAN3's DumbAN3.py,
//! cross-checked against the R2D2_Idle.xlsx byte breakdown.
//!
//! Header (version byte at 0x00; 0x34 = '4' -> '4INA', uses the 0x20 identity
//! rotation logic):
//!   0x00  version ('4INA'/'6INA'/'8INA')
//!   0x04  u16  num_bones
//!   0x06  u16 ] time fields: margin-1 style (old_time, new_time)
//!   0x08  u16 ]
//!   0x0A  u16 ]
//!   0x1C  f32  base_add   (static table decode)
//!   0x20  f32  base_mul   (static table decode)
//!   0x24  u32  ptr movpar
//!   0x28  u32  ptr static
//!   0x2C  u32  ptr matrix
//!   0x30  u32  ptr movdata
//!   0x34  u32  ptr footer
//!   0x38  u32  ptr optional
//!   0x3C  name (cstring)
//!
//! static table: raw u16. Decoded value = raw * base_mul + base_add. These two
//! header floats bring the AN3's internal coordinate space onto the model.
//!
//! matrix: u16 per bone per channel (9 total: T xyz, R xyz, S xyz).
//!   value == 0x06  -> animated keyframe channel (type 06)
//!   value == 0x07  -> animated keyframe channel (type 07, 12-bit weights)
//!   value >= 0x10  -> static, table[ value - 0x10 ]
//!   otherwise       -> default (0.0 for T/R, 1.0 for S)
//!
//! movpar: {scale f32, offset f32} per animated channel, in scan order.
//! movdata: a sequence of "keyblocks", one per frame, length = keyblock_length
//!   bytes (== num_animated * 4 for type-06, or *8 where 07 blocks are present).
//!   A type-06 block is a raw u32 control value; adjacent control values A,B
//!   decode to 4 sampled values:
//!       base = A & 0xFF, weights = arithmetic between A and B, then
//!       value = sampled * scale + offset.
//!   A type-07 block holds 4 raw u16 (12-bit weight interpolation).
//! footer: 1 flag byte per bone (bit 0x20 = uses identity rotation for 4INA).

use anyhow::{bail, Result};
use glam::Mat4;

#[derive(Debug, Clone)]
pub struct An3 {
    pub num_bones: usize,
    /// Playback timeline length (header 0x0A, DumbAN3 "OriginalNumFrames"):
    /// the number of frames the game plays the animation over. The stored data
    /// (see `data_frames`) is linearly stretched across this timeline, see
    /// `remap_playhead`.
    pub num_frames: usize,
    /// Stored frame count (header 0x06, DumbAN3 "NumFrames"): the animation's
    /// actual data spans subframes `[0, data_frames)`. The curve loops at
    /// `data_frames - 1`, which equals frame 0. Subframes at/after this point
    /// are padding/hold and are not sampled.
    pub data_frames: usize,
    /// Playback time offset (header 0x0E): subtracted from the playhead before
    /// the data remap. Zero for all shipped clips; the game's evaluator applies
    /// it as `(0x06 - 1) * (time - 0x0E) / (0x0A - 1)`.
    pub time_offset: u16,
    pub num_moving: usize,
    pub keyblock_len: usize,
    /// 4INA: true -> apply the 0x20 identity-rotation logic.
    pub four_ina: bool,
    /// static table, already decoded with base_mul/base_add.
    pub statics: Vec<f32>,
    /// raw per-bone-per-channel matrix entry (0x06/0x07 animated, >=0x10 static
    /// index into `statics` offset by 0x10, else default).
    pub matrix: Vec<u16>,
    /// per animated channel: its bone*9+chan and (scale, offset).
    pub animated: Vec<usize>,
    pub movpar: Vec<(f32, f32)>,
    /// per animated channel: its type (0x06 or 0x07).
    pub channel_types: Vec<u16>,
    pub channel_offsets: Vec<usize>,
    /// all keyblocks, in row-major [frame][channel].
    pub blocks: Vec<Vec<BlockVal>>,
    /// footer flags, 1 byte per bone.
    pub footer: Vec<u8>,
}

#[derive(Clone, Copy, Debug)]
pub enum BlockVal {
    Six(u32),
    Seven([u16; 4]),
}

impl An3 {
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 0x40 {
            bail!("file too small for AN3 header");
        }
        let u16 = |o: usize| u16::from_le_bytes([data[o], data[o + 1]]);
        let u32 = |o: usize| u32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
        let f32 = |o: usize| {
            f32::from_bits(u32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]))
        };

        let version = [data[0], data[1], data[2], data[3]];
        let four_ina = version == *b"4INA";

        let num_bones = u16(0x04) as usize;
        let keyblock_len = u16(0x08) as usize;
        let num_frames = u16(0x0a).max(1) as usize;
        let data_frames = u16(0x06).max(1) as usize;
        let time_offset = u16(0x0e);

        let base_add = f32(0x1c);
        let base_mul = f32(0x20);

        let ptr_movpar = u32(0x24) as usize;
        let ptr_static = u32(0x28) as usize;
        let ptr_matrix = u32(0x2c) as usize;
        let ptr_movdata = u32(0x30) as usize;
        let ptr_footer = u32(0x34) as usize;

        if num_bones == 0 {
            bail!("zero bones ({num_bones})");
        }

        // static table -> decoded by base_mul/base_add
        let n_static = if ptr_matrix > ptr_static {
            (ptr_matrix - ptr_static) / 2
        } else {
            0
        };
        let statics = (0..n_static)
            .map(|i| u16(ptr_static + i * 2) as f32 * base_mul + base_add)
            .collect::<Vec<_>>();

        // matrix: classify each bone channel; store raw entry + animated list.
        let n_matrix = num_bones * 9;
        let mut matrix = Vec::with_capacity(n_matrix);
        let mut anim_index = vec![usize::MAX; n_matrix];
        let mut animated: Vec<usize> = Vec::new();
        let mut channel_types: Vec<u16> = Vec::new();
        let mut channel_offsets: Vec<usize> = Vec::new();
        let mut movpar: Vec<(f32, f32)> = Vec::new();
        let mut curve_cursor = 0usize;
        for bone in 0..num_bones {
            for ch in 0..9 {
                let value = u16(ptr_matrix + (bone * 9 + ch) * 2);
                matrix.push(value);
                if value == 0x06 || value == 0x07 {
                    let scale = f32(ptr_movpar + animated.len() * 8);
                    let offset = f32(ptr_movpar + animated.len() * 8 + 4);
                    movpar.push((scale, offset));
                    anim_index[bone * 9 + ch] = animated.len();
                    animated.push(bone * 9 + ch);
                    channel_types.push(value);
                    channel_offsets.push(curve_cursor);
                    curve_cursor += if value == 0x06 { 4 } else { 8 };
                }
            }
        }
        let num_moving = animated.len();

        // footer flags
        let footer = data[ptr_footer..ptr_footer + num_bones].to_vec();

        // keyblocks
        let mut blocks: Vec<Vec<BlockVal>> = Vec::new();
        if !animated.is_empty() && keyblock_len > 0 {
            let block_count = (ptr_footer - ptr_movdata) / keyblock_len.max(1);
            for b in 0..block_count {
                let group = ptr_movdata + b * keyblock_len;
                let mut row = Vec::with_capacity(num_moving);
                for i in 0..num_moving {
                    let pos = group + channel_offsets[i];
                    if channel_types[i] == 0x06 {
                        row.push(BlockVal::Six(u32(pos)));
                    } else {
                        row.push(BlockVal::Seven([
                            u16(pos),
                            u16(pos + 2),
                            u16(pos + 4),
                            u16(pos + 6),
                        ]));
                    }
                }
                blocks.push(row);
            }
            // Decode produces one output frame per adjacent (A,B) pair, so pad
            // to an even block count.
            if blocks.len() % 2 != 0 {
                let last = blocks[blocks.len() - 1].clone();
                blocks.push(last);
            }
        }

        Ok(Self {
            num_bones,
            num_frames: num_frames.max(0),
            data_frames,
            time_offset,
            num_moving,
            keyblock_len,
            four_ina,
            statics,
            matrix,
            animated,
            movpar,
            channel_types,
            channel_offsets,
            blocks,
            footer,
        })
    }

    /// Decode a type-06 block pair (A, B) into four sampled control values.
    fn decode_block(cur: &BlockVal, next: &BlockVal) -> [f32; 4] {
        let BlockVal::Six(a) = cur else { return [0.0; 4] };
        let BlockVal::Six(b) = next else { return [0.0; 4] };
        let base_a = (a & 0xff) as f32;
        let base_b = (b & 0xff) as f32;
        let control = a >> 8;
        let mut out = [0f32; 4];
        for i in 0..4 {
            let w = ((control >> (i * 6)) & 0x3f) as f32 / 63.0;
            out[i] = base_a + (base_b - base_a) * w;
        }
        out
    }

    /// Decode a type-07 block pair into four sampled control values.
    fn decode_block_07(cur: &BlockVal, next: &BlockVal) -> [f32; 4] {
        let BlockVal::Seven(a) = cur else { return [0.0; 4] };
        let BlockVal::Seven(b) = next else { return [0.0; 4] };
        let base_a = a[0] as f32;
        let base_b = b[0] as f32;
        let w4 = (a[1] >> 12) | ((a[2] & 0xf000) >> 8) | ((a[3] & 0xf000) >> 4);
        let weights = [
            (a[1] & 0x0fff) as f32 / 4095.0,
            (a[2] & 0x0fff) as f32 / 4095.0,
            (a[3] & 0x0fff) as f32 / 4095.0,
            w4 as f32 / 4095.0,
        ];
        [
            base_a + (base_b - base_a) * weights[0],
            base_a + (base_b - base_a) * weights[1],
            base_a + (base_b - base_a) * weights[2],
            base_a + (base_b - base_a) * weights[3],
        ]
    }

    /// Map a playback-frame `playhead` on the animation timeline
    /// `[0, num_frames)` to the stored data subframe range `[0, data_frames)`.
    ///
    /// Matches the game's evaluator (`NuAnimBuffEvaluate`, e.g. `FUN_005cdd50`):
    /// `data_playhead = (0x06 - 1) * (playhead - 0x0E) / (0x0A - 1)`, clamped to
    /// `[0, 0x06 - 1]`. The game plays each animation over the longer 0x0A
    /// timeline, linearly stretching the 0x06-frame data so the loop-closing
    /// data frame (`data_frames - 1`, which holds frame 0's pose) lands exactly
    /// on the last timeline frame (`num_frames - 1`). This makes the loop
    /// seamless. When 0x0A == 0x06 this is the identity.
    pub fn remap_playhead(&self, playhead: f32) -> f32 {
        let dst = self.num_frames.max(1) as f32;
        let src = self.data_frames.max(1) as f32;
        if dst <= 1.0 || src <= 1.0 {
            return 0.0;
        }
        let t = playhead - self.time_offset as f32;
        (t * (src - 1.0) / (dst - 1.0)).clamp(0.0, src - 1.0)
    }

    /// Evaluated channel value at fractional data-subframe `frame` (range
    /// `[0, data_frames)`; use `remap_playhead` to convert a timeline/playback
    /// frame first). For a static channel returns the (already base/scale
    /// decoded) static value; for an animated
    /// channel it samples the keyblock interpolation.
    pub fn channel_value(&self, bone: usize, chan: usize, frame: f32) -> f32 {
        let m = self.matrix[bone * 9 + chan];
        match m {
            0x06 | 0x07 => {
                // find animated ordinal for (bone, chan) by scan order
                let mut idx = 0usize;
                for &amt in &self.animated {
                    if amt == bone * 9 + chan {
                        return self.sample_channel(idx, frame);
                    }
                    idx += 1;
                }
                0.0
            }
            v if v >= 0x10 => {
                let si = (v - 0x10) as usize;
                if si < self.statics.len() {
                    self.statics[si]
                } else {
                    0.0
                }
            }
            _ => {
                if chan >= 6 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    /// Sample the animated channel at fractional data-subframe `frame` (range
    /// `[0, data_frames)`; use `remap_playhead` to convert a timeline/playback
    /// frame first).
    ///
    /// Reproduces the game's evaluator (`NuAnimBuffEvaluate`, e.g.
    /// `FUN_005cdd50`): the playhead is floored to the sample index and the
    /// value is linearly interpolated toward the next sample by the fractional
    /// remainder (`value = sample[i] + frac * (sample[i+1] - sample[i])`). The
    /// evaluator's index helper truncates toward zero on SSE2 CPUs and rounds
    /// on legacy x87; playheads here are non-negative, so truncation == floor.
    /// Each sample belongs to its own block pair, so a subframe-3 sample
    /// interpolates across the pair boundary into subframe 0 of the next pair
    /// (the tail block of the file provides that next pair).
    fn sample_channel(&self, idx: usize, frame: f32) -> f32 {
        let (scale, offset) = self.movpar[idx];
        let n = self.blocks.len();
        if n < 2 {
            return offset;
        }
        // The animation's real data spans `data_frames` subframes (header
        // 0x06); subframes at/after that are padding and must not be sampled
        // (in original game files they can interpolate toward a zero-padded
        // tail block, producing garbage).
        let decoded_max = (n as f32 - 1.0) * 4.0 - 1.0;
        let data_max = (self.data_frames as f32 - 1.0).min(decoded_max).max(0.0);
        let f = frame.clamp(0.0, data_max);

        let sample = |k: f32| -> f32 {
            let k = k.clamp(0.0, decoded_max);
            let pair = (k / 4.0).floor() as usize;
            let pair = pair.min(n - 2);
            let sub = (k as i64).rem_euclid(4) as usize;
            let a = &self.blocks[pair];
            let b = &self.blocks[pair + 1];
            let raw = if self.channel_types[idx] == 0x07 {
                Self::decode_block_07(&a[idx], &b[idx])
            } else {
                Self::decode_block(&a[idx], &b[idx])
            };
            raw[sub]
        };

        let i = f.floor();
        let frac = f - i;
        let v0 = sample(i);
        let v1 = sample(i + 1.0);
        (v0 + frac * (v1 - v0)) * scale + offset
    }

    /// Effective parent-relative rotation for `bone` at `frame`. For 4INA bones
    /// with footer flag 0x20 the AN3 rotation channels are stored as small
    /// deltas relative to the bone's static rest rotation, so the
    /// parent-relative rotation becomes `rest_local * R(anim)` (or just
    /// `rest_local` when the 0x01 flag is absent and the rotation should be
    /// replaced entirely). `rest_local` is the bone's static parent-relative
    /// rotation from the GHG `local` mat44. Non-0x20 bones use `R(anim)`
    /// directly (their static local rotation is the identity).
    ///
    /// The AN3 rotation channels are stored in the game's mirrored coordinate
    /// space (same mirror that requires negating the translation Z), so all
    /// three euler angles are negated before composing `Rz * Ry * Rx`.
    fn bone_rot(&self, bone: usize, frame: f32, rest_local: Option<&Mat4>) -> Mat4 {
        let r_anim = Mat4::from_rotation_z(-self.channel_value(bone, 5, frame))
            * Mat4::from_rotation_y(-self.channel_value(bone, 4, frame))
            * Mat4::from_rotation_x(-self.channel_value(bone, 3, frame));
        if self.uses_x20(bone) {
            match rest_local {
                Some(rl) => {
                    let rl = Mat4::from_mat3(glam::Mat3::from_mat4(*rl));
                    if self.footer.get(bone).map_or(false, |f| f & 0x01 != 0) {
                        rl * r_anim
                    } else {
                        rl
                    }
                }
                None => r_anim,
            }
        } else {
            r_anim
        }
    }

    /// Local matrix for `bone` at `frame`: T * R * S, where loc/rot come from
    /// the AN3 channels and scale is only read when the footer 0x08 flag is set.
    /// The AN3 channels are stored in the game's coordinate space, whose Z is
    /// mirrored against the GHG's static local matrices, so the translation Z
    /// and the rotation eulers are negated before building the matrix. For 4INA
    /// the parent-relative rotation is composed against the bone's static rest
    /// rotation (see `bone_rot`).
    pub fn bone_local(&self, bone: usize, frame: f32, rest_local: Option<&Mat4>) -> Mat4 {
        let t = glam::Vec3::new(
            self.channel_value(bone, 0, frame),
            self.channel_value(bone, 1, frame),
            -self.channel_value(bone, 2, frame),
        );
        let r = self.bone_rot(bone, frame, rest_local);
        let s = if self.scale_flag(bone) {
            glam::Vec3::new(
                self.channel_value(bone, 6, frame),
                self.channel_value(bone, 7, frame),
                self.channel_value(bone, 8, frame),
            )
        } else {
            glam::Vec3::ONE
        };
        Mat4::from_translation(t) * r * Mat4::from_scale(s)
    }

    /// World matrices for all bones at `frame`: world = parent_world * local.
    /// `parents` must have len == num_bones, root entries = -1. `rest_locals`
    /// holds the static parent-relative rotation mat44 per bone (used by the
    /// 4INA 0x20 logic; can be `Mat4::IDENTITY` when the character is not 4INA).
    pub fn bone_worlds(
        &self,
        parents: &[i32],
        rest_locals: &[Mat4],
        frame: f32,
    ) -> Result<Vec<Mat4>> {
        if parents.len() != self.num_bones {
            bail!(
                "parent count {} != bone count {}",
                parents.len(),
                self.num_bones
            );
        }
        let mut worlds = Vec::with_capacity(self.num_bones);
        for b in 0..self.num_bones {
            let local = self.bone_local(b, frame, rest_locals.get(b));
            let w = if parents[b] < 0 {
                local
            } else {
                worlds[parents[b] as usize] * local
            };
            worlds.push(w);
        }
        Ok(worlds)
    }

    /// Static/neutral value of a channel: the value the channel holds when the
    /// animation is at its rest/identity frame (static channels return their
    /// decoded static value, animated channels return their movpar offset).
    pub fn neutral(&self, bone: usize, chan: usize) -> f32 {
        let m = self.matrix.get(bone * 9 + chan).copied().unwrap_or(0);
        match m {
            0x06 | 0x07 => self
                .channel_index(bone * 9 + chan)
                .and_then(|i| self.movpar.get(i))
                .map_or(0.0, |m| m.1),
            v if v >= 0x10 => self.statics.get((v - 0x10) as usize).copied().unwrap_or(0.0),
            _ => {
                if chan >= 6 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    /// The neutral rotation matrix (composition logic shared with `bone_rot`,
    /// using neutral channel values).
    pub fn neutral_rot(&self, bone: usize, rest_local: Option<&Mat4>) -> Mat4 {
        let r_anim = Mat4::from_rotation_z(-self.neutral(bone, 5))
            * Mat4::from_rotation_y(-self.neutral(bone, 4))
            * Mat4::from_rotation_x(-self.neutral(bone, 3));
        if self.uses_x20(bone) {
            match rest_local {
                Some(rl) => {
                    let rl = Mat4::from_mat3(glam::Mat3::from_mat4(*rl));
                    if self.footer.get(bone).map_or(false, |f| f & 0x01 != 0) {
                        rl * r_anim
                    } else {
                        rl
                    }
                }
                None => r_anim,
            }
        } else {
            r_anim
        }
    }

    fn channel_index(&self, want: usize) -> Option<usize> {
        self.animated.iter().position(|&a| a == want)
    }

    /// The 0x20 identity-rotation flag for a bone (4INA only).
    pub fn uses_x20(&self, bone: usize) -> bool {
        self.four_ina && self.footer.get(bone).map_or(false, |f| f & 0x20 != 0)
    }
    /// Whether scale should be read from the AN3 for this bone (footer bit 0x08).
    pub fn scale_flag(&self, bone: usize) -> bool {
        self.footer.get(bone).map_or(false, |f| f & 0x08 != 0)
    }
}

/// World matrices for a crossfade between `a` at `frame_a` and `b` at
/// `frame_b`, with `t` in `[0,1]` (`0` = clip `a`, `1` = clip `b`).
///
/// Mirrors the game's two-buffer blend (`NuAnimBuffBlendTwo`,
/// `FUN_005ebb90`): both clips are evaluated and their per-bone local
/// transforms are blended before the parent chain is walked. Translations
/// and scales are lerped, rotations are slerped (shortest path). Both
/// clips must share the same bone count and hierarchy as `parents`.
pub fn blended_bone_worlds(
    a: &An3,
    b: &An3,
    parents: &[i32],
    rest_locals: &[Mat4],
    frame_a: f32,
    frame_b: f32,
    t: f32,
) -> Result<Vec<Mat4>> {
    if a.num_bones != b.num_bones || parents.len() != a.num_bones {
        bail!(
            "bone count mismatch for blend ({} vs {}, {} parents)",
            a.num_bones,
            b.num_bones,
            parents.len()
        );
    }
    let mut worlds = Vec::with_capacity(a.num_bones);
    for bone in 0..a.num_bones {
        let la = a.bone_local(bone, frame_a, rest_locals.get(bone));
        let lb = b.bone_local(bone, frame_b, rest_locals.get(bone));
        let (sa, ra, ta) = la.to_scale_rotation_translation();
        let (sb, rb, tb) = lb.to_scale_rotation_translation();
        let local = Mat4::from_scale_rotation_translation(
            sa.lerp(sb, t),
            ra.slerp(rb, t),
            ta.lerp(tb, t),
        );
        let w = if parents[bone] < 0 {
            local
        } else {
            worlds[parents[bone] as usize] * local
        };
        worlds.push(w);
    }
    Ok(worlds)
}
