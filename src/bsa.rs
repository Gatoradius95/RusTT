//! BSA blend-shape (morph weight) animation format, as used by the TT Games
//! engines (same format as .ANI). Each channel drives one shape-key weight on a
//! mesh; the per-vertex morph deltas themselves live in the model file, not
//! here. This module parses the animation and evaluates per-channel weights.
//!
//! Layout and math follow "Tt-Games-Blend-Shape-Animation-Addon"
//! (`bsa_addon.py`: `parse_bsa`, `evaluate_channel`, `create_lookup_key`,
//! `find_key_index`, `hermite`), cross-checked byte-for-byte against the real
//! `backup/CHARS/ANAKIN/*.BSA` files.
//!
//! File header (all little-endian):
//!   0x00  u32  version (2)
//!   0x04  u32  global_adjust: absolute pointers = global_adjust + file offset
//!   0x08  u32  data_header_offset (absolute)
//!
//! Data header at `rel(data_header_offset)`:
//!   0x00  f32  length_in_frames
//!   0x04  u16  group_count
//!   0x06  u16  channels_per_group
//!   0x08  u16  interval_count        (one 32-frame block = 32 keyframe slots)
//!   0x0a  u16  unk_000a
//!   0x0c  u32  channels_offset (abs)
//!   0x10  u32  keyframe_types_offset (abs)
//!   0x14  u32  flags_offset (abs)
//!
//! total_channels = group_count * channels_per_group.
//! Channel table: one entry per channel — KEY_NONE (0) stores a constant f32,
//! KEY_FULL (1)/KEY_COMPRESSED (2) store an absolute pointer to a descriptor.
//! KEY_BOOLEAN (4) is treated as a constant 0.0 channel (matches the addon).
//!
//! Descriptor: { masks_offset u32, interval_offsets_offset u32, data_offset u32 }
//!   masks:        interval_count * 4 bytes (a 32-bit bitmask of keyframe slots)
//!   interval_offsets: interval_count * u16 cumulative keyframe counts
//!   KEY_COMPRESSED data: { tangent_scale f32, value_scale f32 } then
//!       keyframe_count * (s16 raw_value, s8 raw_tangent, u8 time)
//!   KEY_FULL data: (keyframe_count + 1) * 4 * f32, quads of
//!       (time, inverse_duration, value, rate)

use anyhow::{bail, ensure, Context, Result};

pub const KEY_NONE: u8 = 0;
pub const KEY_FULL: u8 = 1;
pub const KEY_COMPRESSED: u8 = 2;
pub const KEY_BOOLEAN: u8 = 4;

/// One shape-key animation channel.
#[derive(Debug, Clone)]
pub struct Channel {
    pub keyframe_type: u8,
    /// Constant value for KEY_NONE channels.
    pub constant_value: f32,
    /// interval_count * 4 bytes; each interval's 32 bits mark keyframe slots.
    pub keyframe_masks: Vec<[u8; 4]>,
    /// Cumulative keyframe counts per interval (interval_count u16s).
    pub interval_offsets: Vec<u16>,
    /// KEY_COMPRESSED: decoded keys `(time, value, tangent)`.
    pub keys: Vec<(f32, f32, f32)>,
    /// KEY_FULL: quads `(time, inverse_duration, value, rate)`.
    pub data: Vec<f32>,
    pub value_scale: f32,
    pub tangent_scale: f32,
}

impl Default for Channel {
    fn default() -> Self {
        Self {
            keyframe_type: KEY_NONE,
            constant_value: 0.0,
            keyframe_masks: Vec::new(),
            interval_offsets: Vec::new(),
            keys: Vec::new(),
            data: Vec::new(),
            value_scale: 0.0,
            tangent_scale: 0.0,
        }
    }
}

/// A parsed BSA file.
#[derive(Debug, Clone)]
pub struct Bsa {
    pub length_in_frames: f32,
    pub group_count: usize,
    pub channels_per_group: usize,
    pub interval_count: usize,
    pub unk_000a: u16,
    pub flags: Vec<u8>,
    pub channels: Vec<Channel>,
}

impl Bsa {
    pub fn total_channels(&self) -> usize {
        self.group_count * self.channels_per_group
    }

    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 0x30 {
            bail!("file too small for BSA header");
        }
        let mut r = Cursor::new(data);

        let version = r.u32()?;
        if version != 2 {
            // Only version 2 is known; be lenient about it but surface it.
            eprintln!("warning: unexpected BSA version {version}");
        }
        let global_adjust = r.u32()?;
        let data_header_offset = r.u32()? as i64;

        let rel = |p: i64| -> i64 { p - global_adjust as i64 };

        r.seek(rel(data_header_offset))?;
        let length_in_frames = r.f32()?;
        let group_count = r.u16()? as usize;
        let channels_per_group = r.u16()? as usize;
        let interval_count = r.u16()? as usize;
        let unk_000a = r.u16()?;
        let channels_offset = r.u32()? as i64;
        let keyframe_types_offset = r.u32()? as i64;
        let flags_offset = r.u32()? as i64;

        let total = group_count * channels_per_group;
        ensure!(group_count > 0, "zero groups");
        ensure!(channels_per_group > 0, "zero channels per group");

        // keyframe types
        r.seek(rel(keyframe_types_offset))?;
        let mut keyframe_types = Vec::with_capacity(total);
        for _ in 0..total {
            keyframe_types.push(r.u8()?);
        }

        // flags
        r.seek(rel(flags_offset))?;
        let mut flags = Vec::with_capacity(group_count);
        for _ in 0..group_count {
            flags.push(r.u8()?);
        }

        // channel table: KEY_NONE -> constant f32, else descriptor pointer.
        r.seek(rel(channels_offset))?;
        let mut channel_offsets: Vec<Option<u64>> = Vec::with_capacity(total);
        let mut constant_values: Vec<f32> = Vec::with_capacity(total);
        for &t in &keyframe_types {
            if t == KEY_NONE {
                constant_values.push(r.f32()?);
                channel_offsets.push(None);
            } else {
                channel_offsets.push(Some(r.u32()? as u64));
                constant_values.push(0.0);
            }
        }

        let mut channels = Vec::with_capacity(total);
        for (idx, &t) in keyframe_types.iter().enumerate() {
            let mut ch = Channel::default();
            if t == KEY_NONE {
                ch.keyframe_type = KEY_NONE;
                ch.constant_value = constant_values[idx];
                channels.push(ch);
                continue;
            }
            if t == KEY_BOOLEAN || (t != KEY_FULL && t != KEY_COMPRESSED) {
                // Unknown/boolean channels carry no data; treat as constant 0.
                ch.keyframe_type = KEY_NONE;
                ch.constant_value = 0.0;
                channels.push(ch);
                continue;
            }
            ch.keyframe_type = t;

            let desc = channel_offsets[idx].expect("non-none channel has offset");
            r.seek(rel(desc as i64))?;
            let masks_offset = r.u32()? as i64;
            let interval_offsets_offset = r.u32()? as i64;
            let data_offset = r.u32()? as i64;

            r.seek(rel(masks_offset))?;
            let mut keyframe_count = 0usize;
            ch.keyframe_masks.reserve(interval_count);
            for _ in 0..interval_count {
                let mut mask = [0u8; 4];
                for slot in mask.iter_mut() {
                    *slot = r.u8()?;
                }
                keyframe_count += (mask[0].count_ones()
                    + mask[1].count_ones()
                    + mask[2].count_ones()
                    + mask[3].count_ones()) as usize;
                ch.keyframe_masks.push(mask);
            }

            r.seek(rel(interval_offsets_offset))?;
            ch.interval_offsets.reserve(interval_count);
            for _ in 0..interval_count {
                ch.interval_offsets.push(r.u16()?);
            }

            r.seek(rel(data_offset))?;
            if t == KEY_COMPRESSED {
                ch.tangent_scale = r.f32()?;
                ch.value_scale = r.f32()?;
                ch.keys.reserve(keyframe_count);
                for _ in 0..keyframe_count {
                    let raw_value = r.s16()? as f32;
                    let raw_tangent = r.s8()? as f32;
                    let time = r.u8()? as f32;
                    ch.keys.push((time, raw_value * ch.value_scale, raw_tangent * ch.tangent_scale));
                }
            } else {
                ch.data.reserve((keyframe_count + 1) * 4);
                for _ in 0..(keyframe_count + 1) * 4 {
                    ch.data.push(r.f32()?);
                }
            }

            channels.push(ch);
        }

        Ok(Self {
            length_in_frames,
            group_count,
            channels_per_group,
            interval_count,
            unk_000a,
            flags,
            channels,
        })
    }

    /// Per-channel weight at fractional `frame` (in the animation's frame
    /// space). Channels are shape-key weights; KEY_NONE returns its constant.
    pub fn evaluate(&self, channel: usize, frame: f32) -> f32 {
        match self.channels.get(channel) {
            Some(ch) => evaluate_channel(ch, self.interval_count, frame),
            None => 0.0,
        }
    }

    /// Sample every channel on every whole frame `0..round(length_in_frames)`:
    /// `baked[channel][frame]`. Mirrors the addon's `bake_bsa`.
    pub fn bake(&self) -> Vec<Vec<f32>> {
        let total_frames = (self.length_in_frames.round().max(1.0)) as usize;
        self.channels
            .iter()
            .map(|ch| {
                (0..total_frames)
                    .map(|f| evaluate_channel(ch, self.interval_count, f as f32))
                    .collect()
            })
            .collect()
    }
}

/// Hermite spline evaluation (port of `hermite`).
fn hermite(
    position: f32,
    t_start: f32,
    t_end: f32,
    inverse_duration: f32,
    blend_start: f32,
    blend_end: f32,
    rate_start: f32,
    rate_end: f32,
) -> f32 {
    let mut u = (position - t_start) * inverse_duration;
    if u < 0.0 {
        u = 0.0;
    } else if u > 1.0 {
        u = 1.0;
    }
    let duration = t_end - t_start;
    let dsr = duration * rate_start;
    let der = duration * rate_end;
    let u2 = u * u;
    let u3 = u2 * u;
    (2.0 * u3 - 3.0 * u2 + 1.0) * blend_start
        + (-2.0 * u3 + 3.0 * u2) * blend_end
        + (u3 - 2.0 * u2 + u) * dsr
        + (u3 - u2) * der
}

/// Map a frame to (interval, subinterval, interval_mask) for keyframe lookup.
fn create_lookup_key(interval_count: usize, mut position: f32) -> (usize, usize, u32) {
    if position < 1.0 {
        position = 1.0;
    }
    let mut interval = ((position as usize - 1) >> 5) as i64;
    if interval >= interval_count as i64 {
        interval = interval_count as i64 - 1;
    }
    if interval < 0 {
        interval = 0;
    }
    let position_after_interval = position - ((interval as i32) << 5) as f32;
    let mut int_position = (position_after_interval - 1.0) as i32;
    if int_position < 0 {
        int_position = 0;
    } else if int_position > 31 {
        int_position = 31;
    }
    let subinterval = int_position >> 3;
    let interval_mask = (1u32 << (((int_position & 0x7) + 1) & 0x1F)) - 1;
    (interval as usize, subinterval as usize, interval_mask)
}

/// Index of the keyframe segment covering `frame` (port of `find_key_index`).
fn find_key_index(channel: &Channel, interval_count: usize, frame: f32) -> usize {
    let (interval, subinterval, interval_mask) = create_lookup_key(interval_count, frame);
    let mask = channel.keyframe_masks[interval];
    let mut keyframe = 0usize;
    for i in 0..subinterval {
        keyframe += mask[i].count_ones() as usize;
    }
    keyframe += (mask[subinterval] as u32 & interval_mask).count_ones() as usize;
    let index = channel.interval_offsets[interval] as i64 + keyframe as i64 - 1;
    index.max(0) as usize
}

/// Evaluate one channel at fractional `frame` (port of `evaluate_channel`).
pub fn evaluate_channel(channel: &Channel, interval_count: usize, frame: f32) -> f32 {
    if channel.keyframe_type == KEY_NONE {
        return channel.constant_value;
    }
    let index = find_key_index(channel, interval_count, frame);

    if channel.keyframe_type == KEY_COMPRESSED {
        let keys = &channel.keys;
        if keys.is_empty() {
            return 0.0;
        }
        if index >= keys.len() - 1 {
            return keys[keys.len() - 1].1;
        }
        let a = keys[index];
        let b = keys[index + 1];
        let duration = b.0 - a.0;
        if duration <= 0.0 {
            return a.1;
        }
        return hermite(frame, a.0, b.0, 1.0 / duration, a.1, b.1, a.2, b.2);
    }

    let data = &channel.data;
    let offset = index * 4;
    if offset + 7 >= data.len() {
        return data.get(offset + 2).copied().unwrap_or(0.0);
    }
    let start = [data[offset], data[offset + 1], data[offset + 2], data[offset + 3]];
    let end = [data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7]];
    hermite(frame.max(1.0), start[0], end[0], start[1], start[2], end[2], start[3], end[3])
}

/// Little-endian cursor with bounds-checked reads.
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn seek(&mut self, offset: i64) -> Result<()> {
        let o = usize::try_from(offset).context("negative seek offset")?;
        ensure!(o <= self.data.len(), "seek out of range ({o:#x})");
        self.pos = o;
        Ok(())
    }

    fn read(&mut self, n: usize) -> Result<&'a [u8]> {
        ensure!(self.pos + n <= self.data.len(), "read past end at {:#x}", self.pos);
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.read(1)?[0])
    }

    fn s8(&mut self) -> Result<i8> {
        Ok(self.read(1)?[0] as i8)
    }

    fn u16(&mut self) -> Result<u16> {
        let b = self.read(2)?;
        Ok(u16::from_le_bytes(b.try_into().unwrap()))
    }

    fn s16(&mut self) -> Result<i16> {
        let b = self.read(2)?;
        Ok(i16::from_le_bytes(b.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32> {
        let b = self.read(4)?;
        Ok(u32::from_le_bytes(b.try_into().unwrap()))
    }

    fn f32(&mut self) -> Result<f32> {
        let b = self.read(4)?;
        Ok(f32::from_le_bytes(b.try_into().unwrap()))
    }
}
