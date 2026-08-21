use anyhow::{ensure, Result};

use crate::ghg::{Texture, TextureFmt};

fn expand565(v: u16) -> (u8, u8, u8) {
    let r = ((v >> 11) & 0x1f) as u32;
    let g = ((v >> 5) & 0x3f) as u32;
    let b = (v & 0x1f) as u32;
    (
        ((r * 255 + 15) / 31) as u8,
        ((g * 255 + 31) / 63) as u8,
        ((b * 255 + 15) / 31) as u8,
    )
}

fn lerp(a: u8, b: u8, num: u32, den: u32) -> u8 {
    ((a as u32 * (den - num) + b as u32 * num) / den) as u8
}

/// Decode a DXT1/BC1 payload (blocks row-major, 4x4) into RGBA8.
fn decode_dxt1(payload: &[u8], w: usize, h: usize, out: &mut [u8]) -> Result<()> {
    let bw = (w + 3) / 4;
    let bh = (h + 3) / 4;
    ensure!(
        payload.len() >= bw * bh * 8,
        "DXT1 payload too small: {} (need {})",
        payload.len(),
        bw * bh * 8
    );
    for by in 0..bh {
        for bx in 0..bw {
            let bo = (by * bw + bx) * 8;
            let c0 = u16::from_le_bytes(payload[bo..bo + 2].try_into().unwrap());
            let c1 = u16::from_le_bytes(payload[bo + 2..bo + 4].try_into().unwrap());
            let bits = u32::from_le_bytes(payload[bo + 4..bo + 8].try_into().unwrap());
            let (r0, g0, b0) = expand565(c0);
            let (r1, g1, b1) = expand565(c1);
            let four = c0 > c1;
            for py in 0..4usize {
                for px in 0..4usize {
                    let idx = (bits >> (2 * (py * 4 + px))) & 3;
                    let (r, g, b, a) = match idx {
                        0 => (r0, g0, b0, 255),
                        1 => (r1, g1, b1, 255),
                        2 if four => (lerp(r0, r1, 1, 3), lerp(g0, g1, 1, 3), lerp(b0, b1, 1, 3), 255),
                        3 if four => (lerp(r0, r1, 2, 3), lerp(g0, g1, 2, 3), lerp(b0, b1, 2, 3), 255),
                        2 => ((r0 as u32 / 2 + r1 as u32 / 2) as u8, (g0 as u32 / 2 + g1 as u32 / 2) as u8, (b0 as u32 / 2 + b1 as u32 / 2) as u8, 255),
                        // 3-color mode index 3 is "transparent" in the spec, but
                        // the game uploads DXT1 as opaque: index 3 is used for
                        // black features (eyes, visors, dark suit areas), so emit
                        // opaque black instead of a transparent hole.
                        _ => (0, 0, 0, 255),
                    };
                    let x = bx * 4 + px;
                    let y = by * 4 + py;
                    if x < w && y < h {
                        let o = (y * w + x) * 4;
                        out[o] = r;
                        out[o + 1] = g;
                        out[o + 2] = b;
                        out[o + 3] = a;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Decode a DXT5/BC3 payload (blocks row-major, 4x4) into RGBA8.
fn decode_dxt5(payload: &[u8], w: usize, h: usize, out: &mut [u8]) -> Result<()> {
    let bw = (w + 3) / 4;
    let bh = (h + 3) / 4;
    ensure!(
        payload.len() >= bw * bh * 16,
        "DXT5 payload too small: {} (need {})",
        payload.len(),
        bw * bh * 16
    );
    for by in 0..bh {
        for bx in 0..bw {
            let bo = (by * bw + bx) * 16;
            let a0 = payload[bo];
            let a1 = payload[bo + 1];
            let mut ab = [0u8; 8];
            ab[..6].copy_from_slice(&payload[bo + 2..bo + 8]);
            let abits = u64::from_le_bytes(ab);
            let c0 = u16::from_le_bytes(payload[bo + 8..bo + 10].try_into().unwrap());
            let c1 = u16::from_le_bytes(payload[bo + 10..bo + 12].try_into().unwrap());
            let bits = u32::from_le_bytes(payload[bo + 12..bo + 16].try_into().unwrap());
            let (r0, g0, b0) = expand565(c0);
            let (r1, g1, b1) = expand565(c1);
            let alpha = |i: u64| -> u8 {
                let ai = a0 as u64;
                let bi = a1 as u64;
                if a0 > a1 {
                    match i {
                        0 => a0,
                        1 => a1,
                        n => ((ai * (8 - n) + bi * (n - 1)) / 7) as u8,
                    }
                } else {
                    match i {
                        0 => a0,
                        1 => a1,
                        6 => 0,
                        7 => 255,
                        n => ((ai * (6 - n) + bi * (n - 1)) / 5) as u8,
                    }
                }
            };
            for py in 0..4usize {
                for px in 0..4usize {
                    let idx = (bits >> (2 * (py * 4 + px))) & 3;
                    let (r, g, b) = match idx {
                        0 => (r0, g0, b0),
                        1 => (r1, g1, b1),
                        2 => (lerp(r0, r1, 1, 3), lerp(g0, g1, 1, 3), lerp(b0, b1, 1, 3)),
                        _ => (lerp(r0, r1, 2, 3), lerp(g0, g1, 2, 3), lerp(b0, b1, 2, 3)),
                    };
                    let ai = (abits >> (3 * (py * 4 + px))) & 7;
                    let a = alpha(ai);
                    let x = bx * 4 + px;
                    let y = by * 4 + py;
                    if x < w && y < h {
                        let o = (y * w + x) * 4;
                        out[o] = r;
                        out[o + 1] = g;
                        out[o + 2] = b;
                        out[o + 3] = a;
                    }
                }
            }
        }
    }
    Ok(())
}

pub fn decode_rgba(w: usize, h: usize, fmt: TextureFmt, payload: &[u8]) -> Result<Vec<u8>> {
    let n = (w as u64)
        .checked_mul(h as u64)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| anyhow::anyhow!("texture dimensions overflow: {}x{}", w, h))?;
    let mut rgba = vec![0u8; n as usize];
    match fmt {
        TextureFmt::Dxt1 => decode_dxt1(payload, w, h, &mut rgba)?,
        TextureFmt::Dxt5 => decode_dxt5(payload, w, h, &mut rgba)?,
    }
    Ok(rgba)
}

pub fn decode(texture: &Texture) -> Result<Vec<u8>> {
    decode_rgba(texture.w, texture.h, texture.fmt, &texture.payload)
}
