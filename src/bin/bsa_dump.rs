//! Inspect BSA blend-shape animation files (see `rustt::bsa`).
//!
//!   bsa_dump <file.BSA> [--bake <channel>] [--json]
//!
//! Prints the header and per-channel summary; `--bake` also prints the sampled
//! per-frame weight for one channel. Useful for eyeballing against the
//! reference Tt-Games-Blend-Shape-Animation-Addon.

use anyhow::{Context, Result};
use rustt::bsa::{Bsa, KEY_BOOLEAN, KEY_COMPRESSED, KEY_FULL, KEY_NONE};

fn type_name(t: u8) -> &'static str {
    match t {
        KEY_NONE => "none",
        KEY_FULL => "full",
        KEY_COMPRESSED => "compressed",
        KEY_BOOLEAN => "boolean",
        _ => "unknown",
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .context("usage: bsa_dump <file.BSA> [--bake <channel>] [--json]")?;
    let data = std::fs::read(path).with_context(|| format!("reading {path}"))?;
    let bsa = Bsa::parse(&data).with_context(|| format!("parsing {path}"))?;

    let bake_idx = args
        .iter()
        .position(|a| a == "--bake")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<usize>().ok());

    let json = args.iter().any(|a| a == "--json");

    if json {
        println!(
            "{{\"frames\":{},\"groups\":{},\"channels_per_group\":{},\"intervals\":{},\"flags\":{:?},\"total_channels\":{}}}",
            bsa.length_in_frames,
            bsa.group_count,
            bsa.channels_per_group,
            bsa.interval_count,
            bsa.flags,
            bsa.total_channels()
        );
        return Ok(());
    }

    println!(
        "{}: frames={} groups={} channels/group={} intervals={} total_channels={} flags={:?}",
        path,
        bsa.length_in_frames,
        bsa.group_count,
        bsa.channels_per_group,
        bsa.interval_count,
        bsa.total_channels(),
        bsa.flags
    );
    println!();
    println!("=== channels ===");
    for (i, ch) in bsa.channels.iter().enumerate() {
        let extra = match ch.keyframe_type {
            KEY_NONE => format!("const={}", ch.constant_value),
            KEY_COMPRESSED => format!(
                "keys={} value_scale={} tangent_scale={}",
                ch.keys.len(),
                ch.value_scale,
                ch.tangent_scale
            ),
            KEY_FULL => format!("quads={}", ch.data.len() / 4),
            _ => String::new(),
        };
        println!("ch{i}: type={} {} ", type_name(ch.keyframe_type), extra);
    }

    if let Some(idx) = bake_idx {
        if idx >= bsa.channels.len() {
            anyhow::bail!("channel {idx} out of range (0..{})", bsa.channels.len());
        }
        println!();
        let baked = bsa.bake();
        let frames = baked[idx].len();
        println!("=== channel {idx} baked ({frames} frames) ===");
        let mut line = String::new();
        for (f, v) in baked[idx].iter().enumerate() {
            line.push_str(&format!("{v:.6} "));
            if (f + 1) % 16 == 0 {
                println!("{:3}: {}", f - 15, line);
                line.clear();
            }
        }
        if !line.is_empty() {
            println!("{:3}: {}", (frames - 1) / 16 * 16, line);
        }
    }
    Ok(())
}
