use anyhow::Result;
use glam::Vec3;
use rustt::ghg;

#[test]
fn correlate() -> Result<()> {
    let m = "backup/CHARS/ANAKIN/ANAKIN_PADAWAN_PC.GHG";
    let data = std::fs::read(m)?;
    let parsed = ghg::parse(&data)?;

    // bind positions = translation of b.bind
    let bind_pos: Vec<Vec3> = parsed
        .bones
        .iter()
        .map(|b| b.bind.w_axis.truncate())
        .collect();
    for (bone, p) in bind_pos.iter().enumerate() {
        println!("bone {bone:2} {:>16} at ({:+.3},{:+.3},{:+.3})", parsed.bones[bone].name, p.x, p.y, p.z);
    }

    for part_i in [5usize, 7, 8, 9] {
        let p = &parsed.parts[part_i];
        let vl = parsed.vertex_lists[p.vl];
        let base = p.off_v * p.stride;
        let n = p.num_v;
        let sk = if p.stride == 44 { 36 } else if p.stride == 40 { 32 } else { continue };
        // bone list from descriptor
        // (recompute desc not needed; use known list — but let's read it generically)
        let mut bones = Vec::<u8>::new();
        for k in 0..8 {
            let b = vl.get(base + k * p.stride + p.stride - 1 - 8).copied().unwrap_or(0);
            let _ = b;
        }
        // decode bones from descriptor by re-reading offsets: easier to skip.

        // geometric nearest bone among candidate set (bones with channels / all)
        let cand: Vec<usize> = (0..parsed.bones.len()).collect();
        let mut agree = [0usize; 8];
        let mut total = 0;
        let mut nearest_dist = 0.0f32;
        for v in 0..n {
            let o = base + v * p.stride;
            let pos = Vec3::new(
                f32::from_le_bytes(vl[o..o + 4].try_into().unwrap()),
                f32::from_le_bytes(vl[o + 4..o + 8].try_into().unwrap()),
                f32::from_le_bytes(vl[o + 8..o + 12].try_into().unwrap()),
            );
            // nearest bone by distance to bind translation
            let mut best = 0usize;
            let mut bd = f32::MAX;
            for &b in &cand {
                let d = (pos - bind_pos[b]).length();
                if d < bd {
                    bd = d;
                    best = b;
                }
            }
            nearest_dist = bd;
            // For each skin byte, compare value to `best` and to (best - some offset)
            let bytes = &vl[o + sk..o + sk + 8];
            // agreement: does this byte equal best? (or best adjusted)
            for k in 0..8 {
                if bytes[k] as usize == best {
                    agree[k] += 1;
                }
            }
            // also try: does byte match best - bind_base where base = first bone of part?
            let _ = nearest_dist;
            let _ = bytes;
        }
        total = n;
        let _ = total;
        println!("part {part_i}: geometric-agreement per skin byte (byte==bone index):");
        for k in 0..8 {
            println!("   byte {k}: {}/{}", agree[k], n);
        }
    }
    Ok(())
}
