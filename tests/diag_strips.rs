use anyhow::Result;
use rustt::ghg;

#[test]
fn find_strips() -> Result<()> {
    let m = "backup/CHARS/ANAKIN/ANAKIN_PADAWAN_PC.GHG";
    let data = std::fs::read(m)?;
    let parsed = ghg::parse(&data)?;

    for part_i in [5usize, 6, 7, 8] {
        let p = &parsed.parts[part_i];
        let vl = parsed.vertex_lists[p.vl];
        let base = p.off_v * p.stride;
        let n = p.num_v;
        // strip bones from descriptor
        // recompute desc
        // (reuse earlier approach: read from parsed.parts? descriptor needed)
        let mut pos = Vec::with_capacity(n);
        for v in 0..n {
            let o = base + v * p.stride;
            pos.push([
                f32::from_le_bytes(vl[o..o + 4].try_into().unwrap()),
                f32::from_le_bytes(vl[o + 4..o + 8].try_into().unwrap()),
                f32::from_le_bytes(vl[o + 8..o + 12].try_into().unwrap()),
            ]);
        }
        // find spatial jumps between consecutive vertices
        let mut jumps = Vec::new();
        for v in 1..n {
            let d = ((pos[v][0] - pos[v - 1][0]).powi(2)
                + (pos[v][1] - pos[v - 1][1]).powi(2)
                + (pos[v][2] - pos[v - 1][2]).powi(2))
            .sqrt();
            if d > 0.05 {
                jumps.push((v, d));
            }
        }
        println!("part {part_i}: nv={n} jumps>0.05: {}", jumps.len());
        for (v, d) in jumps.iter().take(40) {
            println!("   vert {v}: dist {d:.3} pos=({:.3},{:.3},{:.3})",
                pos[*v][0], pos[*v][1], pos[*v][2]);
        }
    }
    Ok(())
}
