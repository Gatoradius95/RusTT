use anyhow::Result;
use rustt::ghg;

#[test]
fn analyze_skinbytes() -> Result<()> {
    let m = "backup/CHARS/ANAKIN/ANAKIN_PADAWAN_PC.GHG";
    let data = std::fs::read(m)?;
    let parsed = ghg::parse(&data)?;

    for part_i in [5usize, 6, 7, 8, 9, 16] {
        let p = &parsed.parts[part_i];
        let vl = parsed.vertex_lists[p.vl];
        let base = p.off_v * p.stride;
        let n = p.num_v;
        // skinning bytes at offset 36..44 for stride 44; stride 40 -> offset 32..40
        let sk = if p.stride == 44 { 36 } else if p.stride == 40 { 32 } else { continue };
        // collect distinct patterns of the 8 skin bytes, count occurrences
        use std::collections::BTreeMap;
        let mut counts: BTreeMap<Vec<u8>, usize> = BTreeMap::new();
        let mut bypos: [BTreeMap<u8, usize>; 8] = Default::default();
        for v in 0..n {
            let o = base + v * p.stride;
            let bytes: Vec<u8> = vl[o + sk..o + sk + 8].to_vec();
            *counts.entry(bytes.clone()).or_insert(0) += 1;
            for k in 0..8 {
                *bypos[k].entry(bytes[k]).or_insert(0) += 1;
            }
        }
        println!("part {part_i} stride={} nv={n} distinct 8-byte patterns: {}", p.stride, counts.len());
        for k in 0..8 {
            let vals: Vec<String> = bypos[k].iter().map(|(b, c)| format!("{:02x}:{c}", b)).collect();
            println!("   byte {sk}+{k}: {}", vals.join(" "));
        }
    }
    Ok(())
}
