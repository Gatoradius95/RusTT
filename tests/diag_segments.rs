use anyhow::Result;
use rustt::ghg;

fn bone_list(parsed: &ghg::Parsed) -> Vec<Vec<u8>> {
    // recompute per-part strip bone lists is done via descriptor; placeholder not used
    vec![]
}

#[test]
fn strip_segments() -> Result<()> {
    let m = "backup/CHARS/ANAKIN/ANAKIN_PADAWAN_PC.GHG";
    let data = std::fs::read(m)?;
    let parsed = ghg::parse(&data)?;

    for part_i in [5usize, 6, 7, 8, 9, 14, 16, 31, 32, 33, 34] {
        let p = &parsed.parts[part_i];
        let il = parsed.index_lists[p.il];
        let istart = p.off_i * 2;
        let n = p.num_i;
        let rd = |k: usize| -> u32 {
            u16::from_le_bytes(il[istart + k * 2..istart + k * 2 + 2].try_into().unwrap()) as u32
        };
        // Split on runs of 2+ identical consecutive indices.
        let mut segments = Vec::new();
        let mut seg_start = 0usize;
        let mut run = 0usize;
        for k in 1..n {
            if rd(k) == rd(k - 1) {
                run += 1;
            } else {
                if run >= 1 {
                    // boundary at k-run-1 (end of previous segment)
                    let seg_len = (k - run - 1) - seg_start;
                    if seg_len > 2 {
                        segments.push((seg_start, seg_start + seg_len));
                    }
                    seg_start = k - run - 1;
                }
                run = 0;
            }
        }
        if n - seg_start > 2 {
            segments.push((seg_start, n));
        }
        println!("part {part_i}: ni={n} segments={}", segments.len());
        for (a, b) in &segments {
            println!("   [{a}..{b}) len={} start_idx={} end_idx={}", b - a, rd(*a), rd(*b - 1));
        }
    }
    Ok(())
}
