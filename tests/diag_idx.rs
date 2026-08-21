use anyhow::Result;
use rustt::ghg;

#[test]
fn dump_idx() -> Result<()> {
    let m = "backup/CHARS/ANAKIN/ANAKIN_PADAWAN_PC.GHG";
    let data = std::fs::read(m)?;
    let parsed = ghg::parse(&data)?;

    // part 5: stride 44, nv=1424, ni=2294, vl=4, il=0
    let p = &parsed.parts[5];
    let il = parsed.index_lists[p.il];
    let istart = p.off_i * 2;
    println!(
        "part5: off_i={:#x} num_i={} il.len={}",
        p.off_i,
        p.num_i,
        il.len()
    );
    let rd = |k: usize| u16::from_le_bytes(il[istart + k * 2..istart + k * 2 + 2].try_into().unwrap());
    let mut line = String::new();
    for k in 0..80 {
        line.push_str(&format!("{:4} ", rd(k)));
    }
    println!("{line}");

    // Find the max index value to see strips (strip restart markers or ranges)
    // Print per-index: the value and whether it's a "new range" jump
    let mut prev = rd(0) as i32;
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut run_start = 0usize;
    let mut jumps = 0;
    for k in 1..p.num_i {
        let v = rd(k) as i32;
        if (v - prev).abs() > 1 {
            jumps += 1;
            if k - run_start > 1 {
                runs.push((run_start, k));
            }
            run_start = k;
        }
        prev = v;
    }
    println!("jumps={jumps} run candidates:");
    for (a, b) in &runs {
        println!("  [{a}..{b}) len={}", b - a);
    }
    Ok(())
}
