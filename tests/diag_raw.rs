use anyhow::Result;
use rustt::ghg;

fn at_i32(d: &[u8], o: i64) -> i32 {
    i32::from_le_bytes(d[o as usize..o as usize + 4].try_into().unwrap())
}

fn rel(d: &[u8], q: i64) -> Result<i64> {
    Ok(q + at_i32(d, q) as i64)
}

#[test]
fn dump_raw_regions() -> Result<()> {
    let m = "backup/CHARS/ANAKIN/ANAKIN_PADAWAN_PC.GHG";
    let data = std::fs::read(m)?;
    let parsed = ghg::parse(&data)?;

    for (i, vl) in parsed.vertex_lists.iter().enumerate() {
        println!("vertex_list[{i}] len={} bytes", vl.len());
    }
    println!();
    for (i, il) in parsed.index_lists.iter().enumerate() {
        println!("index_list[{i}] len={} bytes", il.len());
    }

    let p = &parsed.parts;
    for idx in [5usize, 6, 7, 8, 16, 23, 34] {
        let part = &p[idx];
        let vlist = parsed.vertex_lists[part.vl];
        let byte_off = part.off_v * part.stride;
        println!(
            "\npart {idx} vl={} stride={} off_v={} nv={} byte_off={}",
            part.vl, part.stride, part.off_v, part.num_v, byte_off
        );
        // dump 3 vertices worth
        for v in 0..3 {
            let base = byte_off + v * part.stride;
            println!(
                "  v{v}: {:02x?}",
                &vlist[base..base + part.stride]
            );
        }
    }
    Ok(())
}
