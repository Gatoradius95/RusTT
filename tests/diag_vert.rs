use anyhow::Result;
use rustt::ghg;

fn at_i32(d: &[u8], o: i64) -> i32 {
    i32::from_le_bytes(d[o as usize..o as usize + 4].try_into().unwrap())
}

fn rel(d: &[u8], q: i64) -> Result<i64> {
    Ok(q + at_i32(d, q) as i64)
}

#[test]
fn dump_verts() -> Result<()> {
    let m = "backup/CHARS/ANAKIN/ANAKIN_PADAWAN_PC.GHG";
    let data = std::fs::read(m)?;
    let parsed = ghg::parse(&data)?;

    let num20 = at_i32(&data, 0) as i64;
    let head = num20 + 4 + 4 + 0xc;
    let abs_gsnh = head + 16 + at_i32(&data, head + 12) as i64;
    let gsnh = (abs_gsnh - 12) as usize;
    let mesh_meta = rel(&data, gsnh as i64 + 16 + 4 + 0x28)?;
    let mm = mesh_meta as usize;
    let number_parts = at_i32(&data, (mm + 0x14) as i64) as usize;
    let mut part_pos = mm + 0x14 + 4 + 0x08;

    for i in 0..number_parts {
        let offset_part = at_i32(&data, part_pos as i64) as i64;
        let desc = (part_pos as i64 + offset_part) as usize;
        let num_i = at_i32(&data, (desc + 4) as i64) + 2;
        let stride = i16::from_le_bytes([data[desc + 8], data[desc + 9]]) as usize;
        let off_v = at_i32(&data, (desc + 0x14) as i64) as usize;
        let num_v = at_i32(&data, (desc + 0x18) as i64) as usize;
        let off_i = at_i32(&data, (desc + 0x1c) as i64) as usize;
        let il = at_i32(&data, (desc + 0x20) as i64) as usize;
        let vl = at_i32(&data, (desc + 0x24) as i64) as usize;

        let mut bones = Vec::new();
        for k in 0..8 {
            let b = data[desc + 0x0c + k];
            if b == 0xff {
                break;
            }
            bones.push(b);
        }
        println!(
            "part {i} stride={stride} nv={num_v} ni={num_i} bones={bones:?} off_v={off_v:#x} vl={vl} off_i={off_i:#x} il={il}"
        );
        part_pos += 4;
    }

    // Dump vertices of part 5 (stride 44) raw, first 6 vertices.
    println!("\n--- part 5 vertex raw (stride 44) ---");
    let vl5 = parsed.parts[5].vl;
    let off_v5 = parsed.parts[5].off_v;
    let vldata = parsed.vertex_lists[vl5];
    for v in 0..6 {
        let base = off_v5 + v * 44;
        let bytes = &vldata[base..base + 44];
        let f = |o: usize| f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        println!(
            "v{v}: pos=({:.4},{:.4},{:.4}) nrm=({:.3},{:.3},{:.3}) uv=({:.3},{:.3}) rest={:02x?}",
            f(0),
            f(4),
            f(8),
            f(12),
            f(16),
            f(20),
            f(24),
            f(28),
            &bytes[32..44]
        );
    }
    // dump part 2 (stride 36) for comparison
    println!("\n--- part 2 vertex raw (stride 36) ---");
    let vl2 = parsed.parts[2].vl;
    let off_v2 = parsed.parts[2].off_v;
    let vldata2 = parsed.vertex_lists[vl2];
    for v in 0..4 {
        let base = off_v2 + v * 36;
        let bytes = &vldata2[base..base + 36];
        let f = |o: usize| f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        println!(
            "v{v}: pos=({:.4},{:.4},{:.4}) nrm=({:.3},{:.3},{:.3}) uv=({:.3},{:.3}) rest={:02x?}",
            f(0),
            f(4),
            f(8),
            f(12),
            f(16),
            f(20),
            f(24),
            f(28),
            &bytes[32..36]
        );
    }
    Ok(())
}
