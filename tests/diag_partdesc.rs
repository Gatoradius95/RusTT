use anyhow::Result;
use rustt::ghg;

fn at_i32(d: &[u8], o: i64) -> i32 {
    i32::from_le_bytes(d[o as usize..o as usize + 4].try_into().unwrap())
}

fn rel(d: &[u8], q: i64) -> Result<i64> {
    Ok(q + at_i32(d, q) as i64)
}

#[test]
fn dump_part_descs() -> Result<()> {
    let m = "backup/CHARS/ANAKIN/ANAKIN_PADAWAN_PC.GHG";
    let data = std::fs::read(m)?;
    let parsed = ghg::parse(&data)?;
    let nb = parsed.bones.len();

    let num20 = at_i32(&data, 0) as i64;
    let head = num20 + 4 + 4 + 0xc;
    let abs_gsnh = head + 16 + at_i32(&data, head + 12) as i64;
    let gsnh = (abs_gsnh - 12) as usize;
    let mesh_meta = rel(&data, gsnh as i64 + 16 + 4 + 0x28)?;
    let mm = mesh_meta as usize;
    let number_parts = at_i32(&data, (mm + 0x14) as i64) as usize;
    let mut part_pos = mm + 0x14 + 4 + 0x08;
    println!("mm={mm:#x} part_pos={part_pos:#x} parts={number_parts}");
    for i in 0..number_parts {
        let offset_part = at_i32(&data, part_pos as i64) as i64;
        let desc = (part_pos as i64 + offset_part) as usize;
        println!("--- part {i} desc@{desc:#x}");
        let mut hex = String::new();
        for k in 0..0x28 {
            hex.push_str(&format!("{:02x} ", data[desc + k]));
            if k % 8 == 7 {
                hex.push('\n');
            }
        }
        print!("{hex}");
        let bone_at = at_i32(&data, (desc + 0x00) as i64);
        let f0 = at_i32(&data, (desc + 0x0c) as i64);
        let f1 = at_i32(&data, (desc + 0x10) as i64);
        let f2 = at_i32(&data, (desc + 0x28) as i64);
        println!("  +00={bone_at:#x} +0c={f0:#x} +10={f1:#x} +28={f2:#x}");
        part_pos += 4;
    }
    let _ = nb;
    Ok(())
}
