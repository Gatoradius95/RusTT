use rustt::an3::An3;

#[test]
fn dump_run_an3() {
    let path = "backup/CHARS/ANAKIN/RUN.AN3";
    let data = std::fs::read(path).unwrap();
    println!("\n=== {path} len={} ===", data.len());
    let u16 = |o: usize| u16::from_le_bytes([data[o], data[o + 1]]);
    let u32 = |o: usize| u32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
    let f32 = |o: usize| f32::from_bits(u32(o));
    println!("version {:?}", &data[0..4]);
    println!("num_bones(0x04)   = {}", u16(0x04));
    println!("field(0x06)       = {}", u16(0x06));
    println!("keyblock_len(0x08)= {}", u16(0x08));
    println!("num_frames(0x0a)  = {}", u16(0x0a));
    println!("field(0x0c)       = {}", u16(0x0c));
    println!("base_add(0x1c)    = {}", f32(0x1c));
    println!("base_mul(0x20)    = {}", f32(0x20));
    println!("ptr_movpar(0x24)  = 0x{:x}", u32(0x24));
    println!("ptr_static(0x28)  = 0x{:x}", u32(0x28));
    println!("ptr_matrix(0x2c)  = 0x{:x}", u32(0x2c));
    println!("ptr_movdata(0x30) = 0x{:x}", u32(0x30));
    println!("ptr_footer(0x34)  = 0x{:x}", u32(0x34));
    println!("ptr_opt(0x38)     = 0x{:x}", u32(0x38));
    let n_static = (u32(0x2c) - u32(0x28)) / 2;
    println!("n_static          = {}", n_static);
    let movpar_len = (u32(0x28) as i64 - u32(0x24) as i64).unsigned_abs() / 8;
    println!("n_movpar          = {}", movpar_len);
    let keyblock_len = u16(0x08) as u32;
    let block_count = (u32(0x34) - u32(0x30)) / keyblock_len.max(1);
    println!("raw block_count   = {} (footer-movdata)/keyblock_len", block_count);

    let a = An3::parse(&data).unwrap();
    println!("\nparsed: {} bones, {} frames, {} moving, keyblock_len={}, blocks.len={}",
        a.num_bones, a.num_frames, a.num_moving, a.keyblock_len, a.blocks.len());
    println!("channel_types: {:?}", a.channel_types);
    println!("channel_offsets: {:?}", a.channel_offsets);
    println!("movpar (scale,offset): {:?}", a.movpar);

    // Which bones have which animated channels?
    println!("\nper-bone channel matrix (0x06/07=anim, >=0x10 static index, else default):");
    for b in 0..a.num_bones {
        let chans: Vec<String> = (0..9)
            .map(|ch| {
                let v = a.matrix[b * 9 + ch];
                match v {
                    0x06 => "AN6".into(),
                    0x07 => "AN7".into(),
                    0x10.. => format!("S{}", v - 0x10),
                    _ => "..".into(),
                }
            })
            .collect();
        println!("  bone {b:2}: {}", chans.join(" "));
    }

    // Sample each animated channel over all output frames; find where values stop changing.
    println!("\nper-channel value evolution (frames 0..=full):");
    let total = a.blocks.len().saturating_sub(1) * 4;
    println!("  (decoded frame range 0..{total})");
    for idx in 0..a.num_moving {
        let bone = a.animated[idx] / 9;
        let ch = a.animated[idx] % 9;
        let (scale, offset) = a.movpar[idx];
        let vals: Vec<f32> = (0..=total).map(|f| a.channel_value(bone, ch, f as f32)).collect();
        let first = vals[0];
        let changed: Vec<usize> = vals
            .iter()
            .enumerate()
            .filter(|(i, v)| i + 1 < vals.len() && (*v - vals[i + 1]).abs() > 1e-6)
            .map(|(i, _)| i)
            .collect();
        let last_change = changed.last().map(|i| i + 1).unwrap_or(0);
        let uniq: Vec<u64> = vals
            .iter()
            .map(|v| (v * 1000.0).round() as i64 as u64)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        println!(
            "  ch{idx:2} bone{bone:2} T/R/S[{ch}] scale={scale:+.4} off={offset:+.4} last_change@f{last_change} first={first:+.4} n_uniq={} vals=[{:+.2}]",
            uniq.len(),
            vals.iter().map(|v| format!("{v:+.2}")).collect::<Vec<_>>().join(" ")
        );
    }
}
