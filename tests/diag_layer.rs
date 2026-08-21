use anyhow::Result;
use rustt::ghg;

fn at_i32(d: &[u8], o: i64) -> i32 {
    i32::from_le_bytes(d[o as usize..o as usize + 4].try_into().unwrap())
}

fn rel(d: &[u8], q: i64) -> Result<i64> {
    Ok(q + at_i32(d, q) as i64)
}

#[test]
fn dump_layers() -> Result<()> {
    let m = "backup/CHARS/ANAKIN/ANAKIN_PADAWAN_PC.GHG";
    let data = std::fs::read(m)?;
    let parsed = ghg::parse(&data)?;
    let nb = parsed.bones.len();

    let num20 = at_i32(&data, 0) as i64;
    let head = num20 + 4 + 4 + 0xc;
    let abs_gsnh = head + 16 + at_i32(&data, head + 12) as i64;
    let gsnh = (abs_gsnh - 12) as usize;
    let bones_base = gsnh as i64 + 16 + 4 + 0x28 + 4 + 0x130;
    let number_layer = at_i32(&data, bones_base + 16 + 24) as usize;
    let abs_layer = rel(&data, bones_base + 20 + 24)?;

    println!("number_bones={nb} number_layer={number_layer} abs_layer={abs_layer:#x}");
    let mut layer_pos = abs_layer;
    for li in 0..number_layer {
        let name_p = rel(&data, layer_pos)?;
        let name = {
            let rest = &data[name_p as usize..];
            let end = rest.iter().position(|&c| c == 0).unwrap_or(rest.len());
            String::from_utf8_lossy(&rest[..end]).into_owned()
        };
        let mut lp = [0i64; 4];
        let mut q = layer_pos + 4;
        for slot in lp.iter_mut() {
            let tmp = at_i32(&data, q);
            if tmp != 0 {
                *slot = rel(&data, q)?;
            }
            q += 4;
        }
        println!("--- layer {li} '{name}' layer_pos={layer_pos:#x}");
        for (si, sl) in lp.iter().enumerate() {
            println!("  lp[{si}]={sl:#x}");
            if *sl == 0 {
                continue;
            }
            if si % 2 == 0 {
                // per-bone table
                for bone in 0..nb {
                    let tmp = at_i32(&data, *sl + bone as i64 * 4);
                    print!(" {bone}:{tmp:#x}");
                }
                println!();
                // walk each nonzero
                for bone in 0..nb {
                    let tmp = at_i32(&data, *sl + bone as i64 * 4);
                    if tmp != 0 {
                        let tgt = *sl + bone as i64 * 4 + tmp as i64;
                        println!("    bone {bone} -> tgt {tgt:#x} ({tgt})");
                        let mut r = tgt;
                        r += 8;
                        r = rel(&data, r)?;
                        r += 0xb0;
                        r = rel(&data, r)?;
                        let pn = at_i32(&data, r);
                        println!("      pn={pn} matlist@ {:x}", r + 4);
                    }
                }
            } else {
                // not-per-bone: dump first 16 bytes then walk pos0+8
                let off = *sl as usize;
                println!(
                    "    first16: {:02x?}",
                    &data[off.min(data.len())..(off + 16).min(data.len())]
                );
                let mut r = *sl + 8;
                r = rel(&data, r)?;
                r += 0xb0;
                r = rel(&data, r)?;
                let pn = at_i32(&data, r);
                println!("      pn={pn} matlist@ {:x}", r + 4);
            }
        }
        layer_pos = q;
    }
    Ok(())
}
