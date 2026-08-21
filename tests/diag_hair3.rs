use rustt::ghg;

fn bounds(pos: &[[f32; 3]]) -> [f32; 6] {
    let mut b = [f32::INFINITY, f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];
    for p in pos {
        for (i, v) in p.iter().enumerate() {
            if *v < b[i] {
                b[i] = *v;
            }
            if *v > b[i + 3] {
                b[i + 3] = *v;
            }
        }
    }
    b
}

#[test]
fn diag_hair3() {
    let dir = "backup/CHARS/ANAKIN";
    let ghg_data = std::fs::read(format!("{dir}/ANAKIN_PADAWAN_PC.GHG")).unwrap();
    let parsed = ghg::parse(&ghg_data).unwrap();
    let raw = rustt::glb::build_meshes(&parsed);
    println!("== part materials/textures ==");
    let mut seen = std::collections::BTreeSet::new();
    for (i, item) in parsed.render.iter().enumerate() {
        if !seen.insert(item.part) {
            continue;
        }
        let Some(md) = raw.get(i) else { continue };
        let bname = parsed
            .bones
            .get(item.bone as usize)
            .map(|b| b.name.as_str())
            .unwrap_or("(static)");
        let mat = parsed
            .materials
            .iter()
            .find(|m| m.id == item.mat);
        let tex = mat
            .and_then(|m| (m.tex_id >= 0).then(|| parsed.textures.get(m.tex_id as usize)))
            .flatten();
        let b = bounds(&md.pos);
        let fmt = tex
            .map(|t| match t.fmt {
                rustt::ghg::TextureFmt::Dxt1 => "Dxt1",
                rustt::ghg::TextureFmt::Dxt5 => "Dxt5",
            })
            .unwrap_or("-");
        println!(
            "part={:<3} bone={:>2} {:<12} mat={:<4} tex={:?} fmt={} x=({:6.3},{:6.3}) y=({:6.3},{:6.3}) z=({:6.3},{:6.3})",
            item.part,
            item.bone,
            bname,
            item.mat,
            mat.map(|m| m.tex_id),
            fmt,
            b[0], b[3], b[1], b[4], b[2], b[5]
        );
    }
}
