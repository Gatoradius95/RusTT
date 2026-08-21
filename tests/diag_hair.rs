use rustt::an3::An3;
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
fn diag_hair() {
    let dir = "backup/CHARS/ANAKIN";
    let ghg_data = std::fs::read(format!("{dir}/ANAKIN_PADAWAN_PC.GHG")).unwrap();
    let parsed = ghg::parse(&ghg_data).unwrap();
    let raw = rustt::glb::build_meshes(&parsed);
    println!("== render items: raw bounds + bone ==");
    for (i, item) in parsed.render.iter().enumerate() {
        let Some(md) = raw.get(i) else { continue };
        let bname = parsed
            .bones
            .get(item.bone as usize)
            .map(|b| b.name.as_str())
            .unwrap_or("(static)");
        let b = bounds(&md.pos);
        println!(
            "{i:>2} part={:<3} bone={:>2} {:<12} verts={:<5} x=({:6.3},{:6.3}) y=({:6.3},{:6.3}) z=({:6.3},{:6.3})",
            item.part,
            item.bone,
            bname,
            md.pos.len(),
            b[0], b[3], b[1], b[4], b[2], b[5]
        );
    }
}
