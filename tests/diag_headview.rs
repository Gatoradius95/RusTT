use rustt::ghg;
use glam::Mat4;

fn skinned(parsed: &rustt::ghg::Parsed, i: usize, m: &Mat4) -> Vec<[f32; 3]> {
    let raw = rustt::glb::build_meshes(parsed);
    let md = &raw[i];
    md.pos.iter().map(|p| m.transform_point3(glam::Vec3::from(*p)).to_array()).collect()
}

#[test]
fn diag_headview() {
    let dir = "backup/CHARS/ANAKIN";
    let ghg_data = std::fs::read(format!("{dir}/ANAKIN_PADAWAN_PC.GHG")).unwrap();
    let parsed = ghg::parse(&ghg_data).unwrap();
    let raw = rustt::glb::build_meshes(&parsed);
    let static_w: Vec<Mat4> = parsed.bones.iter().map(|b| b.world).collect();

    // front view ASCII: y up, x right, around the head/neck region
    let x0 = -0.25f32;
    let x1 = 0.25f32;
    let y0 = 0.05f32;
    let y1 = 0.50f32;
    let cols = 100usize;
    let rows = 60usize;
    let mut grid = vec![vec!['.'; 100]; 60];
    let part_idents = [0usize, 1, 4, 5, 6, 7, 8, 9, 34, 35];
    for (i, item) in parsed.render.iter().enumerate() {
        let ident = part_idents.iter().position(|p| *p == item.part);
        let Some(ident) = ident else { continue };
        let ch = b"0123456789ABC"[ident] as char;
        let pos = if item.bone >= 0 {
            skinned(&parsed, i, &static_w[item.bone as usize])
        } else {
            raw[i].pos.clone()
        };
        for p in &pos {
            let (x, y, z) = (p[0], p[1], p[2]);
            if x < x0 || x > x1 || y < y0 || y > y1 {
                continue;
            }
            let c = ((x - x0) / (x1 - x0) * (cols as f32 - 1.0)).round() as usize;
            let r = ((y1 - y) / (y1 - y0) * (rows as f32 - 1.0)).round() as usize;
            let cur = grid[r][c];
            grid[r][c] = if cur == '.' { ch } else if cur != ch { '#' } else { ch };
        }
    }
    println!("front view, y={y0}..{y1}, x={x0}..{x1}; parts: 0=^ 1=! 4=head@ 5=& 6=* 7=( 8=) 9=- 34=+ 35=helmet=");
    for r in 0..rows {
        let line: String = grid[r].iter().collect();
        println!("{line}");
    }
}
