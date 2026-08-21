use rustt::an3::An3;
use rustt::ghg;
use glam::Mat4;

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

fn skinned_part(parsed: &rustt::ghg::Parsed, i: usize, m: &Mat4) -> Vec<[f32; 3]> {
    let raw = rustt::glb::build_meshes(parsed);
    let md = &raw[i];
    md.pos.iter().map(|p| m.transform_point3(glam::Vec3::from(*p)).to_array()).collect()
}

#[test]
fn diag_hair2() {
    let dir = "backup/CHARS/ANAKIN";
    let ghg_data = std::fs::read(format!("{dir}/ANAKIN_PADAWAN_PC.GHG")).unwrap();
    let parsed = ghg::parse(&ghg_data).unwrap();
    let raw = rustt::glb::build_meshes(&parsed);
    let an3 = An3::parse(&std::fs::read(format!("{dir}/IDLE.AN3")).unwrap()).unwrap();
    let parents: Vec<i32> = parsed.bones.iter().map(|b| b.parent).collect();
    let rest_locals: Vec<Mat4> = parsed.bones.iter().map(|b| b.local).collect();
    let aw = an3.bone_worlds(&parents, &rest_locals, 0.0).unwrap();
    let static_w: Vec<Mat4> = parsed.bones.iter().map(|b| b.world).collect();

    println!("{:<4} {:<4} {:>4}  {:>24} {:>24} {:>10}", "idx", "prt", "bone", "static y-zy", "anim y-z", "animY-cy");
    for (i, item) in parsed.render.iter().enumerate() {
        let Some(md) = raw.get(i) else { continue };
        let st = bounds(&skinned_part(&parsed, i, &static_w[item.bone.max(0) as usize]));
        let an = if item.bone >= 0 {
            bounds(&skinned_part(&parsed, i, &aw[item.bone as usize]))
        } else {
            let b = bounds(&md.pos);
            b
        };
        let cy = (st[1] + st[4]) / 2.0;
        let ay = (an[1] + an[4]) / 2.0;
        println!(
            "{:<4} {:<4} {:>4}  y=({:6.3},{:6.3}) z=({:6.3},{:6.3})  y=({:6.3},{:6.3}) z=({:6.3},{:6.3})  {:>10.3}",
            i, item.part, item.bone, st[1], st[4], st[2], st[5], an[1], an[4], an[2], an[5], ay - cy
        );
    }
}
