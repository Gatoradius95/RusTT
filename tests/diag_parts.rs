use rustt::an3::An3;
use rustt::ghg;

fn bounds(mesh: &[[f32; 3]]) -> [f32; 6] {
    let mut mn = [f32::INFINITY; 3];
    let mut mx = [f32::NEG_INFINITY; 3];
    for p in mesh {
        for k in 0..3 {
            mn[k] = mn[k].min(p[k]);
            mx[k] = mx[k].max(p[k]);
        }
    }
    [mn[0], mn[1], mn[2], mx[0], mx[1], mx[2]]
}

fn skinned_part(parsed: &ghg::Parsed, i: usize, world: &glam::Mat4) -> Vec<[f32; 3]> {
    let raw = rustt::glb::build_meshes(parsed);
    let md = &raw[i];
    md.pos
        .iter()
        .map(|p| world.transform_point3(glam::Vec3::from(*p)).to_array())
        .collect()
}

#[test]
fn diag_parts() {
    let dir = "backup/CHARS/ANAKIN";
    let ghg_data = std::fs::read(format!("{dir}/ANAKIN_PADAWAN_PC.GHG")).unwrap();
    let parsed = ghg::parse(&ghg_data).unwrap();
    let an3 = An3::parse(&std::fs::read(format!("{dir}/FORCECHOKE.AN3")).unwrap()).unwrap();
    let parents: Vec<i32> = parsed.bones.iter().map(|b| b.parent).collect();
    let ids: Vec<glam::Mat4> = parsed.bones.iter().map(|b| b.local).collect();
    let w = an3.bone_worlds(&parents, &ids, 0.0).unwrap();

    let raw = rustt::glb::build_meshes(&parsed);
    println!("== ANAKIN render items ==");
    for (i, item) in parsed.render.iter().enumerate() {
        let md = &raw[i];
        let bname = parsed
            .bones
            .get(item.bone as usize)
            .map(|b| b.name.as_str())
            .unwrap_or("-");
        let raw_b = bounds(&md.pos);
        let world_m = parsed.bones[item.bone.max(0) as usize].world;
        let anim_m = if item.bone >= 0 { w[item.bone as usize] } else { glam::Mat4::IDENTITY };
        let stat = bounds(&skinned_part(&parsed, i, &world_m));
        let anim = if item.bone >= 0 {
            bounds(&skinned_part(&parsed, i, &anim_m))
        } else {
            raw_b
        };
        println!(
            "{i:2} part={:<3} bone={:>2} {:<13} verts={:<5} raw=({:7.3},{:7.3},{:7.3})|({:7.3},{:7.3},{:7.3})",
            item.part,
            item.bone,
            bname,
            md.pos.len(),
            raw_b[0],
            raw_b[1],
            raw_b[2],
            raw_b[3],
            raw_b[4],
            raw_b[5]
        );
        println!(
            "       static=({:7.3},{:7.3},{:7.3})|({:7.3},{:7.3},{:7.3})",
            stat[0], stat[1], stat[2], stat[3], stat[4], stat[5]
        );
        println!(
            "       anim  =({:7.3},{:7.3},{:7.3})|({:7.3},{:7.3},{:7.3})",
            anim[0], anim[1], anim[2], anim[3], anim[4], anim[5]
        );
    }
}
