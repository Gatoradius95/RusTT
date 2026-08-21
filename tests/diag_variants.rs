use rustt::an3::An3;
use rustt::ghg;
use glam::{Mat3, Mat4};

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

fn head_skinned_bounds(parsed: &ghg::Parsed, world: &Mat4) -> [f32; 6] {
    let raw = rustt::glb::build_meshes(parsed);
    let mut all = Vec::new();
    for (i, item) in parsed.render.iter().enumerate() {
        if item.bone == 21 {
            for p in &raw[i].pos {
                all.push(world.transform_point3(glam::Vec3::from(*p)).to_array());
            }
        }
    }
    bounds(&all)
}

fn rot3(m: &Mat4) -> Mat3 {
    Mat3::from_mat4(*m)
}

fn make_local(
    variant: usize,
    b: usize,
    an3: &An3,
    f: f32,
    parsed: &ghg::Parsed,
    ids: &[Mat4],
) -> Mat4 {
    let r = Mat4::from_rotation_z(an3.channel_value(b, 5, f))
        * Mat4::from_rotation_y(an3.channel_value(b, 4, f))
        * Mat4::from_rotation_x(an3.channel_value(b, 3, f));
    let inv_id = rot3(&ids[b]);
    let rl = rot3(&parsed.bones[b].local);
    let inv_rl = rl.inverse();
    let rot = match variant {
        0 => Mat4::from_mat3(inv_id * rot3(&r)),
        1 => Mat4::from_mat3(rl * rot3(&r)),
        2 => Mat4::from_mat3(inv_rl * inv_id * rot3(&r) * rl),
        3 => Mat4::from_mat3(inv_id * rot3(&r) * rl),
        4 => Mat4::from_mat3(rot3(&r) * rl),
        5 => Mat4::from_mat3(inv_rl * inv_id * rot3(&r)),
        _ => unreachable!(),
    };
    let t = glam::Vec3::new(
        an3.channel_value(b, 0, f),
        an3.channel_value(b, 1, f),
        an3.channel_value(b, 2, f),
    );
    Mat4::from_translation(t) * rot
}

#[test]
fn diag_variants() {
    let dir = "backup/CHARS/ANAKIN";
    let ghg_data = std::fs::read(format!("{dir}/ANAKIN_PADAWAN_PC.GHG")).unwrap();
    let parsed = ghg::parse(&ghg_data).unwrap();
    let parents: Vec<i32> = parsed.bones.iter().map(|b| b.parent).collect();
    let ids: Vec<Mat4> = parsed.bones.iter().map(|b| b.identity).collect();
    let statics: Vec<Mat4> = parsed.bones.iter().map(|b| b.world).collect();
    let head = 21usize;
    let static_b = head_skinned_bounds(&parsed, &statics[head]);
    println!("static head bounds: y {:.3}..{:.3}", static_b[1], static_b[4]);
    let names = [
        "A:id^-1*R     ",
        "B:loc*R       ",
        "D:rl^-1*id^-1*R*l",
        "E:id^-1*R*loc ",
        "F:R*loc       ",
        "G:rl^-1*id^-1*R",
    ];
    for (label, frames) in [("IDLE", vec![0.0, 10.0]), ("FORCECHOKE", vec![0.0, 5.0])] {
        let data = std::fs::read(format!("{dir}/{label}.AN3")).unwrap();
        let an3 = An3::parse(&data).unwrap();
        for f in frames {
            println!("== {label} f{f} ==");
            for variant in 0..names.len() {
                let mut worlds: Vec<Mat4> = Vec::with_capacity(31);
                for b in 0..31 {
                    let local = make_local(variant, b, &an3, f, &parsed, &ids);
                    let w = if parents[b] < 0 { local } else { worlds[parents[b] as usize] * local };
                    worlds.push(w);
                }
                let hb = head_skinned_bounds(&parsed, &worlds[head]);
                println!("   {}  head-y {:.3}..{:.3}", names[variant], hb[1], hb[4]);
            }
        }
    }
}
