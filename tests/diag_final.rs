use rustt::an3::An3;
use rustt::ghg;
use glam::{Mat3, Mat4, Quat};

fn rot3(m: &Mat4) -> Mat3 {
    Mat3::from_mat4(*m)
}

fn delta(a: &Mat4, b: &Mat4) -> f32 {
    let qa = Quat::from_mat3(&Mat3::from_mat4(*a));
    let qb = Quat::from_mat3(&Mat3::from_mat4(*b));
    qa.dot(qb).abs().min(1.0).acos() * 2.0 * 180.0 / std::f32::consts::PI
}

fn make_local(variant: usize, b: usize, an3: &An3, f: f32, parsed: &ghg::Parsed, ids: &[Mat4]) -> Mat4 {
    let r = Mat4::from_rotation_z(an3.channel_value(b, 5, f))
        * Mat4::from_rotation_y(an3.channel_value(b, 4, f))
        * Mat4::from_rotation_x(an3.channel_value(b, 3, f));
    let inv_id = rot3(&ids[b]);
    let rl = rot3(&parsed.bones[b].local);
    let m = Mat3::from_diagonal(glam::Vec3::new(-1.0, 1.0, 1.0));
    let rot = match variant {
        0 => Mat4::from_mat3(inv_id * rot3(&r)),
        1 => Mat4::from_mat3(rl * rot3(&r)),
        2 => Mat4::from_mat3(rl.inverse() * inv_id * rot3(&r) * rl),
        3 => Mat4::from_mat3(rot3(&r) * rl),
        4 => Mat4::from_mat3(m * inv_id * rot3(&r) * m),
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
fn diag_final() {
    let dir = "backup/CHARS/ANAKIN";
    let ghg_data = std::fs::read(format!("{dir}/ANAKIN_PADAWAN_PC.GHG")).unwrap();
    let parsed = ghg::parse(&ghg_data).unwrap();
    let parents: Vec<i32> = parsed.bones.iter().map(|b| b.parent).collect();
    let ids: Vec<Mat4> = parsed.bones.iter().map(|b| b.identity).collect();
    let statics: Vec<Mat4> = parsed.bones.iter().map(|b| b.world).collect();

    let names = ["A:id^-1*R  ", "B:loc*R    ", "D:rl^-1*id^-1*R*rl", "F:R*loc    ", "H:M*id^-1*R*M"];
    println!("IDLE f0: per-bone world rotation delta vs static (deg); small=reproduces rest");
    for variant in 0..5 {
        let data = std::fs::read(format!("{dir}/IDLE.AN3")).unwrap();
        let an3 = An3::parse(&data).unwrap();
        let mut worlds: Vec<Mat4> = Vec::with_capacity(31);
        for b in 0..31 {
            let local = make_local(variant, b, &an3, 0.0, &parsed, &ids);
            let w = if parents[b] < 0 { local } else { worlds[parents[b] as usize] * local };
            worlds.push(w);
        }
        let mut max_d = 0.0f32;
        let mut bad = Vec::new();
        for b in 0..31 {
            let d = delta(&worlds[b], &statics[b]);
            max_d = max_d.max(d);
            if d > 1.0 {
                bad.push((b, d));
            }
        }
        println!(
            "  {} max_d={:6.2}  bones>1deg: {}",
            names[variant],
            max_d,
            bad.iter().map(|(b, d)| format!("{b}:{d:.1}")).collect::<Vec<_>>().join(" ")
        );
    }
    println!();
    println!("Full-model skinned bounds (y-range) for several animations at various frames:");
    println!("  (rest bound should stay ~0.15..0.55; explosions = wrong)");
    for (label, frames) in [
        ("IDLE", vec![0.0, 60.0, 120.0]),
        ("FORCECHOKE", vec![0.0, 5.0, 10.0]),
        ("RUN", vec![0.0, 5.0, 10.0]),
        ("BACKFLIP", vec![0.0, 8.0, 15.0]),
    ] {
        let Ok(data) = std::fs::read(format!("{dir}/{label}.AN3")) else {
            continue;
        };
        let an3 = An3::parse(&data).unwrap();
        for f in frames {
            println!("== {label} f{f} ==");
            for variant in 0..5 {
                let mut worlds: Vec<Mat4> = Vec::with_capacity(31);
                for b in 0..31 {
                    let local = make_local(variant, b, &an3, f, &parsed, &ids);
                    let w = if parents[b] < 0 { local } else { worlds[parents[b] as usize] * local };
                    worlds.push(w);
                }
                let raw = rustt::glb::build_meshes(&parsed);
                let mut mn = [f32::INFINITY; 3];
                let mut mx = [f32::NEG_INFINITY; 3];
                for (i, item) in parsed.render.iter().enumerate() {
                    let wm = if item.bone >= 0 { worlds[item.bone as usize] } else { Mat4::IDENTITY };
                    for p in &raw[i].pos {
                        let q = wm.transform_point3(glam::Vec3::from(*p));
                        for k in 0..3 {
                            mn[k] = mn[k].min(q[k]);
                            mx[k] = mx[k].max(q[k]);
                        }
                    }
                }
                println!(
                    "   {}  x [{:.2},{:.2}] y [{:.2},{:.2}] z [{:.2},{:.2}]",
                    names[variant], mn[0], mx[0], mn[1], mx[1], mn[2], mx[2]
                );
            }
        }
    }
}
