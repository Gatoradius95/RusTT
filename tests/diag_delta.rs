use rustt::an3::An3;
use rustt::ghg;

fn delta_angle(a: &glam::Mat4, b: &glam::Mat4) -> f32 {
    let qa = glam::Quat::from_mat3(&glam::Mat3::from_mat4(*a));
    let qb = glam::Quat::from_mat3(&glam::Mat3::from_mat4(*b));
    let d = qa.dot(qb).abs().min(1.0);
    d.acos() * 2.0 * 180.0 / std::f32::consts::PI
}

#[test]
fn diag_delta() {
    let dir = "backup/CHARS/ANAKIN";
    let ghg_data = std::fs::read(format!("{dir}/ANAKIN_PADAWAN_PC.GHG")).unwrap();
    let parsed = ghg::parse(&ghg_data).unwrap();
    let parents: Vec<i32> = parsed.bones.iter().map(|b| b.parent).collect();
    let ids: Vec<glam::Mat4> = parsed.bones.iter().map(|b| b.local).collect();
    let static_w: Vec<glam::Mat4> = parsed.bones.iter().map(|b| b.world).collect();

    let targets = [1usize, 3, 4, 8, 10, 14, 21, 22];
    println!("rotation delta (deg) between static world and animated world");
    println!("(small = sane pose; ~90+ = wrong)");
    for (label, fname, frames) in [
        ("IDLE", "IDLE.AN3", vec![0.0, 20.0, 60.0, 120.0]),
        ("FC", "FORCECHOKE.AN3", vec![0.0, 2.0, 5.0, 10.0]),
        ("NOHANDIDLE", "NOHANDIDLE.AN3", vec![0.0, 20.0, 60.0]),
    ] {
        let Ok(data) = std::fs::read(format!("{dir}/{fname}")) else {
            continue;
        };
        let Ok(an3) = An3::parse(&data) else {
            continue;
        };
        println!("== {label} (bones={}) ==", an3.num_bones);
        for f in frames {
            let Ok(w) = an3.bone_worlds(&parents, &ids, f) else {
                continue;
            };
            let mut row = String::new();
            for &bi in &targets {
                row.push_str(&format!(
                    "  {:<11}={:>6.1}",
                    parsed.bones[bi].name,
                    delta_angle(&static_w[bi], &w[bi])
                ));
            }
            println!("  f{f:>3}:{row}");
        }
    }
}
