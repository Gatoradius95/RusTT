use rustt::an3::An3;
use rustt::ghg;

fn rot_of(m: &glam::Mat4) -> (f32, f32, f32) {
    let m3 = glam::Mat3::from_mat4(*m);
    let q = glam::Quat::from_mat3(&m3);
    let e = q.to_euler(glam::EulerRot::ZYX);
    (e.0.to_degrees(), e.1.to_degrees(), e.2.to_degrees())
}

#[test]
fn diag_rot() {
    let dir = "backup/CHARS/ANAKIN";
    let ghg_data = std::fs::read(format!("{dir}/ANAKIN_PADAWAN_PC.GHG")).unwrap();
    let parsed = ghg::parse(&ghg_data).unwrap();
    let parents: Vec<i32> = parsed.bones.iter().map(|b| b.parent).collect();
    let ids: Vec<glam::Mat4> = parsed.bones.iter().map(|b| b.local).collect();

    let targets = [1usize, 3, 4, 8, 10, 14, 21, 22];
    println!("== head (bone 21) matrix dump ==");
    println!("identity = {:?}", parsed.bones[21].identity.to_cols_array());
    println!("local    = {:?}", parsed.bones[21].local.to_cols_array());
    println!("world    = {:?}", parsed.bones[21].world.to_cols_array());
    println!("world euler xyz = {:?}", rot_of(&parsed.bones[21].world));

    for (label, fname, frame) in [
        ("FC f0", "FORCECHOKE.AN3", 0.0),
        ("FC f3", "FORCECHOKE.AN3", 3.0),
        ("IDLE f0", "IDLE.AN3", 0.0),
    ] {
        let Ok(data) = std::fs::read(format!("{dir}/{fname}")) else {
            println!("{label}: {fname} not found");
            continue;
        };
        let Ok(an3) = An3::parse(&data) else {
            println!("{label}: parse failed");
            continue;
        };
        let w = an3.bone_worlds(&parents, &ids, frame).unwrap();
        println!("== {label} ==");
        for &bi in &targets {
            let lr = rot_of(&an3.bone_local(bi, frame, ids.get(bi)));
            let wr = rot_of(&w[bi]);
            println!(
                "  bone {bi:>2} {:<12} local_xyz=({:>7.1},{:>7.1},{:>7.1})  world_t=({:7.4},{:7.4},{:7.4})  world_xyz=({:7.1},{:7.1},{:7.1})",
                parsed.bones[bi].name,
                lr.0,
                lr.1,
                lr.2,
                w[bi].w_axis.x,
                w[bi].w_axis.y,
                w[bi].w_axis.z,
                wr.0,
                wr.1,
                wr.2
            );
        }
        println!();
    }
    let d = std::fs::read(format!("{dir}/FORCECHOKE.AN3")).unwrap();
    let an3 = An3::parse(&d).unwrap();
    println!("4INA={} footer(bone1)={:#04x} footer(21)={:#04x}", an3.four_ina, an3.footer[1], an3.footer[21]);
}
