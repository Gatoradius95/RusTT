use rustt::an3::An3;
use rustt::ghg;
use glam::{Mat3, Mat4, Quat};

fn euler(m: &Mat4) -> [f32; 3] {
    let q = Quat::from_mat3(&Mat3::from_mat4(*m));
    let (x, y, z) = q.to_euler(glam::EulerRot::ZYX);
    [x * 180.0 / std::f32::consts::PI, y * 180.0 / std::f32::consts::PI, z * 180.0 / std::f32::consts::PI]
}

fn delta(a: &Mat4, b: &Mat4) -> f32 {
    let qa = Quat::from_mat3(&Mat3::from_mat4(*a));
    let qb = Quat::from_mat3(&Mat3::from_mat4(*b));
    qa.dot(qb).abs().min(1.0).acos() * 2.0 * 180.0 / std::f32::consts::PI
}

#[test]
fn diag_compose() {
    let dir = "backup/CHARS/ANAKIN";
    let ghg_data = std::fs::read(format!("{dir}/ANAKIN_PADAWAN_PC.GHG")).unwrap();
    let parsed = ghg::parse(&ghg_data).unwrap();
    let parents: Vec<i32> = parsed.bones.iter().map(|b| b.parent).collect();
    let ids: Vec<Mat4> = parsed.bones.iter().map(|b| b.identity).collect();
    let static_w: Vec<Mat4> = parsed.bones.iter().map(|b| b.world).collect();

    let idata = std::fs::read(format!("{dir}/IDLE.AN3")).unwrap();
    let an3 = An3::parse(&idata).unwrap();

    println!("f0 rest check for IDLE: local_anim variants vs static world (delta deg per bone)");
    println!("  {:<12} {:>8} {:>8} {:>8} {:>8}", "bone", "inv_id*R", "idT*R", "local*R", "R");
    let names = ["root", "lowerTorso", "upperTorso", "head", "lHand", "rHand"];
    for &bi in &[0usize, 1, 3, 8, 14, 21, 22, 23, 24, 27, 28] {
        let parent_w = if parents[bi] < 0 { Mat4::IDENTITY } else { static_w[parents[bi] as usize] };
        let r_anim = Mat4::from_rotation_z(an3.channel_value(bi, 5, 0.0))
            * Mat4::from_rotation_y(an3.channel_value(bi, 4, 0.0))
            * Mat4::from_rotation_x(an3.channel_value(bi, 3, 0.0));
        let id_rot = Mat3::from_mat4(ids[bi]);
        let inv_id = Mat4::from_mat3(id_rot);
        let id_t = Mat4::from_mat3(id_rot.transpose());
        let local_static = parsed.bones[bi].local;

        let v_a = parent_w * (inv_id * r_anim);
        let v_b = parent_w * (id_t * r_anim);
        let v_c = parent_w * (local_static * r_anim);
        let v_d = parent_w * r_anim;
        println!(
            "  {:<12} {:>8.1} {:>8.1} {:>8.1} {:>8.1}",
            names.get(bi).copied().unwrap_or("bone"),
            delta(&v_a, &static_w[bi]),
            delta(&v_b, &static_w[bi]),
            delta(&v_c, &static_w[bi]),
            delta(&v_d, &static_w[bi]),
        );
    }
    println!();
    println!("head bone 21 static local euler = {:?}", euler(&parsed.bones[21].local));
    println!("head bone 21 identity  euler    = {:?}", euler(&ids[21]));
    println!("head bone 21 inv(identity) euler = {:?}", euler(&Mat4::from_mat3(Mat3::from_mat4(ids[21]))));
}
