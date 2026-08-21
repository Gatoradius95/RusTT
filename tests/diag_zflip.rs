use rustt::an3::An3;
use rustt::ghg;
use glam::{Mat3, Mat4, Quat};

fn delta(a: &Mat4, b: &Mat4) -> f32 {
    let qa = Quat::from_mat3(&Mat3::from_mat4(*a));
    let qb = Quat::from_mat3(&Mat3::from_mat4(*b));
    qa.dot(qb).abs().min(1.0).acos() * 2.0 * 180.0 / std::f32::consts::PI
}

#[test]
fn diag_zflip() {
    let dir = "backup/CHARS/ANAKIN";
    let ghg_data = std::fs::read(format!("{dir}/ANAKIN_PADAWAN_PC.GHG")).unwrap();
    let parsed = ghg::parse(&ghg_data).unwrap();
    let an3 = An3::parse(&std::fs::read(format!("{dir}/ANAKIN_PADAWAN.AN3")).unwrap()).unwrap();
    let parents: Vec<i32> = parsed.bones.iter().map(|b| b.parent).collect();
    let rest_locals: Vec<Mat4> = parsed.bones.iter().map(|b| b.local).collect();
    let static_w: Vec<Mat4> = parsed.bones.iter().map(|b| b.world).collect();

    let names = ["character", "upperTorso", "body", "chest", "leftArm", "leftShoulder", "leftElbow", "leftElbowLen", "leftHand", "weaponLeft", "rightArm", "rightShoulder", "rightElbow", "rightElbowLen", "rightHand", "weaponRight", "cloak0", "cloak1", "cloak2", "cloak3", "cloak4", "head", "helmet", "rightLeg", "rightKnee", "rightAnkle", "rightToe", "leftLeg", "leftKnee", "leftAnkle", "leftToe"];

    // variant Z: negate the AN3 translation z channel
    let mut worlds = Vec::with_capacity(an3.num_bones);
    for b in 0..an3.num_bones {
        let mut t = glam::Vec3::new(
            an3.channel_value(b, 0, 0.0),
            an3.channel_value(b, 1, 0.0),
            an3.channel_value(b, 2, 0.0),
        );
        t.z = -t.z;
        let rl = Mat4::from_mat3(Mat3::from_mat4(rest_locals[b]));
        let r_anim = Mat4::from_rotation_z(an3.channel_value(b, 5, 0.0))
            * Mat4::from_rotation_y(an3.channel_value(b, 4, 0.0))
            * Mat4::from_rotation_x(an3.channel_value(b, 3, 0.0));
        let local = if an3.uses_x20(b) {
            Mat4::from_translation(t) * rl * r_anim
        } else {
            Mat4::from_translation(t) * r_anim
        };
        let w = if parents[b] < 0 {
            local
        } else {
            worlds[parents[b] as usize] * local
        };
        worlds.push(w);
    }
    println!("REST AN3 with z-negated translations vs static world:");
    let mut worst = 0.0f32;
    for (bi, b) in parsed.bones.iter().enumerate() {
        let d = delta(&worlds[bi], &static_w[bi]);
        let dt = (worlds[bi].w_axis - static_w[bi].w_axis).length();
        worst = worst.max(d);
        let nm = names.get(bi).copied().unwrap_or("");
        if d > 1.0 || dt > 0.02 {
            println!(
                "  {:<12}[{}] rot_delta={:6.2} deg  t_delta={:.5}",
                nm, bi, d, dt
            );
        }
    }
    println!("worst rot delta over all bones: {worst:.2} deg");
}
