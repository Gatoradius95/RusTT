use rustt::an3::An3;
use rustt::ghg;
use glam::{Mat3, Mat4, Quat};

fn euler(m: &Mat4) -> [f32; 3] {
    let q = Quat::from_mat3(&Mat3::from_mat4(*m));
    let (x, y, z) = q.to_euler(glam::EulerRot::ZYX);
    [
        x * 180.0 / std::f32::consts::PI,
        y * 180.0 / std::f32::consts::PI,
        z * 180.0 / std::f32::consts::PI,
    ]
}

#[test]
fn diag_id_rl() {
    let dir = "backup/CHARS/ANAKIN";
    let ghg_data = std::fs::read(format!("{dir}/ANAKIN_PADAWAN_PC.GHG")).unwrap();
    let parsed = ghg::parse(&ghg_data).unwrap();
    let an3 = An3::parse(&std::fs::read(format!("{dir}/IDLE.AN3")).unwrap()).unwrap();
    println!("{:<18} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}", "bone", "idEul", "rlEul", "id*rlEul", "chanEul", "rl*chanEul", "A:id*rl?ch", "chanMag");
    let names = ["character", "upperTorso", "body", "chest", "leftArm", "leftShoulder", "leftElbow", "leftElbowLen", "leftHand", "weaponLeft", "rightArm", "rightShoulder", "rightElbow", "rightElbowLen", "rightHand", "weaponRight", "cloak0", "cloak1", "cloak2", "cloak3", "cloak4", "head", "helmet", "rightLeg", "rightKnee", "rightAnkle", "rightToe", "leftLeg", "leftKnee", "leftAnkle", "leftToe"];
    for (bi, b) in parsed.bones.iter().enumerate() {
        let id = Mat4::from_mat3(Mat3::from_mat4(b.identity));
        let rl = Mat4::from_mat3(Mat3::from_mat4(b.local));
        let id_rl = Mat4::from_mat3(Mat3::from_mat4(b.identity) * Mat3::from_mat4(b.local));
        let r = Mat4::from_rotation_z(an3.channel_value(bi, 5, 0.0))
            * Mat4::from_rotation_y(an3.channel_value(bi, 4, 0.0))
            * Mat4::from_rotation_x(an3.channel_value(bi, 3, 0.0));
        let chan = euler(&r);
        let rl_r = Mat4::from_mat3(Mat3::from_mat4(b.local) * Mat3::from_mat4(r));
        let q_idrl = Quat::from_mat3(&Mat3::from_mat4(id_rl));
        let q_r = Quat::from_mat3(&Mat3::from_mat4(r));
        let q_rl = Quat::from_mat3(&Mat3::from_mat4(rl));
        let q_rlr = Quat::from_mat3(&Mat3::from_mat4(rl_r));
        let dA = q_idrl.dot(q_r).abs().min(1.0).acos() * 2.0 * 180.0 / std::f32::consts::PI;
        let dB = q_rlr.dot(q_rl).abs().min(1.0).acos() * 2.0 * 180.0 / std::f32::consts::PI;
        let chan_mag = (an3.channel_value(bi, 3, 0.0).abs()
            + an3.channel_value(bi, 4, 0.0).abs()
            + an3.channel_value(bi, 5, 0.0).abs())
            * 180.0 / std::f32::consts::PI;
        let nm = names.get(bi).copied().unwrap_or("");
        println!(
            "{:<18} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9.2} {:>9.2}",
            format!("{nm}[{bi}]"),
            format!("{:?}", euler(&id)),
            format!("{:?}", euler(&rl)),
            format!("{:?}", euler(&id_rl)),
            format!("{:?}", chan),
            format!("{:?}", euler(&rl_r)),
            dA,
            chan_mag,
        );
    }
}
