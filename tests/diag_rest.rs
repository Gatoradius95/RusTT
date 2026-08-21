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

fn qd(a: &Mat3, b: &Mat3) -> f32 {
    let qa = Quat::from_mat3(a);
    let qb = Quat::from_mat3(b);
    qa.dot(qb).abs().min(1.0).acos() * 2.0 * 180.0 / std::f32::consts::PI
}

#[test]
fn diag_rest() {
    let dir = "backup/CHARS/ANAKIN";
    let ghg_data = std::fs::read(format!("{dir}/ANAKIN_PADAWAN_PC.GHG")).unwrap();
    let parsed = ghg::parse(&ghg_data).unwrap();
    let an3 = An3::parse(&std::fs::read(format!("{dir}/ANAKIN_PADAWAN.AN3")).unwrap()).unwrap();
    let names = ["character", "upperTorso", "body", "chest", "leftArm", "leftShoulder", "leftElbow", "leftElbowLen", "leftHand", "weaponLeft", "rightArm", "rightShoulder", "rightElbow", "rightElbowLen", "rightHand", "weaponRight", "cloak0", "cloak1", "cloak2", "cloak3", "cloak4", "head", "helmet", "rightLeg", "rightKnee", "rightAnkle", "rightToe", "leftLeg", "leftKnee", "leftAnkle", "leftToe"];
    println!("channels are static: {:?}", an3.keyblock_len);
    println!("{:<18} {:>9} {:>9} {:>9} {:>9}", "bone", "chanEul", "chanDeg", "A:id*rl?ch", "B:chMag");
    for (bi, b) in parsed.bones.iter().enumerate() {
        let r = Mat4::from_rotation_z(an3.channel_value(bi, 5, 0.0))
            * Mat4::from_rotation_y(an3.channel_value(bi, 4, 0.0))
            * Mat4::from_rotation_x(an3.channel_value(bi, 3, 0.0));
        let id = Mat3::from_mat4(b.identity);
        let rl = Mat3::from_mat4(b.local);
        let dA = qd(&(id * rl), &Mat3::from_mat4(r));
        let ch_mag = (an3.channel_value(bi, 3, 0.0).abs()
            + an3.channel_value(bi, 4, 0.0).abs()
            + an3.channel_value(bi, 5, 0.0).abs())
            * 180.0 / std::f32::consts::PI;
        let nm = names.get(bi).copied().unwrap_or("");
        println!(
            "{:<18} {:>9} {:>9.2} {:>9.2} {:>9.2}",
            format!("{nm}[{bi}]"),
            format!("{:?}", euler(&r)),
            (an3.channel_value(bi, 3, 0.0).abs() + an3.channel_value(bi, 4, 0.0).abs() + an3.channel_value(bi, 5, 0.0).abs()) * 180.0 / std::f32::consts::PI,
            dA,
            ch_mag,
        );
    }
}
