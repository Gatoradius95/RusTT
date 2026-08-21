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
fn diag_id() {
    let dir = "backup/CHARS/ANAKIN";
    let ghg_data = std::fs::read(format!("{dir}/ANAKIN_PADAWAN_PC.GHG")).unwrap();
    let parsed = ghg::parse(&ghg_data).unwrap();
    println!("{:<12} {:>9} {:>9} {:>9} {:>9}", "bone", "locEuler", "idT", "local^T", "local*id");
    let names = ["root", "lowerTorso", "upperTorso", "chest", "lUpperArm", "lHand", "rHand", "head", "helmet", "b23", "b24", "b25", "b27", "b28", "b29"];
    for (bi, b) in parsed.bones.iter().enumerate() {
        let id_rot = Mat3::from_mat4(b.identity);
        let loc_rot = Mat3::from_mat4(b.local);
        let id_t = Mat4::from_mat3(id_rot.transpose());
        let loc_t = Mat4::from_mat3(loc_rot.transpose());
        let loc_x_id = Mat4::from_mat3(loc_rot * id_rot);
        let d1 = delta(&Mat4::from_mat3(loc_rot), &id_t);
        let d2 = delta(&Mat4::from_mat3(loc_rot), &loc_t);
        let d3 = delta(&Mat4::from_mat3(loc_rot), &loc_x_id);
        let nm = names.get(bi).copied().unwrap_or("bone");
        println!(
            "{:<12} {:>9} {:>9.1} {:>9.1} {:>9.1}",
            format!("{nm}[{bi}]"),
            format!("{:?}", euler(&b.local)),
            d1, d2, d3
        );
    }
}
