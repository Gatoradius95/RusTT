use rustt::an3::An3;
use rustt::ghg;
use glam::{Mat3, Mat4, Quat};

fn tr(m: &Mat4) -> (f32, f32, f32) {
    let t = m.w_axis;
    (t.x, t.y, t.z)
}

fn eul(m: &Mat4) -> (f32, f32, f32) {
    let q = Quat::from_mat3(&Mat3::from_mat4(*m));
    let e = q.to_euler(glam::EulerRot::ZYX);
    (e.0.to_degrees(), e.1.to_degrees(), e.2.to_degrees())
}

#[test]
fn diag_headchain() {
    let dir = "backup/CHARS/ANAKIN";
    let ghg_data = std::fs::read(format!("{dir}/ANAKIN_PADAWAN_PC.GHG")).unwrap();
    let parsed = ghg::parse(&ghg_data).unwrap();
    let an3 = An3::parse(&std::fs::read(format!("{dir}/IDLE.AN3")).unwrap()).unwrap();
    let parents: Vec<i32> = parsed.bones.iter().map(|b| b.parent).collect();
    let rest_locals: Vec<Mat4> = parsed.bones.iter().map(|b| b.local).collect();
    let aw = an3.bone_worlds(&parents, &rest_locals, 0.0).unwrap();

    for bi in [21usize, 22, 3, 4, 10] {
        let b = &parsed.bones[bi];
        println!(
            "bone {bi} {:<12} parent={}",
            b.name,
            b.parent
        );
        let lt = tr(&b.local);
        let le = eul(&b.local);
        let st = tr(&b.world);
        let se = eul(&b.world);
        let at = tr(&aw[bi]);
        let ae = eul(&aw[bi]);
        let qs = Quat::from_mat3(&Mat3::from_mat4(b.world));
        let qa = Quat::from_mat3(&Mat3::from_mat4(aw[bi]));
        let deg = qs.dot(qa).abs().min(1.0).acos() * 2.0 * 180.0 / std::f32::consts::PI;
        let zs = (b.world * glam::Vec4::Z.truncate().extend(1.0)).truncate().normalize();
        let za = (aw[bi] * glam::Vec4::Z.truncate().extend(1.0)).truncate().normalize();
        println!(
            "  local   t=({:7.4},{:7.4},{:7.4}) eul=({:7.2},{:7.2},{:7.2})",
            lt.0, lt.1, lt.2, le.0, le.1, le.2
        );
        println!(
            "  static  t=({:7.4},{:7.4},{:7.4}) eul=({:7.2},{:7.2},{:7.2})",
            st.0, st.1, st.2, se.0, se.1, se.2
        );
        println!(
            "  animf0  t=({:7.4},{:7.4},{:7.4}) eul=({:7.2},{:7.2},{:7.2})",
            at.0, at.1, at.2, ae.0, ae.1, ae.2
        );
        println!(
            "  stat-anim delta={:.2} deg; local +Z world: static=({:.3},{:.3},{:.3}) anim=({:.3},{:.3},{:.3})",
            deg,
            zs.x, zs.y, zs.z,
            za.x, za.y, za.z
        );
    }
}
