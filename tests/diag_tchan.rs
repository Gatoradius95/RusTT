use rustt::an3::An3;
use rustt::ghg;
use glam::{Mat3, Mat4, Quat};

#[test]
fn diag_tchan() {
    let dir = "backup/CHARS/ANAKIN";
    let ghg_data = std::fs::read(format!("{dir}/ANAKIN_PADAWAN_PC.GHG")).unwrap();
    let parsed = ghg::parse(&ghg_data).unwrap();
    for (label, fname, frame) in [
        ("IDLE f0", "IDLE.AN3", 0.0),
        ("IDLE f1", "IDLE.AN3", 1.0),
        ("FC f0", "FORCECHOKE.AN3", 0.0),
        ("REST", "ANAKIN_PADAWAN.AN3", 0.0),
    ] {
        let data = std::fs::read(format!("{dir}/{fname}")).unwrap();
        let an3 = An3::parse(&data).unwrap();
        println!("== {label} (num_frames={}) ==", an3.num_frames);
        println!(
            "{:<12} {:>10} {:>10} {:>10}   {:>10}",
            "bone", "anim_t.x", "anim_t.y", "anim_t.z", "static_t"
        );
        for bi in [0usize, 1, 2, 3, 4, 8, 10, 14, 21, 22, 23] {
            let b = &parsed.bones[bi];
            let t = (
                an3.channel_value(bi, 0, frame),
                an3.channel_value(bi, 1, frame),
                an3.channel_value(bi, 2, frame),
            );
            let st = b.local.w_axis;
            println!(
                "{:<12} {:>10.5} {:>10.5} {:>10.5}   ({:7.4},{:7.4},{:7.4})",
                b.name,
                t.0, t.1, t.2,
                st.x, st.y, st.z
            );
        }
    }
    let qs = Quat::from_mat3(&Mat3::from_mat4(parsed.bones[21].local));
    let e = qs.to_euler(glam::EulerRot::ZYX);
    println!(
        "\nhead local quat euler zyx = ({:.2},{:.2},{:.2}) deg",
        e.0.to_degrees(),
        e.1.to_degrees(),
        e.2.to_degrees()
    );
}
