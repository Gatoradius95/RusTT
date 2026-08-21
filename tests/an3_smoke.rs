use rustt::an3::An3;
use rustt::ghg;

#[test]
fn an3_bone_worlds_are_finite_and_parents_align() {
    let dir = std::path::Path::new("backup/CHARS/4LOM");
    if !dir.exists() {
        eprintln!("skipping: backup assets not present");
        return;
    }
    let ghg_path = dir.join("4LOM_PC.GHG");
    let data = std::fs::read(&ghg_path).expect("read ghg");
    let parsed = ghg::parse(&data).expect("parse ghg");

    let an3_path = dir.join("DEACTIVATE.AN3");
    let data = std::fs::read(&an3_path).expect("read an3");
    let an3 = An3::parse(&data).expect("parse an3");

    assert_eq!(an3.num_bones, parsed.bones.len(), "bone counts must match");
    let parents: Vec<i32> = parsed.bones.iter().map(|b| b.parent).collect();
    let rest_locals: Vec<glam::Mat4> = parsed.bones.iter().map(|b| b.local).collect();

    for frame in [0.0, 1.5, (an3.num_frames - 1) as f32] {
        let worlds = an3
            .bone_worlds(&parents, &rest_locals, frame)
            .expect("bone_worlds");
        assert_eq!(worlds.len(), an3.num_bones);
        for (i, w) in worlds.iter().enumerate() {
            assert!(
                w.is_finite(),
                "bone {i} world at frame {frame} is not finite: {w}"
            );
        }
    }
}
