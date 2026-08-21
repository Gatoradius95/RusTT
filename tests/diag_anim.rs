use rustt::an3::An3;
use rustt::ghg;

#[test]
fn diag_anim() {
    let dir = "backup/CHARS/ANAKIN";
    let ghg_data = std::fs::read(format!("{dir}/ANAKIN_PADAWAN_PC.GHG")).unwrap();
    let parsed = ghg::parse(&ghg_data).unwrap();
    let an3 = An3::parse(&std::fs::read(format!("{dir}/FORCECHOKE.AN3")).unwrap()).unwrap();
    let parents: Vec<i32> = parsed.bones.iter().map(|b| b.parent).collect();
    let ids: Vec<glam::Mat4> = parsed.bones.iter().map(|b| b.local).collect();

    let names = ["upperTorso", "chest", "head", "helmet", "leftArm", "leftHand", "rightArm", "rightHand"];
    println!("== bone world translations ==");
    println!("bone   static");
    for (bi, b) in parsed.bones.iter().enumerate() {
        let (tx, ty, tz) = (
            b.world.w_axis.x,
            b.world.w_axis.y,
            b.world.w_axis.z,
        );
        println!("{bi:>3} {:<12} static=({:9.4},{:9.4},{:9.4}) id_scale={:.4}", b.name, tx, ty, tz, b.world.x_axis.length());
    }
    println!("\nanim");
    for f in [0.0, 5.0, 10.0, 15.0, 20.0] {
        let w = an3.bone_worlds(&parents, &ids, f).unwrap();
        println!("frame {f}:");
        for (bi, b) in parsed.bones.iter().enumerate() {
            if names.contains(&b.name.as_str()) {
                println!(
                    "  bone {bi:>3} {:<12} anim=({:9.4},{:9.4},{:9.4})",
                    b.name,
                    w[bi].w_axis.x,
                    w[bi].w_axis.y,
                    w[bi].w_axis.z
                );
            }
        }
    }
}
