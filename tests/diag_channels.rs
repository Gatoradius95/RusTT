use rustt::an3::An3;

#[test]
fn diag_channels() {
    let dir = "backup/CHARS/ANAKIN";
    for (label, fname) in [
        ("FORCECHOKE", "FORCECHOKE.AN3"),
        ("IDLE", "IDLE.AN3"),
        ("NOHANDIDLE", "NOHANDIDLE.AN3"),
    ] {
        let data = std::fs::read(format!("{dir}/{fname}")).unwrap();
        let an3 = An3::parse(&data).unwrap();
        println!(
            "== {label}: bones={} frames={} moving={} keyblock_len={} 4INA={} ==",
            an3.num_bones, an3.num_frames, an3.num_moving, an3.keyblock_len, an3.four_ina
        );
        for (idx, &bc) in an3.animated.iter().enumerate() {
            let bone = bc / 9;
            let ch = bc % 9;
            let (scale, offset) = an3.movpar[idx];
            let typ = an3.channel_types[idx];
            let v0 = an3.channel_value(bone, ch, 0.0);
            println!(
                "  anim[{idx:>2}] bone={bone:>2} ch={ch} type={} scale={:>12.5} off={:>12.5} f0={:>12.5}",
                if typ == 7 { "07" } else { "06" },
                scale,
                offset,
                v0
            );
        }
        println!();
    }
}
