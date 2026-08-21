use rustt::an3::An3;
use rustt::ghg;

#[test]
fn diag_flags() {
    let dir = "backup/CHARS/ANAKIN";
    let ghg_data = std::fs::read(format!("{dir}/ANAKIN_PADAWAN_PC.GHG")).unwrap();
    let parsed = ghg::parse(&ghg_data).unwrap();
    for (label, fname) in [
        ("IDLE", "IDLE.AN3"),
        ("FORCECHOKE", "FORCECHOKE.AN3"),
        ("NOHANDIDLE", "NOHANDIDLE.AN3"),
        ("RUN", "RUN.AN3"),
        ("BACKFLIP", "BACKFLIP.AN3"),
    ] {
        let Ok(data) = std::fs::read(format!("{dir}/{fname}")) else {
            continue;
        };
        let Ok(an3) = An3::parse(&data) else {
            continue;
        };
        println!("== {label} 4INA={} bones={} ==", an3.four_ina, an3.num_bones);
        for (bi, b) in parsed.bones.iter().enumerate() {
            let f = an3.footer.get(bi).copied().unwrap_or(0);
            if f != 0 {
                println!("  bone {bi:>2} {:<14} footer={:#04x} x20={} 01={}", b.name, f, f & 0x20 != 0, f & 0x01 != 0);
            }
        }
    }
}
