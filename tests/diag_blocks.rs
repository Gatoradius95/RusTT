use rustt::an3::{An3, BlockVal};

fn low(v: &BlockVal) -> u8 {
    match v {
        BlockVal::Six(u) => (u & 0xff) as u8,
        BlockVal::Seven(u) => (u[0] & 0xff) as u8,
    }
}

fn dump(path: &str) {
    let data = std::fs::read(path).unwrap();
    let u16 = |o: usize| u16::from_le_bytes([data[o], data[o + 1]]);
    let u32 = |o: usize| u32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
    let f32 = |o: usize| f32::from_bits(u32(o));
    println!(
        "\n=== {path} len={} ===\n  NumFrames(0x06)={} CurveGroupSize(0x08)={} OriginalNumFrames(0x0a)={} NumCurves(0x0c)={}",
        data.len(),
        u16(0x06),
        u16(0x08),
        u16(0x0a),
        u16(0x0c)
    );
    println!(
        "  base_add={:.4} base_mul={:.6} movpar=0x{:x} static=0x{:x} matrix=0x{:x} movdata=0x{:x} footer=0x{:x}",
        f32(0x1c),
        f32(0x20),
        u32(0x24),
        u32(0x28),
        u32(0x2c),
        u32(0x30),
        u32(0x34)
    );
    let a = An3::parse(&data).unwrap();
    println!(
        "  parsed: num_frames(0x0a)={} blocks.len={} num_moving={} keyblock_len={}",
        a.num_frames,
        a.blocks.len(),
        a.num_moving,
        a.keyblock_len
    );

    // Raw block values for the first animated channel (raw u32) + low byte (base value).
    println!("  raw blocks (channel 0), and all channels' low base byte:");
    for (bi, row) in a.blocks.iter().enumerate() {
        let bases: Vec<String> = row
            .iter()
            .map(|v| match v {
                BlockVal::Six(u) => format!("{:02x}", u & 0xff),
                BlockVal::Seven(u) => format!("{:02x}", u[0] & 0xff),
            })
            .collect();
        let first = match row[0] {
            BlockVal::Six(u) => format!("0x{u:08x}"),
            BlockVal::Seven(u) => format!("{:?}", u),
        };
        println!(
            "    block {bi}: first={first} lowbyte(ch0)={:02x}  bases(ch0..={:?})",
            low(&row[0]),
            bases
        );
    }
    println!("  last 4 raw blocks of channel0 as u32:");
    for (bi, row) in a.blocks.iter().enumerate().skip(a.blocks.len().saturating_sub(4)) {
        if let BlockVal::Six(u) = row[0] {
            println!("    block {bi}: 0x{u:08x} base={:02x} weights3={:06x}", u & 0xff, u >> 8);
        }
    }
}

#[test]
fn dump_raw_blocks() {
    dump("backup/CHARS/ANAKIN/RUN.AN3");
    dump("backup/CHARS/ANAKIN/WALK.AN3");
    dump("backup/CHARS/ANAKIN/IDLE.AN3");
}
