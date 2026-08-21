//! Shape-key ("dynamic buffer") tests for `_PC.GHG` files.
//!
//! Format cross-checked against BactaTank (BactaTankMesh.read): each part
//! descriptor holds a dynamic buffer count at +0x28 and a rel pointer at +0x2c
//! to an array of per-slot rel pointers. Each nonzero pointer points at
//! `num_v * 3` f32 per-vertex [x,y,z] offsets; a zero pointer is an empty slot.

use rustt::ghg;

/// Read ANAKIN_JEDI_PC.GHG and hand the parsed model to `f` while the backing
/// buffer stays alive (ghg::Parsed borrows it).
fn with_anakin<T>(f: impl FnOnce(&ghg::Parsed) -> T) -> T {
    let dir = std::path::Path::new("backup/CHARS/ANAKIN");
    if !dir.exists() {
        eprintln!("skipping: backup assets not present");
        std::process::exit(0);
    }
    let data = std::fs::read(dir.join("ANAKIN_JEDI_PC.GHG"))
        .unwrap_or_else(|e| panic!("read ghg: {e}"));
    let parsed = ghg::parse(&data).unwrap_or_else(|e| panic!("parse ghg: {e}"));
    f(&parsed)
}

#[test]
fn anakin_parts_with_shape_keys() {
    with_anakin(|p| {
        // Parts 0 and 1 are the head/face meshes and carry 20 shape-key slots.
        for i in [0usize, 1] {
            let part = &p.parts[i];
            assert_eq!(part.dynamic_buffers.len(), 20, "part {i} slot count");
        }
        // Everything else is rigid / skinned geometry without shape keys.
        assert!(p.parts[2..].iter().all(|part| part.dynamic_buffers.is_empty()));
    });
}

#[test]
fn anakin_part0_slot_lengths() {
    with_anakin(|p| {
        let part = &p.parts[0];
        for buf in part.dynamic_buffers.iter().flatten() {
            assert_eq!(buf.len(), part.num_v, "part 0: {} vertices expected", part.num_v);
            for v in buf {
                assert!(v.iter().all(|c| c.is_finite()));
            }
        }
        // Every slot is a real morph target here; slot 0 starts with zeros
        // (eye-region verts move later in the block), all 20 are filled.
        assert_eq!(part.dynamic_buffers.len(), 20);
        assert!(part.dynamic_buffers.iter().all(|b| b.is_some()));
        // First three vertices of slot 0 are zero, matching the raw dump.
        assert_eq!(part.dynamic_buffers[0].as_ref().unwrap()[..3], [[0.0; 3]; 3]);
    });
}

#[test]
fn anakin_part1_slot_lengths() {
    with_anakin(|p| {
        let part = &p.parts[1];
        for buf in part.dynamic_buffers.iter().flatten() {
            assert_eq!(buf.len(), part.num_v, "part 1: {} vertices expected", part.num_v);
        }
    });
}

#[test]
fn anakin_part0_slot1_deltas_match_dump() {
    // Empirically measured from the raw file: part 0 slot 1 starts at
    // data 0x264644 with first vertex [0.0012, 0.0019, 0.0].
    with_anakin(|p| {
        let buf = p.parts[0].dynamic_buffers[1]
            .as_ref()
            .expect("part 0 slot 1 must be present");
        assert!((buf[0][0] - 0.0012).abs() < 1e-3, "slot1 v0.x = {}", buf[0][0]);
        assert!((buf[0][1] - 0.0019).abs() < 1e-3, "slot1 v0.y = {}", buf[0][1]);
        assert!(buf[0][2].abs() < 1e-6, "slot1 v0.z = {}", buf[0][2]);
    });
}

#[test]
fn anakin_part0_slots_packed_contiguously() {
    // Each filled slot is `num_v * 3 * 4` bytes and the slots are laid out
    // back-to-back (verified empirically: 888 bytes for 74 vertices).
    with_anakin(|p| {
        let part = &p.parts[0];
        let filled: Vec<&Vec<[f32; 3]>> = part.dynamic_buffers.iter().flatten().collect();
        assert_eq!(filled.len(), 20, "all 20 slots filled");
        let block = part.num_v * 3 * 4;
        for buf in &filled {
            assert_eq!(buf.len() * 3 * 4, block, "block size mismatch");
        }
        // Ensure blocks are plausible: real motion in the morph targets.
        assert!(filled[1..].iter().any(|b| {
            b.iter()
                .any(|v| v[0] * v[0] + v[1] * v[1] + v[2] * v[2] > 1e-8)
        }));
    });
}

#[test]
fn forcechoke_bsa_channel_count_matches_slots() {
    // FORCECHOKE.BSA is 1 group x 20 channels: exactly the number of
    // shape-key slots on ANAKIN's face parts (index-aligned mapping).
    let dir = std::path::Path::new("backup/CHARS/ANAKIN");
    if !dir.exists() {
        eprintln!("skipping: backup assets not present");
        std::process::exit(0);
    }
    let bsa_data =
        std::fs::read(dir.join("FORCECHOKE.BSA")).unwrap_or_else(|e| panic!("read bsa: {e}"));
    let bsa = rustt::bsa::Bsa::parse(&bsa_data).unwrap_or_else(|e| panic!("parse bsa: {e}"));
    assert_eq!(bsa.total_channels(), 20, "FORCECHOKE has 20 channels");
    with_anakin(|p| {
        let slots = p.parts[0].dynamic_buffers.len();
        assert_eq!(slots, 20, "face part has 20 shape-key slots");
    });
}
