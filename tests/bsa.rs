//! BSA blend-shape animation tests: header decoding, channel classification,
//! and baked per-frame weight evaluation. Golden values cross-checked against a
//! standalone port of `bsa_addon.py` (`parse_bsa`/`bake_bsa`) on the same files.

use rustt::bsa::{Bsa, KEY_COMPRESSED, KEY_NONE};

fn parse_anakin(name: &str) -> Bsa {
    let dir = std::path::Path::new("backup/CHARS/ANAKIN");
    if !dir.exists() {
        eprintln!("skipping: backup assets not present");
        std::process::exit(0);
    }
    let data = std::fs::read(dir.join(name)).unwrap_or_else(|e| panic!("read {name}: {e}"));
    Bsa::parse(&data).unwrap_or_else(|e| panic!("parse {name}: {e}"))
}

#[test]
fn idle_header_fields() {
    let bsa = parse_anakin("IDLE.BSA");
    assert_eq!(bsa.length_in_frames, 249.0);
    assert_eq!(bsa.group_count, 2);
    assert_eq!(bsa.channels_per_group, 5);
    assert_eq!(bsa.interval_count, 8);
    assert_eq!(bsa.total_channels(), 10);
    assert_eq!(bsa.flags, vec![0, 0]);
}

#[test]
fn idle_channel_types() {
    let bsa = parse_anakin("IDLE.BSA");
    let types: Vec<u8> = bsa.channels.iter().map(|c| c.keyframe_type).collect();
    // channels 0,2,3,4 are compressed; the rest are constant.
    assert_eq!(types, vec![KEY_COMPRESSED, KEY_NONE, KEY_COMPRESSED, KEY_COMPRESSED, KEY_COMPRESSED, KEY_NONE, KEY_NONE, KEY_NONE, KEY_NONE, KEY_NONE]);
    // channel 0 holds 24 keyframes (matches its 32-bit mask population).
    assert_eq!(bsa.channels[0].keys.len(), 24);
}

#[test]
fn idle_ch0_golden_weights() {
    // First 8 frames of IDLE channel 0 (a blink-ish pulse), from the reference.
    let bsa = parse_anakin("IDLE.BSA");
    let baked = bsa.bake();
    assert_eq!(baked.len(), 10);
    assert_eq!(baked[0].len(), 249);
    let expected = [0.0, 0.259259, 0.740741, 1.0, 1.0, 1.0, 1.0, 0.740741];
    for (i, &exp) in expected.iter().enumerate() {
        let got = bsa.evaluate(0, i as f32);
        assert!((got - exp).abs() < 1e-4, "frame {i}: got {got}, expected {exp}");
    }
}

#[test]
fn idle_ch4_animates() {
    let bsa = parse_anakin("IDLE.BSA");
    let baked = bsa.bake();
    let v = &baked[4];
    // Constant channels hold one value; channel 4 is keyframed (0.4516..0.3871).
    assert!((v[0] - 0.451613).abs() < 1e-4, "ch4 frame 0: {}", v[0]);
    assert!(v[0] != v[124], "ch4 must not be constant ({} vs {})", v[0], v[124]);
    for x in v {
        assert!(x.is_finite() && (0.0..=1.0).contains(x), "ch4 out of range: {x}");
    }
}

#[test]
fn run_all_constant() {
    let bsa = parse_anakin("RUN.BSA");
    assert_eq!(bsa.length_in_frames, 24.0);
    assert_eq!(bsa.group_count, 3);
    assert_eq!(bsa.channels_per_group, 6);
    assert_eq!(bsa.interval_count, 1);
    assert!(bsa.channels.iter().all(|c| c.keyframe_type == KEY_NONE));
    let baked = bsa.bake();
    assert_eq!(baked.len(), 18);
    for (i, vals) in baked.iter().enumerate() {
        assert_eq!(vals.len(), 24);
        assert!(vals.windows(2).all(|w| w[0] == w[1]), "channel {i} not constant");
        assert!(vals[0].is_finite(), "channel {i} non-finite");
    }
    // The only non-zero constant in RUN.BSA is channel 3 (weight 1.0).
    assert_eq!(bsa.channels[3].constant_value, 1.0);
    assert_eq!(bsa.evaluate(3, 12.0), 1.0);
}

#[test]
fn walk_constant_channels() {
    let bsa = parse_anakin("WALK.BSA");
    assert_eq!(bsa.length_in_frames, 30.0);
    assert_eq!(bsa.total_channels(), 15);
    assert_eq!(bsa.channels[4].constant_value, 0.4301);
    assert_eq!(bsa.evaluate(4, 29.0), 0.4301);
}

#[test]
fn forcechoke_single_group() {
    let bsa = parse_anakin("FORCECHOKE.BSA");
    assert_eq!(bsa.length_in_frames, 20.0);
    assert_eq!(bsa.group_count, 1);
    assert_eq!(bsa.channels_per_group, 20);
    assert!(bsa.channels.iter().all(|c| c.keyframe_type == KEY_NONE));
}
