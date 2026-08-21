//! Playback-timeline regression tests: the viewer plays over the 0x0A
//! timeline (`num_frames`) while sampling the 0x06 data (`data_frames`)
//! stretched via `remap_playhead`. The loop must close at the last timeline
//! frame, and no playback frame may sample padding subframes (the zero-padded
//! tail that used to show as frozen/garbage frames).

use rustt::an3::An3;

fn parse_anakin(name: &str) -> An3 {
    let dir = std::path::Path::new("backup/CHARS/ANAKIN");
    if !dir.exists() {
        eprintln!("skipping: backup assets not present");
        std::process::exit(0);
    }
    let data = std::fs::read(dir.join(name)).unwrap_or_else(|e| panic!("read {name}: {e}"));
    An3::parse(&data).unwrap_or_else(|e| panic!("parse {name}: {e}"))
}

/// Sum of joint-position differences between two poses (world matrices).
fn pose_dist(a: &[glam::Mat4], b: &[glam::Mat4]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(wa, wb)| (wa.col(3).truncate() - wb.col(3).truncate()).length())
        .sum()
}

#[test]
fn playhead_maps_to_data_range() {
    for name in ["RUN.AN3", "WALK.AN3", "IDLE.AN3"] {
        let an3 = parse_anakin(name);
        assert!(an3.num_frames >= an3.data_frames, "{name}: timeline shorter than data");
        assert_eq!(an3.remap_playhead(0.0), 0.0, "{name}");
        let end = (an3.num_frames - 1) as f32;
        assert!(
            (an3.remap_playhead(end) - (an3.data_frames - 1) as f32).abs() < 1e-4,
            "{name}: last playhead must land on the loop-closing data frame"
        );
    }
}

#[test]
fn header_fields_are_originalnumframes_and_numframes() {
    // RUN/WALK/IDLE have the classic 2x relationship: 0x0A == 2 * 0x06.
    let run = parse_anakin("RUN.AN3");
    assert_eq!(run.num_frames, 24);
    assert_eq!(run.data_frames, 12);
    let walk = parse_anakin("WALK.AN3");
    assert_eq!(walk.num_frames, 30);
    assert_eq!(walk.data_frames, 15);
    let idle = parse_anakin("IDLE.AN3");
    assert_eq!(idle.num_frames, 249);
    assert_eq!(idle.data_frames, 124);
}

#[test]
fn loop_seam_is_seamless_and_no_garbage_at_tail() {
    for name in ["RUN.AN3", "WALK.AN3", "IDLE.AN3"] {
        let an3 = parse_anakin(name);
        let parents: Vec<i32> = (0..an3.num_bones)
            .map(|i| if i == 0 { -1 } else { 0 })
            .collect();
        let rest = vec![glam::Mat4::IDENTITY; an3.num_bones];

        // Pose at playhead 0 vs playhead num_frames-1 must match (seamless loop).
        let w0 = an3
            .bone_worlds(&parents, &rest, an3.remap_playhead(0.0))
            .expect("worlds@0");
        let wend = an3
            .bone_worlds(&parents, &rest, an3.remap_playhead((an3.num_frames - 1) as f32))
            .expect("worlds@end");
        let d = pose_dist(&w0, &wend);
        assert!(d < 0.02, "{name}: loop seam too large ({d})");

        // Every playhead must produce finite worlds and stay on real data.
        let n = an3.num_frames as f32;
        for p in [0.0, n * 0.25, n * 0.5, n * 0.75, n - 1.0] {
            let data_f = an3.remap_playhead(p);
            assert!(
                data_f <= (an3.data_frames - 1) as f32 + 1e-4,
                "{name}: playhead {p} sampled past real data ({data_f})"
            );
            let ws = an3
                .bone_worlds(&parents, &rest, data_f)
                .expect("bone_worlds");
            assert_eq!(ws.len(), an3.num_bones);
            for w in &ws {
                assert!(w.is_finite(), "{name}: non-finite world at playhead {p}");
            }
        }
    }
}
