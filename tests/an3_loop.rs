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
fn interpolation_is_smooth_with_no_snapping_jumps() {
    // The game's evaluator linearly interpolates between adjacent decoded
    // samples (round-to-nearest index + fractional lerp). A snapping sampler
    // holds each sample and then jumps by a full sample delta at the boundary,
    // which reads as ~1/30s judder. Verify fine-grained sampling never jumps
    // by more than a fraction of the largest sample-to-sample delta.
    for name in ["RUN.AN3", "WALK.AN3", "IDLE.AN3"] {
        let an3 = parse_anakin(name);
        let steps_per_sub = 64.0;

        for bone in 0..an3.num_bones {
            for chan in 0..9 {
                let max_delta: f32 = (0..an3.data_frames)
                    .map(|k| {
                        let a = an3.channel_value(bone, chan, k as f32);
                        let b = an3.channel_value(bone, chan, k as f32 + 1.0);
                        (b - a).abs()
                    })
                    .fold(0.0, f32::max);
                if max_delta < 1e-6 {
                    continue; // static/constant channel
                }

                // Fine-grained max jump over the whole data range.
                let n_points = (an3.data_frames as f32 * steps_per_sub) as usize;
                let mut max_jump = 0.0f32;
                let mut prev = an3.channel_value(bone, chan, 0.0);
                for p in 1..=n_points {
                    let f = p as f32 / steps_per_sub;
                    let v = an3.channel_value(bone, chan, f);
                    max_jump = max_jump.max((v - prev).abs());
                    prev = v;
                }

                // Interpolated rate is bounded by the slope per subframe; a
                // snapping sampler would jump by a full sample delta.
                let slope_budget = max_delta * (1.0 / steps_per_sub) * 2.5;
                assert!(
                    max_jump <= slope_budget,
                    "{name} bone {bone} ch {chan}: max fine jump {max_jump} > slope budget {slope_budget} (max sample delta {max_delta})"
                );
            }
        }
    }
}

#[test]
fn blend_is_endpoint_exact_and_in_between() {
    use rustt::an3::blended_bone_worlds;
    let a = parse_anakin("IDLE.AN3");
    let b = parse_anakin("WALK.AN3");
    let parents: Vec<i32> = (0..a.num_bones)
        .map(|i| if i == 0 { -1 } else { 0 })
        .collect();
    let rest = vec![glam::Mat4::IDENTITY; a.num_bones];
    let fa = a.remap_playhead(3.0);
    let fb = b.remap_playhead(3.0);

    let wa0 = a.bone_worlds(&parents, &rest, fa).expect("a worlds");
    let wb1 = b.bone_worlds(&parents, &rest, fb).expect("b worlds");

    // t = 0 is exactly clip a, t = 1 exactly clip b.
    let w0 = blended_bone_worlds(&a, &b, &parents, &rest, fa, fb, 0.0).expect("blend 0");
    let w1 = blended_bone_worlds(&a, &b, &parents, &rest, fa, fb, 1.0).expect("blend 1");
    assert!(pose_dist(&w0, &wa0) < 1e-4, "t=0 must match clip a");
    assert!(pose_dist(&w1, &wb1) < 1e-4, "t=1 must match clip b");

    // Mid-blend pose is finite and strictly between the two endpoints.
    let w05 = blended_bone_worlds(&a, &b, &parents, &rest, fa, fb, 0.5).expect("blend 0.5");
    assert_eq!(w05.len(), a.num_bones);
    for w in &w05 {
        assert!(w.is_finite(), "non-finite mid-blend world");
    }
    let d_ab = pose_dist(&wa0, &wb1);
    let d_a = pose_dist(&w05, &wa0);
    let d_b = pose_dist(&w05, &wb1);
    assert!(d_ab > 0.0, "clips are distinct");
    assert!(
        d_a < d_ab * 0.95 && d_b < d_ab * 0.95,
        "mid-blend pose must sit between the clips (a {d_a}, b {d_b}, ab {d_ab})"
    );

    // A longer crossfade sweep stays finite across all blend weights.
    for t in [0.1, 0.25, 0.5, 0.75, 0.9] {
        let ws = blended_bone_worlds(&a, &b, &parents, &rest, fa, fb, t).expect("blend");
        for w in &ws {
            assert!(w.is_finite(), "non-finite blend at t={t}");
        }
    }
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
