//! Regression test: the GPU skinning path must reproduce the old CPU formulas.
//!
//! Rigid parts (no per-vertex skin block) store vertices in the bone's LOCAL
//! rest frame and were posed by `anim_world[bone]` directly. The GPU shader
//! only has the shared skin matrix `anim_world[i] * rest[i]^-1`, so the vertex
//! builder bakes `rest[bone]` into the rigid vertices. This test pins the
//! identity `(anim*rest^-1) * (rest*v) == anim*v` over the real ANAKIN file.
//! Skinned parts use `(Σ wᵢ·Mᵢ)·p ≡ Σ wᵢ·(Mᵢ·p)` (linearity), which the shader
//! relies on too and is checked here for one vertex.

use glam::{Mat4, Vec3};
use rustt::ghg;
use rustt::glb;

/// Plausible animated worlds: each bone's rest local rotated slightly and
/// accumulated through the skeleton. The identity under test is pure algebra,
/// so any non-trivial worlds suffice.
fn anim_worlds(parsed: &ghg::Parsed) -> Vec<Mat4> {
    let twist = Mat4::from_rotation_y(0.35) * Mat4::from_rotation_x(-0.15);
    let mut out = Vec::with_capacity(parsed.bones.len());
    for b in &parsed.bones {
        let local = b.local * twist;
        let w = match b.parent {
            -1 => local,
            p if (p as usize) < out.len() => out[p as usize] * local,
            _ => local,
        };
        out.push(w);
    }
    out
}

#[test]
fn rigid_gpu_formula_matches_old_cpu_skin() {
    let data = std::fs::read("backup/CHARS/ANAKIN/ANAKIN_PADAWAN_PC.GHG").unwrap();
    let parsed = ghg::parse(&data).unwrap();
    let raw = glb::build_meshes(&parsed);
    let rest: Vec<Mat4> = parsed.bones.iter().map(|b| b.world).collect();
    let worlds = anim_worlds(&parsed);
    let mut rigid_checked = 0;

    for (i, item) in parsed.render.iter().enumerate() {
        let Some(md) = raw.get(i) else { continue };
        if md.pos.is_empty() {
            continue;
        }
        let skinned = !md.skin.is_empty() && !md.skin_bones.is_empty();
        if skinned || item.bone < 0 {
            continue;
        }
        rigid_checked += 1;
        let bone = item.bone as usize;
        let anim = worlds[bone];
        let r = rest[bone];
        let skin = anim * r.inverse();
        for (v, pos) in md.pos.iter().enumerate().take(8) {
            let v_local = Vec3::from(*pos);
            let old = anim.transform_point3(v_local);
            let new = skin.transform_point3(r.transform_point3(v_local));
            assert!(
                (old - new).length() < 1e-4,
                "item {i} bone {bone} vert {v}: cpu {old:?} != gpu {new:?}"
            );
        }
    }
    assert!(rigid_checked >= 10, "expected several rigid parts, got {rigid_checked}");
}

#[test]
fn skinned_gpu_formula_matches_old_cpu_skin() {
    let data = std::fs::read("backup/CHARS/ANAKIN/ANAKIN_PADAWAN_PC.GHG").unwrap();
    let parsed = ghg::parse(&data).unwrap();
    let raw = glb::build_meshes(&parsed);
    let rest: Vec<Mat4> = parsed.bones.iter().map(|b| b.world).collect();
    let worlds = anim_worlds(&parsed);
    let mut skinned_checked = 0;

    for (i, item) in parsed.render.iter().enumerate() {
        let Some(md) = raw.get(i) else { continue };
        let skinned = !md.skin.is_empty()
            && !md.skin_bones.is_empty()
            && md.skin.len() >= md.pos.len() * 8;
        if !skinned {
            continue;
        }
        skinned_checked += 1;
        let v = 0usize;
        let p = Vec3::from(md.pos[v]);
        // Old CPU path: sum weighted per-influence results.
        let mut old = Vec3::ZERO;
        // New GPU path: weighted matrix blend, then one transform.
        let mut msum = Mat4::ZERO;
        for k in 0..4 {
            let sw = md.skin[v * 8 + k] as f32 / 255.0;
            let li = md.skin[v * 8 + 4 + k] as usize;
            let Some(&g) = md.skin_bones.get(li) else { continue };
            let m = worlds[g as usize] * rest[g as usize].inverse();
            old += sw * m.transform_point3(p);
            msum += sw * m;
        }
        let new = msum.transform_point3(p);
        assert!(
            (old - new).length() < 1e-3,
            "item {i} vert 0: cpu {old:?} != gpu {new:?}"
        );
    }
    assert!(skinned_checked >= 10, "expected several skinned parts, got {skinned_checked}");
}
