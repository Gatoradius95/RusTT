use anyhow::Result;
use glam::{Mat4, Vec3};
use rustt::ghg;
use rustt::glb::{self, MeshData};

fn skin8(m: &MeshData, v: usize) -> [u8; 8] {
    m.skin[v * 8..v * 8 + 8].try_into().unwrap()
}

/// Per-vertex skin: weight the transform of each valid bone.
fn blend(
    pos: &[f32; 3],
    nrm: &[f32; 3],
    s: &[u8],
    skin_bones: &[u16],
    mats: &[Mat4],
) -> ([f32; 3], [f32; 3]) {
    let mut acc = Vec3::ZERO;
    let mut nacc = Vec3::ZERO;
    let mut tot = 0.0f32;
    let mut ntot = 0.0f32;
    for k in 0..4 {
        let li = s[4 + k] as usize;
        let w = s[k] as f32;
        if li >= mats.len() || w <= 0.0 {
            continue;
        }
        tot += w;
        acc += mats[li].transform_point3(Vec3::from(*pos)) * w;
        ntot += w;
        nacc += mats[li].inverse().transpose().transform_vector3(Vec3::from(*nrm)) * w;
    }
    let out = if tot > 0.0 {
        (acc / tot).to_array()
    } else {
        *pos
    };
    let on = if ntot > 0.0 {
        nacc.normalize_or_zero().to_array()
    } else {
        *nrm
    };
    (out, on)
}

#[test]
fn skin_extraction_matches_raw() -> Result<()> {
    let data = std::fs::read("backup/CHARS/ANAKIN/ANAKIN_PADAWAN_PC.GHG")?;
    let p = ghg::parse(&data)?;
    let meshes = glb::build_meshes(&p);

    // Part 5 = legs, stride 44. From diag_skin: skin_bones [23..29], vertex 0
    // weights [f4 0b 00 ff], indices [02 01 00 ff].
    let item5 = p.render.iter().position(|r| r.part == 5).unwrap();
    let md = &meshes[item5];
    assert_eq!(md.skin.len(), md.pos.len() * 8);
    assert_eq!(&md.skin_bones[..], &[23, 24, 25, 26, 27, 28, 29]);
    assert_eq!(&md.skin[0..8], &[0xf4, 0x0b, 0x00, 0xff, 0x02, 0x01, 0x00, 0xff]);

    // Part 9 = belt, stride 40. From diag_skin: skin_bones [7,8,13,14], vertex
    // 0 weights [ff 00 00 ff], indices [x 00 00 ff] with x < 4.
    let item9 = p.render.iter().position(|r| r.part == 9).unwrap();
    let md9 = &meshes[item9];
    assert_eq!(md9.skin.len(), md9.pos.len() * 8);
    assert_eq!(&md9.skin_bones[..], &[7, 8, 13, 14]);
    assert_eq!(&md9.skin[0..4], &[0xff, 0x00, 0x00, 0xff]);
    assert!(md9.skin[4] < 4, "local index 0 of belt v0 must be < 4");
    assert_eq!(&md9.skin[5..8], &[0x00, 0x00, 0xff]);

    // Non-skinned parts (2,3,4 stride 36/32) must have no skin block and no
    // skin bones, so the viewer falls back to rigid single-bone skinning.
    for part in [2usize, 3, 4] {
        let item = p.render.iter().position(|r| r.part == part).unwrap();
        let md = &meshes[item];
        assert!(md.skin.is_empty(), "part {part} should not carry skin data");
        assert!(md.skin_bones.is_empty(), "part {part} should have no skin bones");
    }

    // Cross-check every part: skin_bones presence must agree with skin data.
    for (pi, part) in p.parts.iter().enumerate() {
        let Some(item) = p.render.iter().position(|r| r.part == pi) else {
            continue;
        };
        let md = &meshes[item];
        assert_eq!(
            !md.skin_bones.is_empty(),
            !md.skin.is_empty(),
            "part {pi} skin_bones/skin mismatch"
        );
        if !md.skin.is_empty() {
            assert_eq!(md.skin.len(), md.pos.len() * 8, "part {pi} skin length");
            let locals = md.skin[4..].iter().step_by(8);
            for (k, &li) in locals.enumerate() {
                assert!(
                    li == 0xff || (li as usize) < md.skin_bones.len(),
                    "part {pi} vertex {k} local index {li} out of range"
                );
            }
        }
    }

    // Rest-pose identity: with bone_worlds = the model's own rest worlds, the
    // per-vertex skin matrices must be identity, so vertices stay put.
    let rest: Vec<Mat4> = p.bones.iter().map(|b| b.world).collect();
    for (ri, item) in p.render.iter().enumerate() {
        let md = &meshes[ri];
        if md.skin.is_empty() || md.skin_bones.is_empty() {
            continue;
        }
        let mats: Vec<Mat4> = md
            .skin_bones
            .iter()
            .map(|&b| rest[b as usize] * rest[b as usize].inverse())
            .collect();
        for v in 0..md.pos.len() {
            let (pos, _) = blend(&md.pos[v], &md.nrm[v], &skin8(md, v), &md.skin_bones, &mats);
            for c in 0..3 {
                assert!(
                    (pos[c] - md.pos[v][c]).abs() < 1e-4,
                    "part {} vertex {v} moved at rest: {} -> {}",
                    item.part,
                    md.pos[v][c],
                    pos[c]
                );
            }
        }
    }
    Ok(())
}

