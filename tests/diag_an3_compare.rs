use anyhow::Result;
use glam::{Mat4, Vec3};
use rustt::an3::An3;
use rustt::ghg;
use rustt::glb::{self, MeshData};

fn joint(w: &Mat4) -> Vec3 {
    w.transform_point3(Vec3::ZERO)
}

fn blend_pos(pos: &[f32; 3], s: &[u8], skin_bones: &[u16], mats: &[Mat4]) -> [f32; 3] {
    let mut acc = Vec3::ZERO;
    let mut tot = 0.0f32;
    for k in 0..4 {
        let li = s[4 + k] as usize;
        let w = s[k] as f32;
        if li >= mats.len() || w <= 0.0 {
            continue;
        }
        tot += w;
        acc += mats[li].transform_point3(Vec3::from(*pos)) * w;
    }
    if tot > 0.0 {
        (acc / tot).to_array()
    } else {
        *pos
    }
}

#[test]
fn choke_skinned_mesh_distortion() -> Result<()> {
    let data = std::fs::read("backup/CHARS/ANAKIN/ANAKIN_PADAWAN_PC.GHG")?;
    let p = ghg::parse(&data)?;
    let rest_locals: Vec<Mat4> = p.bones.iter().map(|b| b.local).collect();
    let rest_worlds: Vec<Mat4> = p.bones.iter().map(|b| b.world).collect();
    let meshes = glb::build_meshes(&p);

    // Compare IDLE vs FORCECHOKE: max edge stretch over the whole animation,
    // per part, using rest_len > 0.01 edges only (ignore slivers).
    for name in ["IDLE", "FORCECHOKE"] {
        let an3 = An3::parse(&std::fs::read(format!("backup/CHARS/ANAKIN/{name}.AN3"))?)?;
        let parents: Vec<i32> = (0..an3.num_bones)
            .map(|i| p.bones.get(i).map(|b| b.parent.min(an3.num_bones as i32 - 1)).unwrap_or(-1))
            .collect();
        println!("\n=== {name}: worst edge stretch per part (rest_len>0.01) ===");

        // First check the NEUTRAL pose (all animated channels at their offset).
        {
            let worlds = neutral_worlds(&an3, &parents, &rest_locals);
            let stretch = max_stretch(&meshes, &worlds, &rest_worlds);
            println!("  [neutral] worst stretch per part:");
            for (pi, s) in stretch.iter().enumerate() {
                if *s > 1.5 {
                    println!(
                        "    part {} (render {pi}): verts {}  stretch {:.2}x",
                        p.render[pi].part,
                        meshes[pi].pos.len(),
                        s
                    );
                }
            }
        }

        // Then across all animation frames.
        for frame in 0..an3.num_frames {
            let worlds = an3.bone_worlds(&parents, &rest_locals, frame as f32)?;
            let stretch = max_stretch(&meshes, &worlds, &rest_worlds);
            for (pi, s) in stretch.iter().enumerate() {
                if *s > 1.5 {
                    println!(
                        "  part {} (render {pi}): verts {}  frame {frame:>2}  worst_stretch {:.2}x",
                        p.render[pi].part,
                        meshes[pi].pos.len(),
                        s
                    );
                }
            }
        }
    }

    // Compare IDLE vs CHOKE static rotation channel values for the left arm
    // chain, and the model's own rest local rotation, to see whether the
    // AN3 rotation is meant to be a small delta or an absolute pose.
    println!("\n=== left-arm chain: model rest local rot vs AN3 channel values ===");
    for i in [4usize, 5, 6, 7, 8, 9] {
        let m = Mat4::from_mat3(glam::Mat3::from_mat4(rest_locals[i]));
        let q = glam::Quat::from_mat4(&m);
        let (axis, ang) = q.to_axis_angle();
        println!(
            "{:>2} {:<20} model_rest axis {:+.3},{:+.3},{:+.3} ang {:.1}deg",
            i,
            p.bones.get(i).map(|b| b.name.as_str()).unwrap_or("?"),
            axis.x,
            axis.y,
            axis.z,
            ang.to_degrees()
        );
        for name in ["IDLE", "FORCECHOKE"] {
            let an3 = An3::parse(&std::fs::read(format!("backup/CHARS/ANAKIN/{name}.AN3"))?)?;
            let vals: Vec<f32> = (3..6).map(|c| an3.neutral(i, c)).collect();
            let qq = glam::Quat::from_mat4(
                &(Mat4::from_rotation_z(vals[2])
                    * Mat4::from_rotation_y(vals[1])
                    * Mat4::from_rotation_x(vals[0])),
            );
            let (aa, aa_ang) = qq.to_axis_angle();
            println!(
                "    {name:<10} an3_rx ry rz [{:+.4},{:+.4},{:+.4}]  => axis {:+.3},{:+.3},{:+.3} ang {:.1}deg",
                vals[0],
                vals[1],
                vals[2],
                aa.x,
                aa.y,
                aa.z,
                aa_ang.to_degrees()
            );
        }
    }
    Ok(())
}

#[test]
fn choke_composition_variants() -> Result<()> {
    let data = std::fs::read("backup/CHARS/ANAKIN/ANAKIN_PADAWAN_PC.GHG")?;
    let p = ghg::parse(&data)?;
    let rest_locals: Vec<Mat4> = p.bones.iter().map(|b| b.local).collect();
    let rest_worlds: Vec<Mat4> = p.bones.iter().map(|b| b.world).collect();
    let meshes = glb::build_meshes(&p);
    let an3 = An3::parse(&std::fs::read("backup/CHARS/ANAKIN/FORCECHOKE.AN3")?)?;
    let parents: Vec<i32> = (0..an3.num_bones)
        .map(|i| p.bones.get(i).map(|b| b.parent.min(an3.num_bones as i32 - 1)).unwrap_or(-1))
        .collect();

    for mode in ["composed", "absolute", "r_anim_then_rest"] {
        let worlds = worlds_with_mode(&an3, &parents, &rest_locals, mode, 10.0);
        let stretch = max_stretch(&meshes, &worlds, &rest_worlds);
        let w = stretch.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        println!(
            "\n=== FORCECHOKE mode {mode}: global worst stretch {w:.2}x ===");
        for (pi, s) in stretch.iter().enumerate() {
            if *s > 1.5 {
                println!(
                    "    part {} (render {pi}): verts {}  stretch {:.2}x",
                    p.render[pi].part,
                    meshes[pi].pos.len(),
                    s
                );
            }
        }
        // Print the left-arm chain joint positions under this mode.
        let chain = [4usize, 5, 6, 7, 8, 9];
        println!("    left-arm joints:");
        let mut root = Mat4::IDENTITY;
        for &b in &chain {
            let mut w = root;
            let t = glam::Vec3::new(
                an3.channel_value(b, 0, 10.0),
                an3.channel_value(b, 1, 10.0),
                -an3.channel_value(b, 2, 10.0),
            );
            let r = match mode {
                "composed" => {
                    let rl = Mat4::from_mat3(glam::Mat3::from_mat4(rest_locals[b]));
                    let ra = Mat4::from_rotation_z(an3.channel_value(b, 5, 10.0))
                        * Mat4::from_rotation_y(an3.channel_value(b, 4, 10.0))
                        * Mat4::from_rotation_x(an3.channel_value(b, 3, 10.0));
                    if an3.uses_x20(b) {
                        rl * ra
                    } else {
                        ra
                    }
                }
                "absolute" => {
                    Mat4::from_rotation_z(an3.channel_value(b, 5, 10.0))
                        * Mat4::from_rotation_y(an3.channel_value(b, 4, 10.0))
                        * Mat4::from_rotation_x(an3.channel_value(b, 3, 10.0))
                }
                _ => {
                    let rl = Mat4::from_mat3(glam::Mat3::from_mat4(rest_locals[b]));
                    let ra = Mat4::from_rotation_z(an3.channel_value(b, 5, 10.0))
                        * Mat4::from_rotation_y(an3.channel_value(b, 4, 10.0))
                        * Mat4::from_rotation_x(an3.channel_value(b, 3, 10.0));
                    if an3.uses_x20(b) {
                        ra * rl
                    } else {
                        ra
                    }
                }
            };
            w *= Mat4::from_translation(t) * r;
            let j = joint(&w);
            println!(
                "    bone {:>2} {:>13}  joint {:+.3},{:+.3},{:+.3}",
                b,
                p.bones.get(b).map(|b| b.name.as_str()).unwrap_or("?"),
                j.x,
                j.y,
                j.z
            );
            root = w;
        }
    }
    Ok(())
}

#[test]
fn choke_reference_formula() -> Result<()> {
    let data = std::fs::read("backup/CHARS/ANAKIN/ANAKIN_PADAWAN_PC.GHG")?;
    let p = ghg::parse(&data)?;
    let rest_locals: Vec<Mat4> = p.bones.iter().map(|b| b.local).collect();
    let identities: Vec<Mat4> = p.bones.iter().map(|b| b.identity).collect();
    let rest_worlds: Vec<Mat4> = p.bones.iter().map(|b| b.world).collect();
    let meshes = glb::build_meshes(&p);

    // Is `identity` related to `local` for the left-arm chain?
    println!("\n=== identity vs local for left-arm chain ===");
    for i in [4usize, 5, 6, 7, 8, 9] {
        let id3 = glam::Mat3::from_mat4(identities[i]);
        let loc3 = glam::Mat3::from_mat4(rest_locals[i]);
        let inv_loc = loc3.inverse();
        let n1 = (id3 - loc3).x_axis.length() + (id3 - loc3).y_axis.length() + (id3 - loc3).z_axis.length();
        let d2 = id3 - inv_loc;
        let n2 = d2.x_axis.length() + d2.y_axis.length() + d2.z_axis.length();
        println!(
            "  bone {i:>2} {:<16} |id-local|={n1:.4} |id-local^-1|={n2:.4}",
            p.bones.get(i).map(|b| b.name.as_str()).unwrap_or("?"),
        );
    }

    // Reference prediction (addon Blender convention): the addon builds rest
    // matrices as transpose(local) and needs inv(rest_local_B)*inv(identity)*
    // euler(rot_rest) == rest_local_B at rest. In file space this implies
    // euler(rot_rest) == identity_raw * transpose(rest_local)^2.
    println!("\n=== P1: euler(rot_rest) vs identity_raw * transpose(rest_local)^2 (IDLE 0x20 bones) ===");
    let idle = An3::parse(&std::fs::read("backup/CHARS/ANAKIN/IDLE.AN3")?)?;
    for i in 0..idle.num_bones {
        if !idle.uses_x20(i) {
            continue;
        }
        let vals: Vec<f32> = (3..6).map(|c| idle.neutral(i, c)).collect();
        let e = Mat4::from_rotation_z(vals[2])
            * Mat4::from_rotation_y(vals[1])
            * Mat4::from_rotation_x(vals[0]);
        let id3 = glam::Mat3::from_mat4(identities[i]);
        let loc3 = glam::Mat3::from_mat4(rest_locals[i]);
        let predicted = id3 * loc3.transpose() * loc3.transpose();
        let qe = glam::Quat::from_mat3(&glam::Mat3::from_mat4(e));
        let qp = glam::Quat::from_mat3(&predicted);
        let d = qe.dot(qp).abs().min(1.0);
        let ang = (2.0 * d.acos()).to_degrees();
        println!(
            "  bone {i:>2} {:<16} rot_rest [{:+.4},{:+.4},{:+.4}] vs id*T(local)^2: {ang:>6.2}deg",
            p.bones.get(i).map(|b| b.name.as_str()).unwrap_or("?"),
            vals[0], vals[1], vals[2],
        );
    }

    // Composition modes over IDLE frame 0 / FORCECHOKE frame 10: stretch + max
    // vertex delta from the rest mesh (delta measures pose fidelity).
    let parents: Vec<i32> = (0..idle.num_bones)
        .map(|i| p.bones.get(i).map(|b| b.parent.min(idle.num_bones as i32 - 1)).unwrap_or(-1))
        .collect();
    for name in ["IDLE", "FORCECHOKE"] {
        let a = An3::parse(&std::fs::read(format!("backup/CHARS/ANAKIN/{name}.AN3"))?)?;
        for frame in [0.0f32, 10.0] {
            if (frame as usize) >= a.num_frames {
                continue;
            }
            println!("\n=== {name} frame {frame}: pose fidelity per mode ===");
            for mode in ["composed", "reference", "addonB", "addonB2"] {
                let worlds = worlds_ref_mode(&a, &parents, &rest_locals, &identities, mode, frame);
                let stretch = max_stretch(&meshes, &worlds, &rest_worlds);
                let w = stretch.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut maxd = 0.0f32;
                let mut badparts: Vec<usize> = Vec::new();
                for (pi, md) in meshes.iter().enumerate() {
                    if md.skin.is_empty() {
                        continue;
                    }
                    let mats: Vec<Mat4> = md
                        .skin_bones
                        .iter()
                        .map(|&b| worlds[b as usize] * rest_worlds[b as usize].inverse())
                        .collect();
                    for v in 0..md.pos.len() {
                        let sw = &md.skin[v * 8..v * 8 + 8];
                        let posed = Vec3::from(blend_pos(&md.pos[v], sw, &md.skin_bones, &mats));
                        let d = (posed - Vec3::from(md.pos[v])).length();
                        if d > maxd {
                            maxd = d;
                        }
                    }
                    if stretch[pi] > 1.5 {
                        badparts.push(pi);
                    }
                }
                println!(
                    "  {mode:<18} worst_stretch {w:.2}x  max_vertex_delta {maxd:.3}  parts>1.5x: {:?}",
                    badparts
                );
            }
        }
    }

    // Left-arm joint positions under the reference formula.
    let choke = An3::parse(&std::fs::read("backup/CHARS/ANAKIN/FORCECHOKE.AN3")?)?;
    println!("\n=== FORCECHOKE frame 10: left-arm joints per mode ===");
    for mode in ["composed", "reference"] {
        let worlds = worlds_ref_mode(&choke, &parents, &rest_locals, &identities, mode, 10.0);
        println!("  mode {mode}:");
        for b in [0usize, 3, 4, 5, 6, 7, 8, 9] {
            let j = joint(&worlds[b]);
            println!(
                "    bone {b:>2} {:<15} joint {:+.3},{:+.3},{:+.3}",
                p.bones.get(b).map(|b| b.name.as_str()).unwrap_or("?"),
                j.x,
                j.y,
                j.z
            );
        }
    }
    Ok(())
}

#[test]
fn choke_leg_analysis() -> Result<()> {
    let data = std::fs::read("backup/CHARS/ANAKIN/ANAKIN_PADAWAN_PC.GHG")?;
    let p = ghg::parse(&data)?;
    let rest_locals: Vec<Mat4> = p.bones.iter().map(|b| b.local).collect();
    let rest_worlds: Vec<Mat4> = p.bones.iter().map(|b| b.world).collect();
    let meshes = glb::build_meshes(&p);

    // Which parts/bones cover the legs? Show the leg bone names + parents + footers.
    println!("\n=== leg bones: name parent footer uses_x20 rest_local axis/angle ===");
    for i in [0usize, 1, 2, 3, 23, 24, 25, 26, 27, 28, 29, 30] {
        if let Some(b) = p.bones.get(i) {
            let m = Mat4::from_mat3(glam::Mat3::from_mat4(rest_locals[i]));
            let q = glam::Quat::from_mat4(&m);
            let (ax, ang) = q.to_axis_angle();
            println!(
                "bone {i:>2} {:<14} parent {:<3} local axis {:+.3},{:+.3},{:+.3} ang {:>6.1}deg",
                b.name,
                if b.parent < 0 { -1 } else { b.parent },
                ax.x, ax.y, ax.z,
                ang.to_degrees()
            );
        }
    }

    let idle = An3::parse(&std::fs::read("backup/CHARS/ANAKIN/IDLE.AN3")?)?;
    let choke = An3::parse(&std::fs::read("backup/CHARS/ANAKIN/FORCECHOKE.AN3")?)?;
    let parents: Vec<i32> = (0..idle.num_bones)
        .map(|i| p.bones.get(i).map(|b| b.parent.min(idle.num_bones as i32 - 1)).unwrap_or(-1))
        .collect();

    // Full footer array.
    println!("\n=== footers ===");
    for b in 0..idle.num_bones {
        println!(
            "  bone {b:>2} {:<14} footer 0x{:02x} uses_x20 {}",
            p.bones.get(b).map(|x| x.name.as_str()).unwrap_or("?"),
            idle.footer.get(b).copied().unwrap_or(0),
            idle.uses_x20(b)
        );
    }

    // Neutral rotations for the leg bones, IDLE vs CHOKE.
    println!("\n=== leg neutral rotations (IDLE vs CHOKE) ===");
    for i in [23usize, 24, 25, 27, 28, 29] {
        let iv: Vec<f32> = (3..6).map(|c| idle.neutral(i, c)).collect();
        let cv: Vec<f32> = (3..6).map(|c| choke.neutral(i, c)).collect();
        println!(
            "  bone {i:>2} {:<14} IDLE rot [{:+.4},{:+.4},{:+.4}]  CHOKE rot [{:+.4},{:+.4},{:+.4}]",
            p.bones.get(i).map(|b| b.name.as_str()).unwrap_or("?"),
            iv[0], iv[1], iv[2], cv[0], cv[1], cv[2]
        );
    }

    // World joint positions for the legs: rest, IDLE f0, CHOKE f10.
    let mut legs = vec![23usize, 24, 25, 27, 28, 29];
    for &b in [0usize, 1, 2, 3].iter() {
        legs.push(b);
    }
    for (label, worlds) in [
        ("rest", rest_worlds.clone()),
        ("IDLE f0", idle.bone_worlds(&parents, &rest_locals, 0.0)?),
        ("CHOKE f10", choke.bone_worlds(&parents, &rest_locals, 10.0)?),
    ] {
        println!("\n=== {label}: leg/hip joint world positions + thigh/knee directions ===");
        for i in [0usize, 1, 2, 3, 23, 24, 25, 26, 27, 28, 29] {
            let j = joint(&worlds[i]);
            let name = p.bones.get(i).map(|b| b.name.as_str()).unwrap_or("?");
            println!(
                "  bone {i:>2} {:<14} joint {:+.3},{:+.3},{:+.3}",
                name, j.x, j.y, j.z
            );
        }
        // Knee direction: knee joint relative to its hip (world - parent world).
        for (hip, knee) in [(23usize, 24usize), (27, 28)] {
            let hj = joint(&worlds[hip]);
            let kj = joint(&worlds[knee]);
            let d = kj - hj;
            println!(
                "  {:<8}->{:>7}: knee rel hip {:+.3},{:+.3},{:+.3}  (len {:.3})",
                p.bones.get(hip).map(|b| b.name.as_str()).unwrap_or("?"),
                p.bones.get(knee).map(|b| b.name.as_str()).unwrap_or("?"),
                d.x, d.y, d.z, d.length()
            );
        }
    }
    Ok(())
}

#[test]
fn choke_leg_orientation_check() -> Result<()> {
    let data = std::fs::read("backup/CHARS/ANAKIN/ANAKIN_PADAWAN_PC.GHG")?;
    let p = ghg::parse(&data)?;
    let rest_locals: Vec<Mat4> = p.bones.iter().map(|b| b.local).collect();
    let rest_worlds: Vec<Mat4> = p.bones.iter().map(|b| b.world).collect();
    let parents: Vec<i32> = p.bones.iter().map(|b| b.parent).collect();

    // Reference: rest pose foot orientation.
    let toe_rel = |w: &[Mat4], hip: usize, knee: usize, ankle: usize| -> (Vec3, Vec3, Vec3) {
        (
            joint(&w[knee]) - joint(&w[hip]),
            joint(&w[ankle]) - joint(&w[knee]),
            joint(&w[ankle + 1]) - joint(&w[ankle]),
        )
    };
    let (rt, rs, rf) = toe_rel(&rest_worlds, 23, 24, 25);
    let (lt, ls, lf) = toe_rel(&rest_worlds, 27, 28, 29);
    println!("=== REST foot reference ===");
    println!("  right: thigh {:+.3},{:+.3},{:+.3} shin {:+.3},{:+.3},{:+.3} foot {:+.3},{:+.3},{:+.3}", rt.x, rt.y, rt.z, rs.x, rs.y, rs.z, rf.x, rf.y, rf.z);
    println!("  left : thigh {:+.3},{:+.3},{:+.3} shin {:+.3},{:+.3},{:+.3} foot {:+.3},{:+.3},{:+.3}", lt.x, lt.y, lt.z, ls.x, ls.y, ls.z, lf.x, lf.y, lf.z);

    // Animate WALK and RUN: foot should keep pointing the rest direction, thighs
    // should swing forward/back along the foot axis, knees should not flip.
    for name in ["WALK", "RUN", "FORCECHOKE"] {
        let an3 = An3::parse(&std::fs::read(format!("backup/CHARS/ANAKIN/{name}.AN3"))?)?;
        println!("\n=== {name}: root translation (bone 0) per frame + leg swing/flip flags ===");
        // Sample frames evenly.
        let n = an3.num_frames.min(8);
        for fi in 0..n {
            let f = if an3.num_frames > 1 {
                fi as f32 * (an3.num_frames - 1) as f32 / (n - 1) as f32
            } else {
                0.0
            };
            let w = an3.bone_worlds(&parents, &rest_locals, f)?;
            let root = joint(&w[0]);
            let (rt, rs, rf) = toe_rel(&w, 23, 24, 25);
            let (lt, ls, lf) = toe_rel(&w, 27, 28, 29);
            // foot flip if the foot's Z sign differs from rest (rest foot Z is -).
            let r_fwd = rf.z.signum() == (-1.0f32).signum();
            let l_fwd = lf.z.signum() == (-1.0f32).signum();
            // knee forward-ness: knee rel hip should be mostly -Y (down) with modest Z swing
            println!(
                "  f{f:>5.1} root {:+.3},{:+.3},{:+.3}",
                root.x, root.y, root.z
            );
            println!(
                "    R thigh {:+.3},{:+.3},{:+.3} shin {:+.3},{:+.3},{:+.3} foot {:+.3},{:+.3},{:+.3} foot_fwd={}",
                rt.x, rt.y, rt.z, rs.x, rs.y, rs.z, rf.x, rf.y, rf.z, r_fwd
            );
            println!(
                "    L thigh {:+.3},{:+.3},{:+.3} shin {:+.3},{:+.3},{:+.3} foot {:+.3},{:+.3},{:+.3} foot_fwd={}",
                lt.x, lt.y, lt.z, ls.x, ls.y, ls.z, lf.x, lf.y, lf.z, l_fwd
            );
        }
    }
    Ok(())
}

#[test]
fn choke_mesh_foot_geometry() -> Result<()> {
    let data = std::fs::read("backup/CHARS/ANAKIN/ANAKIN_PADAWAN_PC.GHG")?;
    let p = ghg::parse(&data)?;
    let meshes = glb::build_meshes(&p);
    println!("=== parts skinned to leg bones (23-30) ===");
    for (pi, md) in meshes.iter().enumerate() {
        let leg_bones: Vec<u16> = md
            .skin_bones
            .iter()
            .copied()
            .filter(|&b| (23..=30).contains(&b))
            .collect();
        if leg_bones.is_empty() {
            continue;
        }
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        for v in &md.pos {
            for k in 0..3 {
                min[k] = min[k].min(v[k]);
                max[k] = max[k].max(v[k]);
            }
        }
        println!(
            "  part {} (render {pi}): {:<3} verts, bones {:?}, bbox x[{:.3},{:.3}] y[{:.3},{:.3}] z[{:.3},{:.3}]",
            p.render[pi].part,
            md.pos.len(),
            leg_bones,
            min[0], max[0], min[1], max[1], min[2], max[2]
        );
    }
    println!("\n=== parts skinned to head bones (21,22) ===");
    for (pi, md) in meshes.iter().enumerate() {
        let hb: Vec<u16> = md.skin_bones.iter().copied().filter(|&b| b == 21 || b == 22).collect();
        if hb.is_empty() {
            continue;
        }
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        for v in &md.pos {
            for k in 0..3 {
                min[k] = min[k].min(v[k]);
                max[k] = max[k].max(v[k]);
            }
        }
        println!(
            "  part {} (render {pi}): {:<3} verts, bones {:?}, bbox x[{:.3},{:.3}] y[{:.3},{:.3}] z[{:.3},{:.3}]",
            p.render[pi].part,
            md.pos.len(),
            hb,
            min[0], max[0], min[1], max[1], min[2], max[2]
        );
    }
    // Where is the center of mass / front of the head relative to the root?
    println!("\n=== bone positions (root, head, helmet) ===");
    for b in [0usize, 21, 22] {
        let j = joint(&p.bones[b].world);
        println!("  bone {b} {:<8} joint {:+.3},{:+.3},{:+.3}", p.bones[b].name, j.x, j.y, j.z);
    }
    Ok(())
}

#[test]
fn choke_leg_euler_variants() -> Result<()> {
    let data = std::fs::read("backup/CHARS/ANAKIN/ANAKIN_PADAWAN_PC.GHG")?;
    let p = ghg::parse(&data)?;
    let rest_locals: Vec<Mat4> = p.bones.iter().map(|b| b.local).collect();
    let rest_worlds: Vec<Mat4> = p.bones.iter().map(|b| b.world).collect();
    let meshes = glb::build_meshes(&p);
    let parents: Vec<i32> = p.bones.iter().map(|b| b.parent).collect();

    // For each euler convention applied ONLY to the non-x20 leg bones (23-30),
    // report IDLE frame-0 pose fidelity and CHOKE frame-10 foot/anatomy.
    let conventions: [&str; 9] = [
        "RzRyRx (current)",
        "RxRyRz",
        "RyRxRz",
        "RzRxRy",
        "RxRzRy",
        "RyRzRx",
        "RzRyRx negZ",
        "RzRyRx negXYZ",
        "RzRyRx negXY",
    ];
    for conv in conventions {
        println!("\n=== convention {conv} ===");
        for name in ["IDLE", "FORCECHOKE"] {
            let an3 = An3::parse(&std::fs::read(format!("backup/CHARS/ANAKIN/{name}.AN3"))?)?;
            let frame = if name == "IDLE" { 0.0 } else { 10.0 };
            let worlds = worlds_conv(&an3, &parents, &rest_locals, conv, frame);
            let stretch = max_stretch(&meshes, &worlds, &rest_worlds);
            let w = stretch.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut maxd = 0.0f32;
            for md in &meshes {
                if md.skin.is_empty() {
                    continue;
                }
                let mats: Vec<Mat4> = md
                    .skin_bones
                    .iter()
                    .map(|&b| worlds[b as usize] * rest_worlds[b as usize].inverse())
                    .collect();
                for v in 0..md.pos.len() {
                    let sw = &md.skin[v * 8..v * 8 + 8];
                    let posed = Vec3::from(blend_pos(&md.pos[v], sw, &md.skin_bones, &mats));
                    maxd = maxd.max((posed - Vec3::from(md.pos[v])).length());
                }
            }
            // leg anatomy at this frame
            let toe = |w: &[Mat4], h: usize, k: usize, a: usize| {
                (
                    joint(&w[k]) - joint(&w[h]),
                    joint(&w[a]) - joint(&w[k]),
                    joint(&w[a + 1]) - joint(&w[a]),
                )
            };
            let (rt, rs, rf) = toe(&worlds, 23, 24, 25);
            let (lt, ls, lf) = toe(&worlds, 27, 28, 29);
            let ra = joint(&worlds[25]);
            let la = joint(&worlds[29]);
            println!(
                "  {name:<10} f{frame:>4.0} stretch {w:.2}x delta {maxd:.3} | R ankle {:+.3},{:+.3},{:+.3} L ankle {:+.3},{:+.3},{:+.3}",
                ra.x, ra.y, ra.z, la.x, la.y, la.z
            );
            println!(
                "        R foot rel ankle {:+.3},{:+.3},{:+.3}  L foot rel ankle {:+.3},{:+.3},{:+.3}",
                rf.x, rf.y, rf.z, lf.x, lf.y, lf.z
            );
        }
    }
    Ok(())
}

fn worlds_conv(
    an3: &An3,
    parents: &[i32],
    rest_locals: &[Mat4],
    conv: &str,
    frame: f32,
) -> Vec<Mat4> {
    let mut worlds = Vec::with_capacity(an3.num_bones);
    for b in 0..an3.num_bones {
        let t = glam::Vec3::new(
            an3.channel_value(b, 0, frame),
            an3.channel_value(b, 1, frame),
            -an3.channel_value(b, 2, frame),
        );
        let (x, y, z) = (
            an3.channel_value(b, 3, frame),
            an3.channel_value(b, 4, frame),
            an3.channel_value(b, 5, frame),
        );
        let (x, y, z) = match conv {
            "RzRyRx negZ" => (x, y, -z),
            "RzRyRx negXYZ" => (-x, -y, -z),
            "RzRyRx negXY" => (-x, -y, z),
            "RzRyRx negXZ" => (x, -y, -z),
            _ => (x, y, z),
        };
        let ra = match conv {
            "RxRyRz" => Mat4::from_rotation_x(x) * Mat4::from_rotation_y(y) * Mat4::from_rotation_z(z),
            "RyRxRz" => Mat4::from_rotation_y(y) * Mat4::from_rotation_x(x) * Mat4::from_rotation_z(z),
            "RzRxRy" => Mat4::from_rotation_z(z) * Mat4::from_rotation_x(x) * Mat4::from_rotation_y(y),
            "RxRzRy" => Mat4::from_rotation_x(x) * Mat4::from_rotation_z(z) * Mat4::from_rotation_y(y),
            "RyRzRx" => Mat4::from_rotation_y(y) * Mat4::from_rotation_z(z) * Mat4::from_rotation_x(x),
            _ => Mat4::from_rotation_z(z) * Mat4::from_rotation_y(y) * Mat4::from_rotation_x(x),
        };
        let r = if an3.uses_x20(b) {
            let rl = Mat4::from_mat3(glam::Mat3::from_mat4(rest_locals[b]));
            if an3.footer.get(b).map_or(false, |f| f & 0x01 != 0) {
                rl * ra
            } else {
                rl
            }
        } else {
            ra
        };
        let s = if an3.scale_flag(b) {
            glam::Vec3::new(
                an3.channel_value(b, 6, frame),
                an3.channel_value(b, 7, frame),
                an3.channel_value(b, 8, frame),
            )
        } else {
            glam::Vec3::ONE
        };
        let local = Mat4::from_translation(t) * r * Mat4::from_scale(s);
        let w = if parents[b] < 0 {
            local
        } else {
            worlds[parents[b] as usize] * local
        };
        worlds.push(w);
    }
    worlds
}

#[test]
fn choke_leg_mirror_walk() -> Result<()> {
    let data = std::fs::read("backup/CHARS/ANAKIN/ANAKIN_PADAWAN_PC.GHG")?;
    let p = ghg::parse(&data)?;
    let rest_locals: Vec<Mat4> = p.bones.iter().map(|b| b.local).collect();
    let rest_worlds: Vec<Mat4> = p.bones.iter().map(|b| b.world).collect();
    let parents: Vec<i32> = p.bones.iter().map(|b| b.parent).collect();
    let rf = joint(&rest_worlds[26]) - joint(&rest_worlds[25]);
    let lf = joint(&rest_worlds[30]) - joint(&rest_worlds[29]);
    println!(
        "\nMODEL REST foot: R {:+.3},{:+.3},{:+.3}  L {:+.3},{:+.3},{:+.3}",
        rf.x, rf.y, rf.z, lf.x, lf.y, lf.z
    );

    for conv in ["RzRyRx (current)", "RzRyRx negXYZ"] {
        println!("\n############ convention {conv} ############");
        for name in ["IDLE", "RIDE", "TIEDUP", "FALL", "SLIDE", "BACKFLIP"] {
            let an3 = An3::parse(&std::fs::read(format!("backup/CHARS/ANAKIN/{name}.AN3"))?)?;
            println!("  === {name} ({}) ===", an3.num_frames);
            let n = an3.num_frames.min(5);
            for fi in 0..n {
                let f = if an3.num_frames > 1 {
                    fi as f32 * (an3.num_frames - 1) as f32 / (n - 1) as f32
                } else {
                    0.0
                };
                let w = worlds_conv(&an3, &parents, &rest_locals, conv, f);
                let rt = joint(&w[24]) - joint(&w[23]);
                let lt = joint(&w[28]) - joint(&w[27]);
                let rf = joint(&w[26]) - joint(&w[25]);
                let lf = joint(&w[30]) - joint(&w[29]);
                let rs = joint(&w[25]) - joint(&w[24]);
                let ls = joint(&w[29]) - joint(&w[28]);
                println!(
                    "    f{f:>5.1} R thigh {:+.3},{:+.3},{:+.3} R shank {:+.3},{:+.3},{:+.3} R foot {:+.3},{:+.3},{:+.3}",
                    rt.x, rt.y, rt.z, rs.x, rs.y, rs.z, rf.x, rf.y, rf.z
                );
                println!(
                    "           L thigh {:+.3},{:+.3},{:+.3} L shank {:+.3},{:+.3},{:+.3} L foot {:+.3},{:+.3},{:+.3}",
                    lt.x, lt.y, lt.z, ls.x, ls.y, ls.z, lf.x, lf.y, lf.z
                );
            }
        }
    }
    Ok(())
}

#[test]
fn arm_trajectory_compare() -> Result<()> {
    let data = std::fs::read("backup/CHARS/ANAKIN/ANAKIN_PADAWAN_PC.GHG")?;
    let p = ghg::parse(&data)?;
    let rest_locals: Vec<Mat4> = p.bones.iter().map(|b| b.local).collect();
    let parents: Vec<i32> = p.bones.iter().map(|b| b.parent).collect();

    for name in ["THROW", "SWIPEBEHIND", "SWIPELEFT", "WALK", "RUN"] {
        let an3 = An3::parse(&std::fs::read(format!("backup/CHARS/ANAKIN/{name}.AN3"))?)?;
        println!("\n=== {name} ({}) ===", an3.num_frames);
        for conv in ["RzRyRx (current)", "RzRyRx negXYZ"] {
            let mut rows = Vec::new();
            for fi in 0..an3.num_frames {
                let w = worlds_conv(&an3, &parents, &rest_locals, conv, fi as f32);
                let lh = joint(&w[8]).y;
                let rh = joint(&w[14]).y;
                let ltoe = joint(&w[26]).y;
                let rtoe = joint(&w[30]).y;
                rows.push((fi, lh, rh, ltoe, rtoe));
            }
            println!("  [{conv}]");
            let mut line = String::new();
            for (fi, lh, rh, lt, rt) in rows {
                line.push_str(&format!(
                    " f{fi}: Lh{:.2} Rh{:.2} Lt{:.2} Rt{:.2}",
                    lh, rh, lt, rt
                ));
            }
            println!("{line}");
        }
    }
    Ok(())
}

#[test]
fn convention_verdict() -> Result<()> {
    let data = std::fs::read("backup/CHARS/ANAKIN/ANAKIN_PADAWAN_PC.GHG")?;
    let p = ghg::parse(&data)?;
    let rest_locals: Vec<Mat4> = p.bones.iter().map(|b| b.local).collect();
    let rest_worlds: Vec<Mat4> = p.bones.iter().map(|b| b.world).collect();
    let parents: Vec<i32> = p.bones.iter().map(|b| b.parent).collect();
    let meshes = glb::build_meshes(&p);

    let conventions = ["RzRyRx (current)", "RzRyRx negXYZ", "RzRyRx negXY", "RzRyRx negXZ"];
    for conv in conventions {
        println!("\n========== {conv} ==========");
        // 1) IDLE f0 mesh fit
        {
            let an3 = An3::parse(&std::fs::read("backup/CHARS/ANAKIN/IDLE.AN3")?)?;
            let worlds = worlds_conv(&an3, &parents, &rest_locals, conv, 0.0);
            let mut maxd = 0.0f32;
            for md in &meshes {
                if md.skin.is_empty() {
                    continue;
                }
                let mats: Vec<Mat4> = md
                    .skin_bones
                    .iter()
                    .map(|&b| worlds[b as usize] * rest_worlds[b as usize].inverse())
                    .collect();
                for v in 0..md.pos.len() {
                    let sw = &md.skin[v * 8..v * 8 + 8];
                    let posed = Vec3::from(blend_pos(&md.pos[v], sw, &md.skin_bones, &mats));
                    maxd = maxd.max((posed - Vec3::from(md.pos[v])).length());
                }
            }
            println!("IDLE f0 mesh max-delta: {maxd:.4}");
        }
        // 2) TIPTOE: toe should be BELOW ankle (standing on toes)
        {
            let an3 = An3::parse(&std::fs::read("backup/CHARS/ANAKIN/TIPTOE.AN3")?)?;
            let w = worlds_conv(&an3, &parents, &rest_locals, conv, 0.0);
            let ra = joint(&w[25]).y;
            let rt = joint(&w[26]).y;
            let la = joint(&w[29]).y;
            let lt = joint(&w[30]).y;
            println!(
                "TIPTOE: R ankle {ra:.3} R toe {rt:.3}  L ankle {la:.3} L toe {lt:.3}  (toe<ankle = tiptoe ok)"
            );
        }
        // 3) THROW f0: right hand height + forward reach
        {
            let an3 = An3::parse(&std::fs::read("backup/CHARS/ANAKIN/THROW.AN3")?)?;
            let w = worlds_conv(&an3, &parents, &rest_locals, conv, 0.0);
            let sh = joint(&w[11]).y;
            let rh = joint(&w[14]).y;
            let rp = joint(&w[14]).z;
            let wep = joint(&w[15]).y;
            println!("THROW: R shoulder {sh:.3} R hand {rh:.3} (z {rp:.3}) weaponR {wep:.3}  (hand high+front = throw)");
        }
        // 4) CHOKE f10: hand reach + feet
        {
            let an3 = An3::parse(&std::fs::read("backup/CHARS/ANAKIN/FORCECHOKE.AN3")?)?;
            let w = worlds_conv(&an3, &parents, &rest_locals, conv, 10.0);
            let lh = joint(&w[8]).y;
            let rh = joint(&w[14]).y;
            let neck = joint(&w[21]).y;
            let lt = joint(&w[30]).y;
            let rt = joint(&w[26]).y;
            println!(
                "CHOKE: neck {neck:.3} Lhand {lh:.3} Rhand {rh:.3} Ltoe {lt:.3} Rtoe {rt:.3}"
            );
        }
        // 5) RIDE f0: thigh direction
        {
            let an3 = An3::parse(&std::fs::read("backup/CHARS/ANAKIN/RIDE.AN3")?)?;
            let w = worlds_conv(&an3, &parents, &rest_locals, conv, 0.0);
            let rk = joint(&w[24]) - joint(&w[23]);
            let lk = joint(&w[28]) - joint(&w[27]);
            println!(
                "RIDE: R thigh ({:+.3},{:+.3},{:+.3}) L thigh ({:+.3},{:+.3},{:+.3})  (z<0 = forward)",
                rk.x, rk.y, rk.z, lk.x, lk.y, lk.z
            );
        }
        // 6) WALK: feet lowest point over cycle (should reach ~0 near ground)
        {
            let an3 = An3::parse(&std::fs::read("backup/CHARS/ANAKIN/WALK.AN3")?)?;
            let mut minrt = f32::INFINITY;
            let mut minlt = f32::INFINITY;
            let mut maxr = f32::NEG_INFINITY;
            let mut maxl = f32::NEG_INFINITY;
            for fi in 0..an3.num_frames {
                let w = worlds_conv(&an3, &parents, &rest_locals, conv, fi as f32);
                let rt = joint(&w[26]).y;
                let lt = joint(&w[30]).y;
                minrt = minrt.min(rt);
                minlt = minlt.min(lt);
                maxr = maxr.max(rt);
                maxl = maxl.max(lt);
            }
            println!("WALK toe y range: R {minrt:.3}..{maxr:.3}  L {minlt:.3}..{maxl:.3}");
        }
    }
    Ok(())
}

#[test]
fn pose_dump_compare() -> Result<()> {
    let data = std::fs::read("backup/CHARS/ANAKIN/ANAKIN_PADAWAN_PC.GHG")?;
    let p = ghg::parse(&data)?;
    let rest_locals: Vec<Mat4> = p.bones.iter().map(|b| b.local).collect();
    let parents: Vec<i32> = p.bones.iter().map(|b| b.parent).collect();

    for (name, frame) in [
        ("PUSH", 0.0),
        ("SLAM", 0.0),
        ("SWIPEBEHIND", 3.0),
    ] {
        let an3 = An3::parse(&std::fs::read(format!("backup/CHARS/ANAKIN/{name}.AN3"))?)?;
        for conv in ["RzRyRx (current)", "RzRyRx negXYZ", "RzRyRx negXY"] {
            let w = worlds_conv(&an3, &parents, &rest_locals, conv, frame);
            println!("\n== {name} f{frame} [{conv}] ==");
            let names = [
                "char", "upperTorso", "body", "chest",
                "leftArm", "leftShoulder", "leftElbow", "leftElbowLen", "leftHand", "weaponL",
                "rightArm", "rightShoulder", "rightElbow", "rightElbowLen", "rightHand", "weaponR",
                "cloak0", "cloak1", "cloak2", "cloak3", "cloak4",
                "head", "helmet",
                "rightLeg", "rightKnee", "rightAnkle", "rightToe",
                "leftLeg", "leftKnee", "leftAnkle", "leftToe",
            ];
            for i in 0..an3.num_bones {
                if i >= names.len() {
                    break;
                }
                let j = joint(&w[i]);
                println!(
                    "{i:2} {:<8} {:+.3} {:+.3} {:+.3}",
                    names[i], j.x, j.y, j.z
                );
            }
        }
    }
    Ok(())
}

#[test]
fn choke_skeleton_hierarchy() -> Result<()> {
    let data = std::fs::read("backup/CHARS/ANAKIN/ANAKIN_PADAWAN_PC.GHG")?;
    let p = ghg::parse(&data)?;
    println!("=== skeleton hierarchy (idx name parent) ===");
    for (i, b) in p.bones.iter().enumerate() {
        println!(
            "{i:>2} {:<16} parent {}",
            b.name,
            if b.parent < 0 { -1 } else { b.parent }
        );
    }
    Ok(())
}

#[test]
fn choke_skin_matrix_rigidity() -> Result<()> {
    let data = std::fs::read("backup/CHARS/ANAKIN/ANAKIN_PADAWAN_PC.GHG")?;
    let p = ghg::parse(&data)?;
    let rest_locals: Vec<Mat4> = p.bones.iter().map(|b| b.local).collect();
    let rest_worlds: Vec<Mat4> = p.bones.iter().map(|b| b.world).collect();
    let meshes = glb::build_meshes(&p);
    let an3 = An3::parse(&std::fs::read("backup/CHARS/ANAKIN/FORCECHOKE.AN3")?)?;
    let parents: Vec<i32> = (0..an3.num_bones)
        .map(|i| p.bones.get(i).map(|b| b.parent.min(an3.num_bones as i32 - 1)).unwrap_or(-1))
        .collect();
    let worlds = an3.bone_worlds(&parents, &rest_locals, 10.0)?;

    // Which bones does part 34 (the 62-vert left-arm part) actually use?
    for (pi, md) in meshes.iter().enumerate() {
        if pi != 34 {
            continue;
        }
        println!("\n=== part 34 skin bones (render idx {pi}) ===");
        for &b in &md.skin_bones {
            let m = worlds[b as usize] * rest_worlds[b as usize].inverse();
            let m3 = m.to_cols_array();
            let colx = glam::Vec3::new(m3[0], m3[1], m3[2]);
            let coly = glam::Vec3::new(m3[4], m3[5], m3[6]);
            let colz = glam::Vec3::new(m3[8], m3[9], m3[10]);
            let s = [
                colx.length(),
                coly.length(),
                colz.length(),
                colx.dot(coly.cross(colz)),
            ];
            let name = p.bones.get(b as usize).map(|b| b.name.as_str()).unwrap_or("?");
            println!(
                "  bone {b:>2} {name:<15} skin mat col lengths {:.4},{:.4},{:.4} (det~1 rigid)",
                s[0], s[1], s[2]
            );
        }
        // Show the weights per vertex for the first few verts.
        println!("  first 8 verts (weights, local bone ids):");
        for v in 0..8.min(md.pos.len()) {
            let sw = &md.skin[v * 8..v * 8 + 8];
            let ws: Vec<String> = (0..4)
                .filter(|&k| (sw[4 + k] as usize) < md.skin_bones.len())
                .map(|k| format!("{}:{}", md.skin_bones[sw[4 + k] as usize], sw[k]))
                .collect();
            println!("    v{v}: {} pos {:.3},{:.3},{:.3}", ws.join(" "), md.pos[v][0], md.pos[v][1], md.pos[v][2]);
        }
        // Compute the worst single edge and where it is.
        let mats: Vec<Mat4> = md
            .skin_bones
            .iter()
            .map(|&b| worlds[b as usize] * rest_worlds[b as usize].inverse())
            .collect();
        let mut posed: Vec<[f32; 3]> = Vec::with_capacity(md.pos.len());
        for v in 0..md.pos.len() {
            let sw = &md.skin[v * 8..v * 8 + 8];
            posed.push(blend_pos(&md.pos[v], sw, &md.skin_bones, &mats));
        }
        let mut worst: Option<(usize, f32, f32)> = None;
        for tri in md.idx.chunks(3) {
            for k in 0..3 {
                let a = tri[k] as usize;
                let b = tri[(k + 1) % 3] as usize;
                let len = (Vec3::from(posed[a]) - Vec3::from(posed[b])).length();
                let rlen = (Vec3::from(md.pos[a]) - Vec3::from(md.pos[b])).length();
                if rlen > 0.01 {
                    let ratio = len / rlen;
                    if worst.map_or(true, |(_, _, wr)| ratio > wr) {
                        worst = Some((a, ratio, rlen));
                    }
                }
            }
        }
        if let Some((vi, ratio, rlen)) = worst {
            println!(
                "  worst edge: vertex {vi} ratio {ratio:.2}x (rest len {rlen:.3}), pos {:.3},{:.3},{:.3}, posed {:.3},{:.3},{:.3}",
                md.pos[vi][0], md.pos[vi][1], md.pos[vi][2],
                posed[vi][0], posed[vi][1], posed[vi][2]
            );
        }
    }
    Ok(())
}

#[test]
fn choke_world_hand_positions() -> Result<()> {
    let data = std::fs::read("backup/CHARS/ANAKIN/ANAKIN_PADAWAN_PC.GHG")?;
    let p = ghg::parse(&data)?;
    let rest_locals: Vec<Mat4> = p.bones.iter().map(|b| b.local).collect();
    let rest_worlds: Vec<Mat4> = p.bones.iter().map(|b| b.world).collect();
    let an3 = An3::parse(&std::fs::read("backup/CHARS/ANAKIN/FORCECHOKE.AN3")?)?;
    let parents: Vec<i32> = (0..an3.num_bones)
        .map(|i| p.bones.get(i).map(|b| b.parent.min(an3.num_bones as i32 - 1)).unwrap_or(-1))
        .collect();

    for name in ["IDLE", "FORCECHOKE"] {
        let a = An3::parse(&std::fs::read(format!("backup/CHARS/ANAKIN/{name}.AN3"))?)?;
        println!("\n=== {name}: world joint positions, rest vs composed vs absolute ===");
        for mode in ["rest", "composed", "absolute"] {
            let worlds: Vec<Mat4> = if mode == "rest" {
                rest_worlds.clone()
            } else {
                worlds_with_mode(&a, &parents, &rest_locals, mode, 10.0)
            };
            for b in [0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9] {
                let j = joint(&worlds[b]);
                println!(
                    "  {mode:<9} bone {b:>2} {:<15} joint {:+.3},{:+.3},{:+.3}",
                    p.bones.get(b).map(|b| b.name.as_str()).unwrap_or("?"),
                    j.x,
                    j.y,
                    j.z
                );
            }
        }
    }
    Ok(())
}

#[test]
fn choke_all_animated_channels() -> Result<()> {
    let an3 = An3::parse(&std::fs::read("backup/CHARS/ANAKIN/FORCECHOKE.AN3")?)?;
    println!("\n=== FORCECHOKE: all animated (0x06/0x07) channels ===");
    for b in 0..an3.num_bones {
        for c in 0..9usize {
            let m = an3.matrix[b * 9 + c];
            if m == 0x06 || m == 0x07 {
                let idx = an3.animated.iter().position(|&a| a == b * 9 + c).unwrap();
                let (scale, offset) = an3.movpar[idx];
                println!("  bone {b:>2} chan {c}: anim{m:02x} scale={scale:.4} offset={offset:.4}");
            }
        }
    }
    // Static (non-animated) bones with non-trivial rotation offsets vs IDLE.
    println!("\n=== FORCECHOKE vs IDLE: static rotation channels (bone 3,4,5) ===");
    for name in ["IDLE", "FORCECHOKE"] {
        let a = An3::parse(&std::fs::read(format!("backup/CHARS/ANAKIN/{name}.AN3"))?)?;
        println!("  {name}:");
        for b in 0..a.num_bones {
            let mut vals = [0f32; 3];
            let mut any = false;
            for (k, c) in (3..6).enumerate() {
                let m = a.matrix[b * 9 + c];
                vals[k] = match m {
                    0x06 | 0x07 => a
                        .animated
                        .iter()
                        .position(|&x| x == b * 9 + c)
                        .map_or(0.0, |i| a.movpar[i].1),
                    v if v >= 0x10 => a.statics[(v - 0x10) as usize],
                    _ => 0.0,
                };
                if m != 0 {
                    any = true;
                }
            }
            if any && vals.iter().map(|v| v.abs()).sum::<f32>() > 0.02 {
                println!(
                    "    bone {b:>2} rot [{:+.4},{:+.4},{:+.4}]",
                    vals[0], vals[1], vals[2]
                );
            }
        }
    }
    Ok(())
}

#[test]
fn choke_channel_storage() -> Result<()> {
    for name in ["IDLE", "FORCECHOKE"] {
        let an3 = An3::parse(&std::fs::read(format!("backup/CHARS/ANAKIN/{name}.AN3"))?)?;
        println!("\n=== {name}: raw channel storage for left-arm bones ===");
        for b in [4usize, 5, 6, 7, 8, 9] {
            println!("  bone {b}: matrix row T(x,y,z) R(x,y,z) S(x,y,z):");
            for c in 0..9usize {
                let m = an3.matrix[b * 9 + c];
                let desc = match m {
                    0x06 => {
                        // find animated ordinal
                        let idx = an3
                            .animated
                            .iter()
                            .position(|&a| a == b * 9 + c)
                            .unwrap();
                        let (scale, offset) = an3.movpar[idx];
                        format!("anim06 idx{idx} scale={scale:.4} offset={offset:.4}")
                    }
                    0x07 => {
                        let idx = an3
                            .animated
                            .iter()
                            .position(|&a| a == b * 9 + c)
                            .unwrap();
                        let (scale, offset) = an3.movpar[idx];
                        format!("anim07 idx{idx} scale={scale:.4} offset={offset:.4}")
                    }
                    v if v >= 0x10 => {
                        let si = (v - 0x10) as usize;
                        format!("static table[{si}] = {}", an3.statics[si])
                    }
                    _ => format!("default ({})", if c >= 6 { "1.0" } else { "0.0" }),
                };
                println!("    chan {c}: 0x{m:04x}  {desc}");
            }
            println!(
                "    footer = 0x{:02x}  uses_x20 = {}",
                an3.footer.get(b).copied().unwrap_or(0),
                an3.uses_x20(b)
            );
        }
    }
    Ok(())
}

fn worlds_with_mode(
    an3: &An3,
    parents: &[i32],
    rest_locals: &[Mat4],
    mode: &str,
    frame: f32,
) -> Vec<Mat4> {
    let mut worlds = Vec::with_capacity(an3.num_bones);
    for b in 0..an3.num_bones {
        let t = glam::Vec3::new(
            an3.channel_value(b, 0, frame),
            an3.channel_value(b, 1, frame),
            -an3.channel_value(b, 2, frame),
        );
        let ra = Mat4::from_rotation_z(an3.channel_value(b, 5, frame))
            * Mat4::from_rotation_y(an3.channel_value(b, 4, frame))
            * Mat4::from_rotation_x(an3.channel_value(b, 3, frame));
        let r = match mode {
            "absolute" => ra,
            "r_anim_then_rest" => {
                if an3.uses_x20(b) {
                    let rl = Mat4::from_mat3(glam::Mat3::from_mat4(rest_locals[b]));
                    ra * rl
                } else {
                    ra
                }
            }
            _ => {
                if an3.uses_x20(b) {
                    let rl = Mat4::from_mat3(glam::Mat3::from_mat4(rest_locals[b]));
                    rl * ra
                } else {
                    ra
                }
            }
        };
        let s = if an3.scale_flag(b) {
            glam::Vec3::new(
                an3.channel_value(b, 6, frame),
                an3.channel_value(b, 7, frame),
                an3.channel_value(b, 8, frame),
            )
        } else {
            glam::Vec3::ONE
        };
        let local = Mat4::from_translation(t) * r * Mat4::from_scale(s);
        let w = if parents[b] < 0 {
            local
        } else {
            worlds[parents[b] as usize] * local
        };
        worlds.push(w);
    }
    worlds
}

fn worlds_ref_mode(
    an3: &An3,
    parents: &[i32],
    rest_locals: &[Mat4],
    identities: &[Mat4],
    mode: &str,
    frame: f32,
) -> Vec<Mat4> {
    let mut worlds = Vec::with_capacity(an3.num_bones);
    for b in 0..an3.num_bones {
        let t = glam::Vec3::new(
            an3.channel_value(b, 0, frame),
            an3.channel_value(b, 1, frame),
            -an3.channel_value(b, 2, frame),
        );
        let ra = Mat4::from_rotation_z(an3.channel_value(b, 5, frame))
            * Mat4::from_rotation_y(an3.channel_value(b, 4, frame))
            * Mat4::from_rotation_x(an3.channel_value(b, 3, frame));
        let apply_rot = an3.footer.get(b).map_or(false, |f| f & 0x01 != 0);
        let r = if an3.uses_x20(b) {
            let rl3 = glam::Mat3::from_mat4(rest_locals[b]);
            let rl = Mat4::from_mat3(rl3);
            let id3 = glam::Mat3::from_mat4(identities.get(b).copied().unwrap_or(Mat4::IDENTITY));
            let inv_id = Mat4::from_mat3(id3.inverse());
            let ra3 = glam::Mat3::from_mat4(ra);
            match mode {
                "reference" => {
                    let m = if apply_rot { inv_id * ra } else { inv_id };
                    Mat4::from_mat3(rl3.inverse()) * m
                }
                "addonB" => {
                    // Blender-space final = rest_local_F * inv_identity * euler(rot);
                    // back to file space = transpose of that.
                    let inner = if apply_rot { rl3 * id3.inverse() * ra3 } else { rl3 * id3.inverse() };
                    Mat4::from_mat3(inner.transpose())
                }
                "addonB2" => {
                    let inner = if apply_rot { rl3 * ra3 } else { rl3 };
                    Mat4::from_mat3(inner.transpose())
                }
                "ref_no_identity" => {
                    if apply_rot { Mat4::from_mat3(rl3.inverse()) * ra } else { Mat4::from_mat3(rl3.inverse()) }
                }
                "ref_identity_only" => {
                    let m = if apply_rot { inv_id * ra } else { inv_id };
                    m
                }
                _ => {
                    if apply_rot { rl * ra } else { rl }
                }
            }
        } else {
            ra
        };
        let s = if an3.scale_flag(b) {
            glam::Vec3::new(
                an3.channel_value(b, 6, frame),
                an3.channel_value(b, 7, frame),
                an3.channel_value(b, 8, frame),
            )
        } else {
            glam::Vec3::ONE
        };
        let local = Mat4::from_translation(t) * r * Mat4::from_scale(s);
        let w = if parents[b] < 0 {
            local
        } else {
            worlds[parents[b] as usize] * local
        };
        worlds.push(w);
    }
    worlds
}

fn neutral_worlds(an3: &An3, parents: &[i32], rest_locals: &[Mat4]) -> Vec<Mat4> {
    let mut worlds = Vec::with_capacity(an3.num_bones);
    for b in 0..an3.num_bones {
        let t = glam::Vec3::new(an3.neutral(b, 0), an3.neutral(b, 1), -an3.neutral(b, 2));
        let r = an3.neutral_rot(b, rest_locals.get(b));
        let s = if an3.scale_flag(b) {
            glam::Vec3::new(an3.neutral(b, 6), an3.neutral(b, 7), an3.neutral(b, 8))
        } else {
            glam::Vec3::ONE
        };
        let local = Mat4::from_translation(t) * r * Mat4::from_scale(s);
        let w = if parents[b] < 0 {
            local
        } else {
            worlds[parents[b] as usize] * local
        };
        worlds.push(w);
    }
    worlds
}

fn max_stretch(meshes: &[MeshData], worlds: &[Mat4], rest_worlds: &[Mat4]) -> Vec<f32> {
    meshes
        .iter()
        .map(|md| {
            if md.skin.is_empty() {
                return 0.0;
            }
            let mats: Vec<Mat4> = md
                .skin_bones
                .iter()
                .map(|&b| worlds[b as usize] * rest_worlds[b as usize].inverse())
                .collect();
            let mut posed: Vec<[f32; 3]> = Vec::with_capacity(md.pos.len());
            for v in 0..md.pos.len() {
                let sw = &md.skin[v * 8..v * 8 + 8];
                posed.push(blend_pos(&md.pos[v], sw, &md.skin_bones, &mats));
            }
            let mut worst = 0.0f32;
            for tri in md.idx.chunks(3) {
                let pts = [
                    Vec3::from(posed[tri[0] as usize]),
                    Vec3::from(posed[tri[1] as usize]),
                    Vec3::from(posed[tri[2] as usize]),
                ];
                let rpts = [
                    Vec3::from(md.pos[tri[0] as usize]),
                    Vec3::from(md.pos[tri[1] as usize]),
                    Vec3::from(md.pos[tri[2] as usize]),
                ];
                for k in 0..3 {
                    let len = (pts[k] - pts[(k + 1) % 3]).length();
                    let rlen = (rpts[k] - rpts[(k + 1) % 3]).length();
                    if rlen > 0.01 {
                        worst = worst.max(len / rlen);
                    }
                }
            }
            worst
        })
        .collect()
}
