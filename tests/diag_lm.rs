//! CPU probe for the lightmap path: sample the actual LM0..2 texels at the
//! real LM UVs the shader would use, for a few lightmapped meshes.
//! Run: cargo test --test diag_lm -- --nocapture

use std::collections::BTreeMap;

use rustt::dxt;
use rustt::map;
use rustt::mapmesh;

fn texel_at(rgba: &[u8], w: usize, h: usize, u: f32, v: f32) -> [f32; 3] {
    let x = ((u.clamp(0.0, 0.9999999)) * w as f32).floor() as usize % w;
    let y = ((v.clamp(0.0, 0.9999999)) * h as f32).floor() as usize % h;
    let o = (y * w + x) * 4;
    [
        rgba[o] as f32 / 255.0,
        rgba[o + 1] as f32 / 255.0,
        rgba[o + 2] as f32 / 255.0,
    ]
}

fn lum(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

fn sample_dir_ref(n: [f32; 3], lms: [&[u8]; 3], dims: [(usize, usize); 3], u: f32, v: f32) -> [f32; 3] {
    let (w0, w1, w2) = (
        n[0] * -0.4082482904 + n[1] * -0.7071067811 + n[2] * 0.5773502691,
        n[0] * -0.4082482904 + n[1] * 0.7071067811 + n[2] * 0.5773502691,
        n[0] * 0.8164965809 + n[1] * 0.0 + n[2] * 0.5773502691,
    );
    let s = |i: usize| {
        let c = texel_at(lms[i], dims[i].0, dims[i].1, u, v);
        [
            c[0] * 0.0 + w0 * c[0],
            c[1] * w1 + c[2] * w2,
            if i == 0 {
                0.0
            } else if i == 1 {
                0.0
            } else {
                0.0
            },
        ]
    };
    let a = s(0);
    let b = s(1);
    let c = s(2);
    [
        (a[0] + a[1] + a[2]).max(0.0),
        (b[0] + b[1]).max(0.0),
        (c[0] + c[1]).max(0.0),
    ]
}

#[test]
fn probe_lightmap_uvset_fallback() {
    // The shipped vertex shader adds lightmapOffset to the HIGHEST UV set the
    // mesh declares (USE_vs_uvSet3..uvSet0 chain). Meshes whose material's
    // declared lightmap set falls outside the vertex stride used to lose the
    // lightmap entirely (lm_uv empty -> v_lmuv.x <= 0 -> diffuse = 1.0 =
    // "white metal" surfaces). Count how many meshes/verts the set-fallback
    // in expand_mesh rescues.
    let dir = std::path::Path::new("backup/LEVELS/MAP/MAP");
    let gsc = dir.join("MAP_PC.GSC");
    if !gsc.exists() {
        eprintln!("skipping: backup assets not present");
        return;
    }
    let data = std::fs::read(&gsc).expect("read gsc");
    let m = map::parse(&data).expect("parse map");

    let mut staged = 0usize;
    let mut rescued = 0usize;
    let mut rescued_verts = 0usize;
    let mut rescued_u0 = 0usize;
    for mesh in &m.meshes {
        let Some(mat) = m
            .render_parts
            .iter()
            .find(|p| m.meshes.get(p.mesh).map(|mm| mm.address) == Some(mesh.address))
            .and_then(|p| m.materials.get(p.material))
        else {
            continue;
        };
        if mat.lightmap_stage() == 0 {
            continue;
        }
        staged += 1;
        let stride = mesh.vertex_size as usize;
        let vf = mat.vertex_format_bits;
        let declared = mat.lightmap_uvset() as usize;
        let old = mapmesh::uv_set_offset(stride, vf, declared);
        let new = (0..=declared)
            .rev()
            .find_map(|s| mapmesh::uv_set_offset(stride, vf, s));
        if old.is_none() && new.is_some() {
            rescued += 1;
            let Some(md) = mapmesh::expand_mesh(&m, mesh) else { continue };
            let good = md.lm_uv.iter().filter(|u| u[0] > 0.0).count();
            let any = md.lm_uv.iter().filter(|u| u[0] <= 0.0).count();
            rescued_verts += good;
            rescued_u0 += any;
            eprintln!(
                "mesh {} stride={} vf=0x{:x} mat_uvset={declared} -> set {}: rescued",
                mesh.address, stride, vf, new.unwrap()
            );
        }
    }
    eprintln!(
        "fallback: {rescued}/{staged} staged meshes rescued ({} u>0 verts, {} u<=0)",
        rescued_verts, rescued_u0
    );
}

#[test]
fn probe_lightmap_texels() {
    let dir = std::path::Path::new("backup/LEVELS/MAP/MAP");
    let gsc = dir.join("MAP_PC.GSC");
    if !gsc.exists() {
        eprintln!("skipping: backup assets not present");
        return;
    }
    let data = std::fs::read(&gsc).expect("read gsc");
    let m = map::parse(&data).expect("parse map");

    // Decode the three LM textures (real ids 2..4 = slots via tex_slot).
    let lm_slots: Vec<usize> = (1..=2u16)
        .map(|i| m.tex_slot(i as i16))
        .filter_map(|s| s)
        .collect();
    // use set candidates tex_slot(2), tex_slot(3), tex_slot(4)
    let mut lm_slots = Vec::new();
    for real in 2u16..=4 {
        if let Some(s) = m.tex_slot(real as i16) {
            lm_slots.push(s);
        }
    }
    if lm_slots.len() < 2 {
        eprintln!("could not resolve LM slots ({} found)", lm_slots.len());
        return;
    }
    let decode = |s: usize| {
        let t = &m.textures[s];
        eprintln!("  tex{}: {}x{} payload={}B", s, t.w, t.h, t.payload.len());
        let rgba = dxt::decode_rgba(t.w, t.h, t.fmt, &t.payload).expect("decode");
        (rgba, t.w, t.h)
    };
    let l0 = decode(lm_slots[0]);
    let l1 = decode(lm_slots[1]);
    let l2 = if lm_slots.len() > 2 { Some(decode(lm_slots[2])) } else { None };
    eprintln!("LM slots: {:?}", lm_slots);

    let row_bands = |rgba: &[u8], w: usize, h: usize, bands: usize| -> Vec<f32> {
        let mut out = vec![0f32; bands];
        let mut cnt = vec![0f32; bands];
        for y in 0..h {
            for x in 0..w {
                let o = (y * w + x) * 4;
                let l = 0.2126 * rgba[o] as f32 + 0.7152 * rgba[o + 1] as f32 + 0.0722 * rgba[o + 2] as f32;
                let b = (y * bands / h).min(bands - 1);
                out[b] += l;
                cnt[b] += 1.0;
            }
        }
        for b in 0..bands {
            out[b] /= cnt[b].max(1.0);
        }
        out
    };
    let fmt_bands = |b: &[f32]| {
        b.iter().map(|v| format!("{:.3}", v)).collect::<Vec<_>>().join(" ")
    };
    eprintln!(
        "tex2 row-band meanL 16: {}",
        fmt_bands(&row_bands(&l0.0, l0.1, l0.2, 16))
    );
    eprintln!(
        "tex3 row-band meanL 16: {}",
        fmt_bands(&row_bands(&l1.0, l1.1, l1.2, 16))
    );
    if let Some(t) = &l2 {
        eprintln!(
            "tex4 row-band meanL 16: {}",
            fmt_bands(&row_bands(&t.0, t.1, t.2, 16))
        );
    }

    // First mesh of several lightmapped materials with u>0 coverage.
    let mut shown = 0usize;
    for mesh in &m.meshes {
        if shown >= 6 {
            break;
        }
        let Some(md) = mapmesh::expand_mesh(&m, mesh) else { continue; };
        let Some(pos_uv) = md.lm_uv.iter().find(|u| u[0] > 0.05) else { continue; };
        shown += 1;
        let u = pos_uv[0];
        let v = pos_uv[1];
        let vf = v - v.floor(); // WGSL fract() semantics
        eprintln!(
            "mesh {}: lm_uv=({:.4},{:.4}) fract_v={:.4}",
            mesh.address, u, v, vf
        );
        let st = |rgba: &[u8], w: usize, h: usize| texel_at(rgba, w, h, u, vf);
        let t0 = st(&l0.0, l0.1, l0.2);
        let t1 = st(&l1.0, l1.1, l1.2);
        let t2 = l2.as_ref().map(|t| st(&t.0, t.1, t.2)).unwrap_or([0.0; 3]);
        eprintln!(
            "  lm0=({:.3},{:.3},{:.3}) L={:.3} lm1=({:.3},{:.3},{:.3}) L={:.3} lm2=({:.3},{:.3},{:.3}) L={:.3}",
            t0[0], t0[1], t0[2], lum(t0),
            t1[0], t1[1], t1[2], lum(t1),
            t2[0], t2[1], t2[2], lum(t2),
        );
        for n in [[0f32, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
            let w0 = (n[0] * -0.4082482904 + n[1] * -0.7071067811 + n[2] * 0.5773502691).max(0.0);
            let w1 = (n[0] * -0.4082482904 + n[1] * 0.7071067811 + n[2] * 0.5773502691).max(0.0);
            let w2 = (n[0] * 0.8164965809 + n[1] * 0.0 + n[2] * 0.5773502691).max(0.0);
            let blend = [
                t0[0] * w0 + t1[0] * w1 + t2[0] * w2,
                t0[1] * w0 + t1[1] * w1 + t2[1] * w2,
                t0[2] * w0 + t1[2] * w1 + t2[2] * w2,
            ];
            eprintln!(
                "  n=({},{},{}) w=({:.2},{:.2},{:.2}) blend=({:.3},{:.3},{:.3}) L={:.3}",
                n[0], n[1], n[2], w0, w1, w2, blend[0], blend[1], blend[2], lum(blend)
            );
        }
    }
    if shown == 0 {
        eprintln!("no lightmapped meshes with u>0.05 found");
    }

    // Histogram of sampled texels across ALL lightmapped vertices: %white,
    // %black, %mid. If the UVs point at atlas background instead of lightmap
    // patches, the white share dominates.
    let lms = [(&l0.0, l0.1, l0.2), (&l1.0, l1.1, l1.2)];
    let mut counts = [[0usize; 3]; 2]; // white, black, mid
    let mut verts_seen = 0usize;
    let mut meshes_seen = 0usize;
    for mesh in &m.meshes {
        let Some(md) = mapmesh::expand_mesh(&m, mesh) else { continue; };
        if md.lm_uv.is_empty() {
            continue;
        }
        meshes_seen += 1;
        for uv in md.lm_uv.iter().take(64) {
            if uv[0] <= 0.0 {
                continue;
            }
            verts_seen += 1;
            let vf = uv[1] - uv[1].floor();
            for (t, data) in lms.iter().enumerate() {
                let c = texel_at(data.0, data.1, data.2, uv[0], vf);
                let l = lum(c);
                let b = if l > 0.95 { 0 } else if l < 0.05 { 1 } else { 2 };
                counts[t][b] += 1;
            }
        }
    }
    eprintln!(
        "histogram over {verts_seen} verts / {meshes_seen} meshes: white black mid"
    );
    for t in 0..2 {
        let tot = counts[t].iter().sum::<usize>().max(1) as f64;
        eprintln!(
            "  lm{}: white={:.1}% black={:.1}% mid={:.1}%",
            t,
            counts[t][0] as f64 / tot * 100.0,
            counts[t][1] as f64 / tot * 100.0,
            counts[t][2] as f64 / tot * 100.0,
        );
    }

    // Which UV channel lines up with the atlas content? Raw-scan every
    // candidate UV set per lightmapped mesh (mirroring mapdump --lmuv) and
    // aggregate the mean sampled luminance of tex2 per (stride, set): the
    // real channel should hit bright patches instead of the ~3% mid share.
    let mut agg: BTreeMap<(usize, usize), (usize, f64, usize, usize)> = BTreeMap::new(); // (count, sumL, white, black)
    for mesh in &m.meshes {
        let Some(mat) = m
            .render_parts
            .iter()
            .find(|p| m.meshes.get(p.mesh).map(|mm| mm.address) == Some(mesh.address))
            .and_then(|p| m.materials.get(p.material))
        else {
            continue;
        };
        if mat.lightmap_stage() == 0 {
            continue;
        }
        let stride = mesh.vertex_size as usize;
        let vf = mat.vertex_format_bits;
        let Some(vb) = m.vertex_buffers.get(mesh.vertex_list_id as usize) else { continue };
        let base = mesh.vertex_offset as usize;
        let n = (mesh.vertex_count as usize).min((vb.len() / stride).saturating_sub(base));
        for set in 0..4 {
            let Some(off) = rustt::mapmesh::uv_set_offset(stride, vf, set) else { continue };
            let mut sum = 0.0f64;
            let mut cnt = 0usize;
            let mut wht = 0usize;
            let mut blk = 0usize;
            for v in 0..n {
                let o = base * stride + v * stride + off;
                let u = f32::from_le_bytes(vb[o..o + 4].try_into().unwrap());
                let vv = f32::from_le_bytes(vb[o + 4..o + 8].try_into().unwrap());
                if !u.is_finite() || !vv.is_finite() || u <= 0.0 || !(0.0..=1.0).contains(&u) {
                    continue;
                }
                let c = texel_at(&l0.0, l0.1, l0.2, u, vv - vv.floor());
                let l = lum(c);
                sum += l as f64;
                cnt += 1;
                if l > 0.95 {
                    wht += 1;
                }
                if l < 0.05 {
                    blk += 1;
                }
            }
            if cnt > 0 {
                let e = agg.entry((stride, set)).or_insert((0, 0.0, 0, 0));
                e.0 += cnt;
                e.1 += sum;
                e.2 += wht;
                e.3 += blk;
            }
        }
    }
    eprintln!("(stride,set): verts meanL% white% black%");
    for ((stride, set), (cnt, sum, wht, blk)) in &agg {
        eprintln!(
            "(stride {}, set {}): verts={} meanL={:.1}% white={:.1}% black={:.1}%",
            stride,
            set,
            cnt,
            sum / *cnt as f64 / 255.0 * 100.0,
            *wht as f64 / *cnt as f64 * 100.0,
            *blk as f64 / *cnt as f64 * 100.0,
        );
    }

    // Atlas layout: meanL in an 8x8 grid of tex2/tex3 (+ bounding box of
    // bright content) so we can see where the patches actually sit.
    let atlas_grid = |rgba: &[u8], w: usize, h: usize, g: usize| -> Vec<f32> {
        let mut cell = vec![0.0f64; g * g];
        let mut n = vec![0usize; g * g];
        for y in 0..h {
            for x in 0..w {
                let o = (y * w + x) * 4;
                let l = 0.2126 * rgba[o] as f64 + 0.7152 * rgba[o + 1] as f64 + 0.0722 * rgba[o + 2] as f64;
                let c = (x * g / w).min(g - 1) + (y * g / h).min(g - 1) * g;
                cell[c] += l;
                n[c] += 1;
            }
        }
        cell.iter().zip(n).map(|(s, c)| (s / c.max(1) as f64) as f32).collect()
    };
    let show = |name: &str, grid: &[f32], g: usize| {
        eprint!("{name} meanL/255 8x8:");
        for (i, v) in grid.iter().enumerate() {
            if i % g == 0 {
                eprint!("\n  ");
            }
            eprint!("{:5.0}", v / 255.0 * 100.0);
        }
        eprintln!();
    };
    show("lm0(tex2)", &atlas_grid(&l0.0, l0.1, l0.2, 8), 8);
    show("lm1(tex3)", &atlas_grid(&l1.0, l1.1, l1.2, 8), 8);
    if let Some(t) = &l2 {
        show("lm2(tex4)", &atlas_grid(&t.0, t.1, t.2, 8), 8);
    }
    let bbox = |rgba: &[u8], w: usize, h: usize, thr: f32| -> (f32, f32, f32, f32) {
        let mut umin = f32::MAX;
        let mut umax = f32::MIN;
        let mut vmin = f32::MAX;
        let mut vmax = f32::MIN;
        for y in 0..h {
            for x in 0..w {
                let o = (y * w + x) * 4;
                let l = 0.2126 * rgba[o] as f32 + 0.7152 * rgba[o + 1] as f32 + 0.0722 * rgba[o + 2] as f32;
                if l > thr {
                    umin = umin.min(x as f32 / w as f32);
                    umax = umax.max(x as f32 / w as f32);
                    vmin = vmin.min(y as f32 / h as f32);
                    vmax = vmax.max(y as f32 / h as f32);
                }
            }
        }
        (umin, umax, vmin, vmax)
    };
    for (name, t) in [("lm0", &l0), ("lm1", &l1)] {
        let (u0, u1, v0, v1) = bbox(&t.0, t.1, t.2, 50.0);
        eprintln!(
            "{name} bright(>50/255) bbox: u[{:.3}..{:.3}] v[{:.3}..{:.3}]",
            u0, u1, v0, v1
        );
    }

    // Full texture inventory around each stage-2 material's lightmap set:
    // slot, size, meanL of the decoded base level. The real lightmap atlases
    // are the dense-gradient textures; anything else (black/white/paletted)
    // is not a lightmap.
    let tex_mean = |idx: usize| -> Option<(usize, usize, f32)> {
        let t = m.textures.get(idx)?;
        let rgba = dxt::decode_rgba(t.w, t.h, t.fmt, &t.payload).ok()?;
        let mut sum = 0.0f64;
        for px in rgba.chunks_exact(4) {
            sum += 0.2126 * px[0] as f64 + 0.7152 * px[1] as f64 + 0.0722 * px[2] as f64;
        }
        Some((t.w, t.h, (sum / (rgba.len() / 4).max(1) as f64) as f32))
    };
    let mut seen_sets = std::collections::BTreeSet::new();
    for mat in &m.materials {
        if mat.lightmap_stage() == 0 || !seen_sets.insert(mat.lightmap_set_index) {
            continue;
        }
        let si = mat.lightmap_set_index as i32;
        eprint!(
            "set={si}: tex {si}: {:?} | ",
            tex_mean(si.max(0) as usize)
        );
        for d in 0..5i32 {
            let i = si + d;
            if i >= 0 && (i as usize) < m.textures.len() {
                eprint!(
                    "0x{:02x}={:?} ",
                    i,
                    tex_mean(i as usize).map(|(w, h, l)| format!("{w}x{h} L={l:.0}"))
                );
            }
        }
        eprintln!();
    }

    // UV distribution: where do the raw lightmap UVs actually sit? 16-band
    // histograms of u and vs = v+1 (the fract() mapping) across every
    // lightmapped vertex.
    let mut uh = [0usize; 16];
    let mut vh = [0usize; 16];
    let mut total = 0usize;
    for mesh in &m.meshes {
        let Some(mat) = m
            .render_parts
            .iter()
            .find(|p| m.meshes.get(p.mesh).map(|mm| mm.address) == Some(mesh.address))
            .and_then(|p| m.materials.get(p.material))
        else {
            continue;
        };
        if mat.lightmap_stage() == 0 {
            continue;
        }
        let stride = mesh.vertex_size as usize;
        let Some(off) = rustt::mapmesh::uv_set_offset(
            stride,
            mat.vertex_format_bits,
            mat.lightmap_uvset() as usize,
        ) else {
            continue;
        };
        let Some(vb) = m.vertex_buffers.get(mesh.vertex_list_id as usize) else { continue };
        let base = mesh.vertex_offset as usize;
        let n = (mesh.vertex_count as usize).min((vb.len() / stride).saturating_sub(base));
        for v in 0..n {
            let o = base * stride + v * stride + off;
            let u = f32::from_le_bytes(vb[o..o + 4].try_into().unwrap());
            let vv = f32::from_le_bytes(vb[o + 4..o + 8].try_into().unwrap());
            if !u.is_finite() || !vv.is_finite() || u <= 0.0 {
                continue;
            }
            let vs = vv - vv.floor();
            let bi = |f: f32| -> usize { ((f * 16.0) as usize).min(15) };
            uh[bi(u % 1.0)] += 1;
            vh[bi(vs % 1.0)] += 1;
            total += 1;
        }
    }
    eprintln!("uv distribution ({total} verts):");
    eprintln!(
        "  u:  {}",
        uh.iter().map(|c| format!("{:6}", c * 100 / total.max(1))).collect::<String>()
    );
    eprintln!(
        "  v+: {}",
        vh.iter().map(|c| format!("{:6}", c * 100 / total.max(1))).collect::<String>()
    );

    // ASCII-paint the decoded atlases so we can SEE the structure
    // (patches/gradients vs decoder garbage) with our own eyes.
    let art = |rgba: &[u8], w: usize, h: usize, cols: usize, rows: usize| -> String {
        let ramp = b" .:-=+*#%@";
        let mut out = String::new();
        for r in 0..rows {
            for c in 0..cols {
                let mut sum = 0.0f64;
                let mut n = 0usize;
                for y in r * h / rows..(r + 1) * h / rows {
                    for x in c * w / cols..(c + 1) * w / cols {
                        let o = (y * w + x) * 4;
                        sum += 0.2126 * rgba[o] as f64 + 0.7152 * rgba[o + 1] as f64 + 0.0722 * rgba[o + 2] as f64;
                        n += 1;
                    }
                }
                let l = (sum / n.max(1) as f64) as usize;
                let idx = (l * (ramp.len() - 1) / 255).min(ramp.len() - 1);
                out.push(ramp[idx] as char);
            }
            out.push('\n');
        }
        out
    };
    eprintln!("lm0(tex2 512x128) ASCII 64x32:\n{}", art(&l0.0, l0.1, l0.2, 64, 32));
    eprintln!("lm1(tex3 1024x256) ASCII 64x32:\n{}", art(&l1.0, l1.1, l1.2, 64, 32));

    // Hunt the real lightmap atlas: biggest textures first, with 24x12
    // ASCII thumbnails. Lightmap atlases = large, soft gradients; glyphs =
    // small repeating cells; albedo = strong color noise.
    let mut ranked: Vec<usize> = (0..m.textures.len()).collect();
    ranked.sort_by(|a, b| {
        let s = |i: &usize| m.textures[*i].w * m.textures[*i].h;
        s(b).cmp(&s(a))
    });
    for i in ranked.iter().take(16) {
        let t = &m.textures[*i];
        let Ok(rgba) = dxt::decode_rgba(t.w, t.h, t.fmt, &t.payload) else { continue };
        eprintln!(
            "tex[{i}] {}x{} L={:.0}:\n{}",
            t.w,
            t.h,
            tex_mean(*i).map(|x| x.2).unwrap_or(0.0),
            art(&rgba, t.w, t.h, 24, 12)
        );
    }

    // Sweep big-candidate textures as the lightmap: for every lightmapped
    // vertex (uvset as the game picks it), aggregate sampled luminance and
    // white/black shares. The real lightmap should light up.
    for cand in [100usize, 101, 112, 257, 258, 259, 260] {
        let Some(t) = m.textures.get(cand) else { continue };
        let Ok(rgba) = dxt::decode_rgba(t.w, t.h, t.fmt, &t.payload) else { continue };
        let mut sum = 0.0f64;
        let mut cnt = 0usize;
        let mut wht = 0usize;
        let mut blk = 0usize;
        for mesh in &m.meshes {
            let Some(mat) = m
                .render_parts
                .iter()
                .find(|p| m.meshes.get(p.mesh).map(|mm| mm.address) == Some(mesh.address))
                .and_then(|p| m.materials.get(p.material))
            else {
                continue;
            };
            if mat.lightmap_stage() == 0 {
                continue;
            }
            let stride = mesh.vertex_size as usize;
            let Some(off) = rustt::mapmesh::uv_set_offset(
                stride,
                mat.vertex_format_bits,
                mat.lightmap_uvset() as usize,
            ) else {
                continue;
            };
            let Some(vb) = m.vertex_buffers.get(mesh.vertex_list_id as usize) else { continue };
            let base = mesh.vertex_offset as usize;
            let n = (mesh.vertex_count as usize).min((vb.len() / stride).saturating_sub(base));
            for v in 0..n {
                let o = base * stride + v * stride + off;
                let u = f32::from_le_bytes(vb[o..o + 4].try_into().unwrap());
                let vv = f32::from_le_bytes(vb[o + 4..o + 8].try_into().unwrap());
                if !u.is_finite() || !vv.is_finite() || u <= 0.0 {
                    continue;
                }
                let c = texel_at(&rgba, t.w, t.h, u, vv - vv.floor());
                let l = lum(c);
                sum += l as f64;
                cnt += 1;
                if l > 0.95 {
                    wht += 1;
                }
                if l < 0.05 {
                    blk += 1;
                }
            }
        }
        if cnt > 0 {
            eprintln!(
                "cand tex[{cand}] {}x{}: verts={cnt} meanL={:.1}% white={:.1}% black={:.1}%",
                t.w,
                t.h,
                sum / cnt as f64 / 255.0 * 100.0,
                wht as f64 / cnt as f64 * 100.0,
                blk as f64 / cnt as f64 * 100.0,
            );
        }
    }

    // Sanity: texel_at must read the uniform-bright tex[100] as ~55% at ANY
    // sample point. Broken sampling would explain every fishy histogram.
    let t = &m.textures[100];
    let Ok(rgba) = dxt::decode_rgba(t.w, t.h, t.fmt, &t.payload) else { return };
    eprintln!(
        "sanity buffer len={} expected={} first px {:?} at(0.5,0.5)={:?}",
        rgba.len(),
        t.w * t.h * 4,
        &rgba[0..3],
        texel_at(&rgba, t.w, t.h, 0.5, 0.5)
    );
    let mut s0 = 0.0f64;
    let mut n0 = 0usize;
    for i in 0..20 {
        for j in 0..20 {
            s0 += lum(texel_at(&rgba, t.w, t.h, i as f32 / 20.0, j as f32 / 20.0)) as f64;
            n0 += 1;
        }
    }
    eprintln!(
        "sanity tex[100] grid 20x20: meanL={:.1}% (raw art suggested ~55%)",
        s0 / n0 as f64 / 255.0 * 100.0
    );

    // Payload-vs-DXT-size audit: a short payload makes decode fill the rest
    // with the FIRST blocks (the "uniform bright" look) — i.e. those
    // "lightmaps" are an illusion of the decoder, not real content.
    for i in [2usize, 3, 100, 101, 112, 257, 260] {
        let Some(t) = m.textures.get(i) else { continue };
        let dxt5 = t.fmt == rustt::ghg::TextureFmt::Dxt5;
        let bpp = if dxt5 { 16u64 } else { 8u64 };
        let expect = (t.w as u64).max(1) * (t.h as u64).max(1) / 16 * bpp;
        eprintln!(
            "tex[{i}] {}x{} fmt={} payload={}B expected={}B {}",
            t.w,
            t.h,
            if dxt5 { "DXT5" } else { "DXT1" },
            t.payload.len(),
            expect,
            if (t.payload.len() as u64) >= expect { "OK" } else { "SHORT!" }
        );
    }

    // Decoder coherence: for tex[100] (uniform art) and tex[2] (glyph art),
    // check whether the decode actually varies across the buffer.
    for i in [100usize, 2] {
        let t = &m.textures[i];
        let Ok(rgba) = dxt::decode_rgba(t.w, t.h, t.fmt, &t.payload) else { continue };
        let p0 = &rgba[0..4];
        let mid = &rgba[(rgba.len() / 4)..(rgba.len() / 4 + 4)];
        let p1 = &rgba[rgba.len() - 4..];
        let mut eq = 0usize;
        let step = 4096;
        for o in (0..rgba.len()).step_by(step) {
            if &rgba[o..o + 4] == p0 {
                eq += 1;
            }
        }
        let nonzero = rgba.chunks_exact(4).filter(|p| p[0] != 0 || p[1] != 0 || p[2] != 0).count();
        eprintln!(
            "tex[{i}] decode px0={p0:?} mid={mid:?} last={p1:?} identical-to-px0 in {} sampled spots of {}; nonzero={}",
            eq,
            rgba.len() / step,
            nonzero
        );
    }

    // Same-buffer glyph re-check: paint from a FRESH decode right here.
    let t = &m.textures[2];
    let Ok(rgba2) = dxt::decode_rgba(t.w, t.h, t.fmt, &t.payload) else { return };
    eprintln!("tex[2] fresh ASCII 64x32:\n{}", art(&rgba2, t.w, t.h, 64, 32));

    // Raw UV statistics from the exact same read path the sweep uses.
    let mut umin = f32::MAX;
    let mut umax = f32::MIN;
    let mut vmin = f32::MAX;
    let mut vmax = f32::MIN;
    let mut usum = 0.0f64;
    let mut vsum = 0.0f64;
    let mut n = 0usize;
    for mesh in &m.meshes {
        let Some(mat) = m
            .render_parts
            .iter()
            .find(|p| m.meshes.get(p.mesh).map(|mm| mm.address) == Some(mesh.address))
            .and_then(|p| m.materials.get(p.material))
        else {
            continue;
        };
        if mat.lightmap_stage() == 0 {
            continue;
        }
        let stride = mesh.vertex_size as usize;
        let Some(off) = rustt::mapmesh::uv_set_offset(
            stride,
            mat.vertex_format_bits,
            mat.lightmap_uvset() as usize,
        ) else {
            continue;
        };
        let Some(vb) = m.vertex_buffers.get(mesh.vertex_list_id as usize) else { continue };
        let base = mesh.vertex_offset as usize;
        let cnt = (mesh.vertex_count as usize).min((vb.len() / stride).saturating_sub(base));
        for v in 0..cnt {
            let o = base * stride + v * stride + off;
            let u = f32::from_le_bytes(vb[o..o + 4].try_into().unwrap());
            let vv = f32::from_le_bytes(vb[o + 4..o + 8].try_into().unwrap());
            if !u.is_finite() || !vv.is_finite() {
                continue;
            }
            umin = umin.min(u);
            umax = umax.max(u);
            vmin = vmin.min(vv);
            vmax = vmax.max(vv);
            usum += u as f64;
            vsum += vv as f64;
            n += 1;
        }
    }
    eprintln!(
        "sweep-path UV stats (n={n}): u[{umin:.4}..{umax:.4}] mean={:.4} v[{vmin:.4}..{vmax:.4}] mean={:.4}",
        usum / n.max(1) as f64,
        vsum / n.max(1) as f64
    );

    // Per-mesh cell probe: find the mesh with the most lm-uv verts, print its
    // UV bbox, and overlay that bbox on the tex2 art to see whether it lands
    // on bright cells (in-region lum should be much higher than global).
    let mut best = None;
    for mesh in &m.meshes {
        let Some(mat) = m
            .render_parts
            .iter()
            .find(|p| m.meshes.get(p.mesh).map(|mm| mm.address) == Some(mesh.address))
            .and_then(|p| m.materials.get(p.material))
        else {
            continue;
        };
        if mat.lightmap_stage() == 0 {
            continue;
        }
        let stride = mesh.vertex_size as usize;
        let Some(off) = rustt::mapmesh::uv_set_offset(
            stride,
            mat.vertex_format_bits,
            mat.lightmap_uvset() as usize,
        ) else {
            continue;
        };
        let Some(vb) = m.vertex_buffers.get(mesh.vertex_list_id as usize) else { continue };
        let base = mesh.vertex_offset as usize;
        let cnt = (mesh.vertex_count as usize).min((vb.len() / stride).saturating_sub(base));
        let good = (0..cnt)
            .map(|v| {
                let o = base * stride + v * stride + off;
                (
                    f32::from_le_bytes(vb[o..o + 4].try_into().unwrap()),
                    f32::from_le_bytes(vb[o + 4..o + 8].try_into().unwrap()),
                )
            })
            .filter(|(u, vv)| u.is_finite() && vv.is_finite() && *u > 0.0)
            .count();
        if good > 0 && best.map_or(true, |(_, g)| good > g) {
            best = Some((mesh, good));
        }
    }
    if let Some((mesh, good)) = best {
        let stride = mesh.vertex_size as usize;
        let mat = m
            .render_parts
            .iter()
            .find(|p| m.meshes.get(p.mesh).map(|mm| mm.address) == Some(mesh.address))
            .and_then(|p| m.materials.get(p.material))
            .unwrap();
        let off = rustt::mapmesh::uv_set_offset(
            stride,
            mat.vertex_format_bits,
            mat.lightmap_uvset() as usize,
        )
        .unwrap();
        let vb = &m.vertex_buffers[mesh.vertex_list_id as usize];
        let base = mesh.vertex_offset as usize;
        let cnt = (mesh.vertex_count as usize).min((vb.len() / stride).saturating_sub(base));
        let mut ub0 = f32::MAX;
        let mut ub1 = f32::MIN;
        let mut vb0 = f32::MAX;
        let mut vb1 = f32::MIN;
        for v in 0..cnt {
            let o = base * stride + v * stride + off;
            let u = f32::from_le_bytes(vb[o..o + 4].try_into().unwrap());
            let vv = f32::from_le_bytes(vb[o + 4..o + 8].try_into().unwrap());
            if !u.is_finite() || !vv.is_finite() || u <= 0.0 {
                continue;
            }
            ub0 = ub0.min(u);
            ub1 = ub1.max(u);
            vb0 = vb0.min(vv);
            vb1 = vb1.max(vv);
        }
        let (tw, th) = (l0.1, l0.2);
        let vf0 = vb0.floor();
        let inr = |u: f32, v: f32| u >= ub0 && u <= ub1 && v >= (vb0 - vf0) && v <= (vb1 - vf0);
        let mut s_in = 0.0f64;
        let mut s_out = 0.0f64;
        let mut n_in = 0usize;
        let mut n_out = 0usize;
        for y in 0..64 {
            for x in 0..128 {
                let u = (x as f32 + 0.5) / 128.0;
                let v = (y as f32 + 0.5) / 64.0;
                let l = lum(texel_at(&l0.0, tw, th, u, v)) as f64;
                if inr(u, v) {
                    s_in += l;
                    n_in += 1;
                } else {
                    s_out += l;
                    n_out += 1;
                }
            }
        }
        eprintln!(
            "mesh uv bbox: u[{ub0:.4}..{ub1:.4}] v[{vb0:.4}..{vb1:.4}] stride={stride} verts={good}; tex2 lum mean in-bbox={:.1}% out={:.1}%",
            s_in / n_in.max(1) as f64 * 100.0,
            s_out / n_out.max(1) as f64 * 100.0
        );
        let mut artb = String::new();
        for y in 0..32 {
            for x in 0..64 {
                let u = (x as f32 + 0.5) / 64.0;
                let v = (y as f32 + 0.5) / 32.0;
                let c = texel_at(&l0.0, tw, th, u, v);
                let l = lum(c);
                let ch = if l > 0.7 {
                    '#'
                } else if l > 0.25 {
                    '+'
                } else {
                    '.'
                };
                if inr(u, v) {
                    artb.push(if l > 0.5 { '@' } else { 'o' });
                } else {
                    artb.push(ch);
                }
            }
            artb.push('\n');
        }
        eprintln!("tex2 art with mesh bbox overlay (@=bright in-bbox, o=dark in-bbox, +=mid, .=dark):\n{artb}");

        // Sample ALL FOUR set textures at this mesh's verts. The directional
        // layers (set+1..+3) have never been sampled with real UVs.
        let set = mat.lightmap_set_index;
        let mut per_tex = Vec::new();
        for i in 0..4u16 {
            let t = &m.textures[set as usize + i as usize];
            let Ok(rgba) = dxt::decode_rgba(t.w, t.h, t.fmt, &t.payload) else {
                per_tex.push((i, None));
                continue;
            };
            let mut sum = 0.0f64;
            let mut wht = 0usize;
            let mut blk = 0usize;
            let mut ncnt = 0usize;
            for v2 in 0..cnt {
                let o2 = base * stride + v2 * stride + off;
                let u = f32::from_le_bytes(vb[o2..o2 + 4].try_into().unwrap());
                let vv = f32::from_le_bytes(vb[o2 + 4..o2 + 8].try_into().unwrap());
                if !u.is_finite() || !vv.is_finite() || u <= 0.0 {
                    continue;
                }
                let c = texel_at(&rgba, t.w, t.h, u, vv - vv.floor());
                let l = lum(c);
                sum += l as f64;
                ncnt += 1;
                if l > 0.9 {
                    wht += 1;
                }
                if l < 0.1 {
                    blk += 1;
                }
            }
            per_tex.push((
                i,
                Some((sum / ncnt.max(1) as f64 * 100.0, wht, blk, ncnt, t.w, t.h, (t.fmt == rustt::ghg::TextureFmt::Dxt5))),
            ));
        }
        for (i, r) in per_tex {
            match r {
                Some((mns, wht, blk, cnt, w, h, dxt5)) => eprintln!(
                    "set+{i} tex[{}] {}x{} dxt5={dxt5}: verts={cnt} lum%={mns:.1} white={:.1}% black={:.1}%",
                    set as usize + i as usize,
                    w,
                    h,
                    wht as f64 / cnt.max(1) as f64 * 100.0,
                    blk as f64 / cnt.max(1) as f64 * 100.0
                ),
                None => eprintln!("set+{i}: no texture"),
            }
        }
        // Sample the same verts against the top-view light pages (slots
        // 253..256 = real 243..246 = set+241..+244).
        for i in 0..4u16 {
            let Some(t) = m.textures.get(253usize + i as usize) else { continue };
            let Ok(rgba) = dxt::decode_rgba(t.w, t.h, t.fmt, &t.payload) else {
                eprintln!("page+{i}: decode fail");
                continue;
            };
            let mut sum = 0.0f64;
            let mut wht = 0usize;
            let mut blk = 0usize;
            let mut nn = 0usize;
            for v2 in 0..cnt {
                let o2 = base * stride + v2 * stride + off;
                let u = f32::from_le_bytes(vb[o2..o2 + 4].try_into().unwrap());
                let vv = f32::from_le_bytes(vb[o2 + 4..o2 + 8].try_into().unwrap());
                if !u.is_finite() || !vv.is_finite() || u <= 0.0 {
                    continue;
                }
                let c = texel_at(&rgba, t.w, t.h, u, vv - vv.floor());
                let l = lum(c);
                sum += l as f64;
                nn += 1;
                if l > 0.9 {
                    wht += 1;
                }
                if l < 0.1 {
                    blk += 1;
                }
            }
            eprintln!(
                "page+{i} {}x{}: verts={nn} lum%={:.1} white={:.1}% black={:.1}%",
                t.w,
                t.h,
                sum / nn.max(1) as f64 * 100.0,
                wht as f64 / nn.max(1) as f64 * 100.0,
                blk as f64 / nn.max(1) as f64 * 100.0
            );
        }
    }

    // Lightmap display-command states: type distribution + which texture
    // real-indices (the pages?) actually get bound.
    let mut ty_hist = BTreeMap::<u32, usize>::new();
    let mut tex_hist = BTreeMap::<i32, usize>::new();
    let mut off_ex = Vec::new();
    let mut with_lm = 0usize;
    for st in m.lightmaps.values() {
        *ty_hist.entry(st.ty).or_insert(0) += 1;
        for t in st.tex {
            *tex_hist.entry(t).or_insert(0) += 1;
        }
        if off_ex.len() < 8 {
            off_ex.push((st.ty, st.off));
        }
    }
    for (ty, n) in &ty_hist {
        eprintln!("lightmap cmd type {ty}: {n} states");
    }
    let top: Vec<_> = tex_hist.iter().take(6).collect();
    eprintln!("lightmap tex hist (top 6): {top:?}");
    for (mat_id, lm_addr) in m
        .render_parts
        .iter()
        .filter(|p| p.lightmap != 0)
        .map(|p| (p.material, p.lightmap))
        .collect::<std::collections::HashSet<_>>()
        .iter()
        .take(10)
    {
        let st = m.lightmaps.get(lm_addr).unwrap();
        eprintln!(
            "mat {mat_id}: lightmap ty={} tex={:?} off={:?}",
            st.ty, st.tex, st.off
        );
    }
    with_lm = m.render_parts.iter().filter(|p| p.lightmap != 0).count();
    eprintln!("parts with lightmap cmd: {with_lm}/{}", m.render_parts.len());

    // Set indices across ALL materials (incl. stage-0 excluded from --lm).
    let mut set_all = BTreeMap::<u8, (usize, usize)>::new();
    for mat in &m.materials {
        let e = set_all.entry(mat.lightmap_set_index).or_insert((0, 0));
        e.0 += 1;
        if mat.lightmap_stage() != 0 {
            e.1 += 1;
        }
    }
    for (k, (tot, staged)) in &set_all {
        if *tot > 0 {
            eprintln!("set idx {k}: total={tot} staged={staged}");
        }
    }

    // Slot vs index: what does tex_slot() actually map to? The art above used
    // tex_slot(2), the coherence probe used textures[2].
    for sl in 248i16..293 {
        eprintln!(
            "slot {sl}: real={:?} tex[{}] {}x{} dxt5={}",
            m.tex_slot(sl),
            sl as usize,
            m.textures.get(sl as usize).map(|t| t.w).unwrap_or(0),
            m.textures.get(sl as usize).map(|t| t.h).unwrap_or(0),
            m.textures
                .get(sl as usize)
                .map(|t| t.fmt == rustt::ghg::TextureFmt::Dxt5)
                .unwrap_or(false)
        );
    }

    // Offsets per (stride,set): if every set collapses to the same offset we
    // have been reading one channel only (likely the diffuse UVs) all along.
    let mut spoke = BTreeMap::new();
    for mesh in &m.meshes {
        let Some(mat) = m
            .render_parts
            .iter()
            .find(|p| m.meshes.get(p.mesh).map(|mm| mm.address) == Some(mesh.address))
            .and_then(|p| m.materials.get(p.material))
        else {
            continue;
        };
        let stride = mesh.vertex_size as usize;
        let vf = mat.vertex_format_bits;
        for set in 0..4 {
            if let Some(off) = rustt::mapmesh::uv_set_offset(stride, vf, set) {
                spoke.entry((stride, off)).or_insert_with(|| vec![set, mat.lightmap_uvset() as usize]);
            }
        }
    }
    for ((stride, off), sets) in &spoke {
        eprintln!("stride {stride}: set{off}<->(sets {sets:?})");
    }
}

#[test]
fn trace_tex111_and_hologram() {
    let dir = std::path::Path::new("backup/LEVELS/MAP/MAP");
    let gsc = dir.join("MAP_PC.GSC");
    if !gsc.exists() {
        eprintln!("skipping: backup assets not present");
        return;
    }
    let data = std::fs::read(&gsc).expect("read gsc");
    let m = map::parse(&data).expect("parse map");

    // 1) What real_index corresponds to texture slot 111?
    if let Some(&ri) = m.texture_real_index.get(111) {
        let t = &m.textures[111];
        eprintln!("texture slot 111: real_index={ri}  {}x{}", t.w, t.h);
    } else {
        eprintln!("texture slot 111: OUT OF RANGE (only {} slots)", m.textures.len());
        return;
    }
    let real_111 = m.texture_real_index[111] as i32;

    // 2) Which lightmap states reference real_index 111 in any tex[] slot?
    eprintln!("\n--- lightmap states referencing real_index {real_111} ---");
    for (addr, ls) in &m.lightmaps {
        for (j, &t) in ls.tex.iter().enumerate() {
            if t == real_111 {
                eprintln!(
                    "  lm addr=0x{addr:x} type={} tex={:?} off=({:.4},{:.4},{:.4},{:.4})  <-- tex[{j}]={t}",
                    ls.ty, ls.tex, ls.off[0], ls.off[1], ls.off[2], ls.off[3]
                );
            }
        }
    }

    // 3) Find render parts whose lightmap state references real_111,
    //    and dump their materials + mesh centers.
    eprintln!("\n--- parts using lightmap with real_index {real_111} ---");
    for (pi, part) in m.render_parts.iter().enumerate() {
        let Some(ls) = m.lightmaps.get(&part.lightmap) else {
            continue;
        };
        if !ls.tex.iter().any(|&t| t == real_111) {
            continue;
        }
        let mat = &m.materials[part.material];
        let mesh = &m.meshes[part.mesh];
        let md = mapmesh::expand_mesh(&m, mesh);
        let (xmin, xmax, ymin, ymax, zmin, zmax) = md.as_ref().map_or(
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            |md| {
                let mut xn = f32::INFINITY;
                let mut xx = f32::NEG_INFINITY;
                let mut yn = f32::INFINITY;
                let mut yx = f32::NEG_INFINITY;
                let mut zn = f32::INFINITY;
                let mut zx = f32::NEG_INFINITY;
                for p in &md.pos {
                    xn = xn.min(p[0]); xx = xx.max(p[0]);
                    yn = yn.min(p[1]); yx = yx.max(p[1]);
                    zn = zn.min(p[2]); zx = zx.max(p[2]);
                }
                (xn, xx, yn, yx, zn, zx)
            },
        );
        eprintln!(
            "  part[{pi}] mesh={} mat={} lm_key=0x{:x} type={} tex={:?} off=({:.4},{:.4},{:.4},{:.4})",
            part.mesh, part.material, part.lightmap, ls.ty, ls.tex, ls.off[0], ls.off[1], ls.off[2], ls.off[3]
        );
        eprintln!(
            "    mat: id={} tex_id={} diffuse=({:.3},{:.3},{:.3}) flags=0x{:x} prelit={} lm_stage={} lighting={} spec=({:.3},{:.3},{:.3},{:.3}) vfbits=0x{:x}",
            mat.id, mat.tex_id, mat.diffuse[0], mat.diffuse[1], mat.diffuse[2],
            mat.texture_flags,
            mat.shader_defines & 0x1000 != 0,
            mat.lightmap_stage(),
            mat.lighting_stage,
            mat.specular_params[0], mat.specular_params[1], mat.specular_params[2], mat.specular_params[3],
            mat.vertex_format_bits,
        );
        eprintln!(
            "    bbox: ({:.2},{:.2},{:.2})..({:.2},{:.2},{:.2})  verts={}",
            xmin, ymin, zmin, xmax, ymax, zmax,
            md.as_ref().map_or(0, |md| md.pos.len()),
        );
    }

    // 4) Also search for any materials whose lightmap_set_index maps to
    //    a real_index that resolves to texture slot 111.
    eprintln!("\n--- materials whose lm set includes slot 111 ---");
    for (mi, mat) in m.materials.iter().enumerate() {
        if mat.lightmap_stage() == 0 {
            continue;
        }
        // LM0..2 = lightmap_set_index + 0/1/2 in real-index space
        for offset in 0..=2u16 {
            let real = mat.lightmap_set_index as u16 + offset;
            if m.tex_slot(real as i16) == Some(111) {
                eprintln!(
                    "  mat[{mi}] id={} lm_set_real={} lm0+{offset} resolves to slot 111  diffuse=({:.3},{:.3},{:.3}) flags=0x{:x} vfbits=0x{:x}",
                    mat.id, mat.lightmap_set_index, mat.diffuse[0], mat.diffuse[1], mat.diffuse[2],
                    mat.texture_flags, mat.vertex_format_bits,
                );
            }
        }
    }

    // 5) Parts near the hologram coordinates (x~-26.7, z~-48.7)
    eprintln!("\n--- parts near hologram coords (x~-26.7 z~-48.7) ---");
    for (pi, part) in m.render_parts.iter().enumerate() {
        let mesh = &m.meshes[part.mesh];
        let md = match mapmesh::expand_mesh(&m, mesh) {
            Some(md) => md,
            None => continue,
        };
        if md.pos.is_empty() { continue; }
        let mut cx = 0.0f32;
        let mut cz = 0.0f32;
        for p in &md.pos {
            cx += p[0];
            cz += p[2];
        }
        let n = md.pos.len() as f32;
        cx /= n;
        cz /= n;
        if (cx - (-26.7)).abs() < 5.0 && (cz - (-48.7)).abs() < 5.0 {
            let mat = &m.materials[part.material];
            let lm_info = m.lightmaps.get(&part.lightmap).map(|ls| {
                format!("type={} tex={:?} off=({:.4},{:.4},{:.4},{:.4})", ls.ty, ls.tex, ls.off[0], ls.off[1], ls.off[2], ls.off[3])
            }).unwrap_or_else(|| "no lightmap".into());
            eprintln!(
                "  part[{pi}] mesh={} mat={} center=({:.2},{:.2},{:.2}) lm_key=0x{:x} [{}]",
                part.mesh, part.material, cx, md.pos[0][1], cz, part.lightmap, lm_info
            );
            eprintln!(
                "    mat: id={} tex_id={} diffuse=({:.3},{:.3},{:.3}) prelit={} lm_stage={} lighting={} spec=({:.3},{:.3},{:.3},{:.3})",
                mat.id, mat.tex_id, mat.diffuse[0], mat.diffuse[1], mat.diffuse[2],
                mat.shader_defines & 0x1000 != 0,
                mat.lightmap_stage(),
                mat.lighting_stage,
                mat.specular_params[0], mat.specular_params[1], mat.specular_params[2], mat.specular_params[3],
            );
        }
    }
}

/// Diagnostic: identify all parts likely to render "completely white" or
/// excessively bright, focusing on the table/bar-edge issue.
///
/// White rendering paths in the WGSL uber shader:
///   1. prelit=1 + has_lm=1 + v_lmuv.x<=0 → vertex-lit fallback:
///      diffuse = lm_diffuse = 1.0 (no lightmap darkening), color not
///      multiplied by baked vertex light → full brightness.
///   2. prelit=1 + has_lm=0 → color *= baked, diffuse = 1.0 → bright
///      if vertex color is high (byte 127 → ~0.5; still dimmed but
///      combined with ambient+specular can look white).
///   3. prelit=1 + has_lm=1 + v_lmuv.x>0 → diffuse = lightmap sample.
///      White only if the lightmap texel itself is white.
///   4. prelit=0 + lighting=DISABLE → diffuse = 1.0, full ambient.
///   5. PHONG specular at grazing angles (high kSpecular + wide power).
///
/// Run: cargo test --test diag_lm -- --nocapture
#[test]
fn diagnose_white_surfaces() {
    let dir = std::path::Path::new("backup/LEVELS/MAP/MAP");
    let gsc = dir.join("MAP_PC.GSC");
    if !gsc.exists() {
        eprintln!("skipping: backup assets not present");
        return;
    }
    let data = std::fs::read(&gsc).expect("read gsc");
    let m = map::parse(&data).expect("parse map");

    // Pre-expand all meshes once so we don't duplicate work.
    let expanded: Vec<Option<rustt::glb::MeshData>> = m
        .meshes
        .iter()
        .map(|mesh| mapmesh::expand_mesh(&m, mesh))
        .collect();

    eprintln!("=== WHITE SURFACE DIAGNOSTIC ===");
    eprintln!(
        "  {} parts, {} meshes, {} materials",
        m.render_parts.len(),
        m.meshes.len(),
        m.materials.len()
    );

    // ---- 1) Parts with lightmap override but lm_uv.x <= 0 on all verts ----
    // These are the most likely white-edge candidates: the material says
    // has_lm=1, so the shader skips baking vertex light into color, but
    // lm_uv.x<=0 skips the lightmap sample → diffuse=1.0.
    eprintln!("\n--- [1] Parts with lightmap binding but ALL verts have lm_uv.x<=0 (vertex-lit fallback) ---");
    for (pi, part) in m.render_parts.iter().enumerate() {
        let mat = &m.materials[part.material];
        let prelit = mat.shader_defines & 0x1000 != 0;
        let lm_stage = mat.lightmap_stage();
        if !prelit || lm_stage == 0 {
            continue;
        }
        let Some(mesh) = m.meshes.get(part.mesh) else { continue };
        let Some(ref md) = expanded[part.mesh] else { continue };
        if md.pos.is_empty() {
            continue;
        }
        if md.lm_uv.is_empty() {
            // No lightmap UVs at all → lm_uv is [0,0] from default → v_lmuv.x=0
            // This means the UV-set fallback didn't find ANY set.
            let has_lm_bind = part.lightmap != 0;
            let lm_page_ok = m.lightmaps.get(&part.lightmap).map(|ls| {
                let real = ls.tex[0];
                m.tex_slot(real as i16).is_some()
            }).unwrap_or(false);
            eprintln!(
                "  part[{pi}] mesh={} mat={} NO_LMUV verts={} prelit=1 lm_stage={lm_stage} has_lm_bind={has_lm_bind} lm_page_ok={lm_page_ok}",
                part.mesh, part.material, md.pos.len(),
            );
            continue;
        }
        let good = md.lm_uv.iter().filter(|u| u[0] > 0.0).count();
        let bad = md.lm_uv.len() - good;
        if bad > 0 && good == 0 {
            // ALL vertices have lm_uv.x<=0 → vertex-lit fallback for entire part
            let has_lm_bind = part.lightmap != 0;
            let lm_page_ok = m.lightmaps.get(&part.lightmap).map(|ls| {
                let real = ls.tex[0];
                m.tex_slot(real as i16).is_some()
            }).unwrap_or(false);
            let stride = mesh.vertex_size;
            let vf = mat.vertex_format_bits;
            let declared = mat.lightmap_uvset();
            eprintln!(
                "  part[{pi}] mesh={} mat={} ALL_BAD verts={} stride={stride} vf=0x{vf:x} lm_uvset={declared} has_lm_bind={has_lm_bind} lm_page_ok={lm_page_ok}",
                pi, part.material, md.pos.len(),
            );
        }
    }

    // ---- 2) Part count by "white surface" category ----
    eprintln!("\n--- [2] Render-part surface categories ---");
    let mut cat_prelit_lm = 0u32; // prelit + has_lm (normal lightmapped)
    let mut cat_prelit_lm_bad_uv = 0u32; // prelit + has_lm but ALL lm_uv.x<=0
    let mut cat_prelit_nolm = 0u32; // prelit + no lightmap at all
    let mut cat_unlit = 0u32; // lighting=DISABLE, not prelit
    let mut cat_phong = 0u32; // PHONG with live lights
    let mut cat_other = 0u32;
    for (pi, part) in m.render_parts.iter().enumerate() {
        let mat = &m.materials[part.material];
        let prelit = mat.shader_defines & 0x1000 != 0;
        let lm_stage = mat.lightmap_stage();
        let has_lm_bind = part.lightmap != 0;
        let lm_page_ok = m.lightmaps.get(&part.lightmap).map(|ls| {
            let real = ls.tex[0];
            m.tex_slot(real as i16).is_some()
        }).unwrap_or(false);
        let has_lm_effective = prelit && lm_stage != 0 && has_lm_bind && lm_page_ok;
        let Some(ref md) = expanded[part.mesh] else { continue };
        if md.pos.is_empty() {
            continue;
        }
        let all_bad_lmuv = has_lm_effective && !md.lm_uv.is_empty()
            && md.lm_uv.iter().all(|u| u[0] <= 0.0);

        if prelit && has_lm_effective && !all_bad_lmuv {
            cat_prelit_lm += 1;
        } else if prelit && has_lm_effective && all_bad_lmuv {
            cat_prelit_lm_bad_uv += 1;
        } else if prelit {
            cat_prelit_nolm += 1;
        } else if mat.lighting_stage == 0 {
            cat_unlit += 1;
        } else if mat.lighting_stage == 6 {
            cat_phong += 1;
        } else {
            cat_other += 1;
        }
    }
    eprintln!("  prelit+lm (normal):      {cat_prelit_lm}");
    eprintln!("  prelit+lm BAD_LMUV:      {cat_prelit_lm_bad_uv}");
    eprintln!("  prelit+no-lm (vertex):   {cat_prelit_nolm}");
    eprintln!("  unlit (lighting=0):      {cat_unlit}");
    eprintln!("  phong (lighting=6):      {cat_phong}");
    eprintln!("  other:                   {cat_other}");

    // ---- 3) Materials with prelit but no lightmap: vertex color analysis ----
    // These surfaces get diffuse = 1.0 (since prelit overrides to lm_diffuse=1.0)
    // and color *= baked. Check what baked looks like.
    eprintln!("\n--- [3] Prelit no-lm materials: vertex color analysis ---");
    let mut seen_mats = std::collections::HashSet::new();
    for (pi, part) in m.render_parts.iter().enumerate() {
        let mat = &m.materials[part.material];
        let prelit = mat.shader_defines & 0x1000 != 0;
        let lm_stage = mat.lightmap_stage();
        if !prelit || lm_stage != 0 {
            continue;
        }
        if !seen_mats.insert(part.material) {
            continue;
        }
        let Some(ref md) = expanded[part.mesh] else { continue };
        if md.pos.is_empty() {
            continue;
        }
        // Sample vertex colors
        let mut min_c = [255u8; 3];
        let mut max_c = [0u8; 3];
        let mut avg_c = [0.0f32; 3];
        for c in &md.color {
            for k in 0..3 {
                min_c[k] = min_c[k].min(c[k]);
                max_c[k] = max_c[k].max(c[k]);
                avg_c[k] += c[k] as f32;
            }
        }
        let n = md.color.len() as f32;
        for k in 0..3 {
            avg_c[k] /= n;
        }
        let has_tex = mat.tex_id >= 0 && m.tex_slot(mat.tex_id).is_some();
        eprintln!(
            "  mat[{}] id={} tex_id={} has_tex={has_tex} diffuse=({:.3},{:.3},{:.3}) lighting={} spec=({:.3},{:.3},{:.3},{:.3}) vcol=avg({:.0},{:.0},{:.0}) min({},{},{}) max({},{},{})",
            part.material,
            mat.id,
            mat.tex_id,
            mat.diffuse[0], mat.diffuse[1], mat.diffuse[2],
            mat.lighting_stage,
            mat.specular_params[0], mat.specular_params[1], mat.specular_params[2], mat.specular_params[3],
            avg_c[0], avg_c[1], avg_c[2],
            min_c[0], min_c[1], min_c[2],
            max_c[0], max_c[1], max_c[2],
        );
    }

    // ---- 4) Prelit materials with has_lm: check what lightmap page resolves to
    //    and whether the lm_uv range makes sense ----
    eprintln!("\n--- [4] Lightmapped materials: lm_uv range + page info ---");
    let mut seen_mats2 = std::collections::HashSet::new();
    for (pi, part) in m.render_parts.iter().enumerate() {
        let mat = &m.materials[part.material];
        let prelit = mat.shader_defines & 0x1000 != 0;
        let lm_stage = mat.lightmap_stage();
        if !prelit || lm_stage == 0 {
            continue;
        }
        if !seen_mats2.insert(part.material) {
            continue;
        }
        let Some(ref md) = expanded[part.mesh] else { continue };
        if md.pos.is_empty() || md.lm_uv.is_empty() {
            continue;
        }
        let mut min_u = f32::MAX;
        let mut max_u = f32::MIN;
        let mut min_v = f32::MAX;
        let mut max_v = f32::MIN;
        let mut count_zero = 0u32;
        let mut count_neg = 0u32;
        let mut count_pos = 0u32;
        for uv in &md.lm_uv {
            min_u = min_u.min(uv[0]);
            max_u = max_u.max(uv[0]);
            min_v = min_v.min(uv[1]);
            max_v = max_v.max(uv[1]);
            if uv[0] <= 0.0 {
                count_zero += 1;
            }
            if uv[1] < 0.0 {
                count_neg += 1;
            } else {
                count_pos += 1;
            }
        }
        let total = md.lm_uv.len() as u32;
        let lm_set = mat.lightmap_set_index;
        let lm_uvset = mat.lightmap_uvset();
        let vfbits = mat.vertex_format_bits;
        let stride = m.meshes[part.mesh].vertex_size;
        // Check what lightmap textures would resolve
        let lm0_real = lm_set;
        let lm0_slot = m.tex_slot(lm0_real as i16);
        let lm_has_bind = part.lightmap != 0;
        let lm_info = m.lightmaps.get(&part.lightmap).map(|ls| {
            format!("ty={} tex={:?} off=({:.4},{:.4},{:.4},{:.4})", ls.ty, ls.tex, ls.off[0], ls.off[1], ls.off[2], ls.off[3])
        }).unwrap_or_else(|| "none".into());
        eprintln!(
            "  mat[{}] id={} stride={stride} vf=0x{vfbits:x} lm_set={lm_set} lm_uvset={lm_uvset} lm_has_bind={lm_has_bind} lm0_slot={lm0_slot:?} lm_info=[{lm_info}]",
            part.material, mat.id,
        );
        eprintln!(
            "    lm_uv: u=[{min_u:.4}..{max_u:.4}] v=[{min_v:.4}..{max_v:.4}] zero_u={count_zero}/{total} neg_v={count_neg}/{total}"
        );
    }

    // ---- 5) Parts with PHONG lighting but NO prelit: these get live lights ----
    // Non-prelit PHONG surfaces accumulate ambient + directional light into
    // diffuse. With UBBER_AMBIENT=(0.1,0.1,0.1) and scene_ambient from RTL,
    // total ambient ≈ 0.2. Three directional lights with intensity can push
    // diffuse well above 1.0 if their directions align.
    eprintln!("\n--- [5] Non-prelit PHONG parts (live lighting, possible over-bright) ---");
    let mut count_phong_nonprelit = 0u32;
    for (pi, part) in m.render_parts.iter().enumerate() {
        let mat = &m.materials[part.material];
        let prelit = mat.shader_defines & 0x1000 != 0;
        if prelit || mat.lighting_stage != 6 {
            continue;
        }
        count_phong_nonprelit += 1;
        if count_phong_nonprelit > 20 {
            continue; // cap output
        }
        let Some(ref md) = expanded[part.mesh] else { continue };
        let (mut cx, mut cy, mut cz) = (0.0f32, 0.0f32, 0.0f32);
        for p in &md.pos {
            cx += p[0];
            cy += p[1];
            cz += p[2];
        }
        let n = md.pos.len() as f32;
        cx /= n;
        cy /= n;
        cz /= n;
        eprintln!(
            "  part[{pi}] mesh={} mat={} center=({cx:.2},{cy:.2},{cz:.2}) diffuse=({:.3},{:.3},{:.3}) spec=({:.3},{:.3},{:.3},{:.3}) tex_id={}",
            part.mesh, part.material,
            mat.diffuse[0], mat.diffuse[1], mat.diffuse[2],
            mat.specular_params[0], mat.specular_params[1], mat.specular_params[2], mat.specular_params[3],
            mat.tex_id,
        );
    }
    eprintln!("  total non-prelit PHONG parts: {count_phong_nonprelit}");

    // ---- 6) Parts with very high specular kSpecular ----
    eprintln!("\n--- [6] Materials with high specular (kSpecular > 0.5) ---");
    let mut seen_mats3 = std::collections::HashSet::new();
    for (pi, part) in m.render_parts.iter().enumerate() {
        let mat = &m.materials[part.material];
        if !seen_mats3.insert(part.material) {
            continue;
        }
        if mat.specular_params[1] > 0.5 {
            let prelit = mat.shader_defines & 0x1000 != 0;
            let lm_stage = mat.lightmap_stage();
            let has_tex = mat.tex_id >= 0 && m.tex_slot(mat.tex_id).is_some();
            eprintln!(
                "  mat[{}] id={} prelit={} lm_stage={} lighting={} diffuse=({:.3},{:.3},{:.3}) kSpec={:.3} kPow={:.1} kFres={:.3} kFresP={:.3} has_tex={has_tex} tex_id={}",
                part.material, mat.id, prelit, lm_stage, mat.lighting_stage,
                mat.diffuse[0], mat.diffuse[1], mat.diffuse[2],
                mat.specular_params[1], mat.specular_params[0],
                mat.specular_params[2], mat.specular_params[3],
                mat.tex_id,
            );
        }
    }

    // ---- 7) Dump the map's center area (likely cantina bar/tables) ----
    eprintln!("\n--- [7] Parts in center of map (|x|<10, |z|<10, y>−3) ---");
    for (pi, part) in m.render_parts.iter().enumerate() {
        let mat = &m.materials[part.material];
        let Some(ref md) = expanded[part.mesh] else { continue };
        if md.pos.is_empty() { continue; }
        let (mut cx, mut cy, mut cz) = (0.0f32, 0.0f32, 0.0f32);
        for p in &md.pos {
            cx += p[0]; cy += p[1]; cz += p[2];
        }
        let n = md.pos.len() as f32;
        cx /= n; cy /= n; cz /= n;
        if cx.abs() < 10.0 && cz.abs() < 10.0 && cy > -3.0 {
            let prelit = mat.shader_defines & 0x1000 != 0;
            let lm_stage = mat.lightmap_stage();
            let has_lm_bind = part.lightmap != 0;
            let lm_uv_good = md.lm_uv.iter().filter(|u| u[0] > 0.0).count();
            let lm_uv_bad = md.lm_uv.len() - lm_uv_good;
            let has_tex = mat.tex_id >= 0 && m.tex_slot(mat.tex_id).is_some();
            eprintln!(
                "  part[{pi}] mesh={} mat={} center=({cx:.2},{cy:.2},{cz:.2}) verts={} prelit={prelit} lm_stage={lm_stage} has_lm_bind={has_lm_bind} lm_uv(good/bad)={lm_uv_good}/{lm_uv_bad} tex={has_tex} diffuse=({:.3},{:.3},{:.3}) lighting={}",
                part.mesh, part.material, md.pos.len(),
                mat.diffuse[0], mat.diffuse[1], mat.diffuse[2],
                mat.lighting_stage,
            );
        }
    }
}

/// Identify cantina table/bar materials: group center-of-map parts by material,
/// show vertex bounding box (to distinguish rings from discs), and dump
/// first 3 vertex positions per part.
///
/// Run: cargo test --test diag_lm diagnose_table_materials -- --nocapture
#[test]
fn diagnose_table_materials() {
    let dir = std::path::Path::new("backup/LEVELS/MAP/MAP");
    let gsc = dir.join("MAP_PC.GSC");
    if !gsc.exists() {
        eprintln!("skipping: backup assets not present");
        return;
    }
    let data = std::fs::read(&gsc).expect("read gsc");
    let m = map::parse(&data).expect("parse map");
    let expanded: Vec<Option<rustt::glb::MeshData>> = m
        .meshes
        .iter()
        .map(|mesh| mapmesh::expand_mesh(&m, mesh))
        .collect();

    // Collect center-of-map parts, grouped by material.
    #[derive(Default)]
    struct MatInfo {
        prelit: bool,
        lm_stage: u8,
        lighting: u8,
        diffuse: [f32; 3],
        spec: [f32; 4],
        has_tex: bool,
        tex_id: i16,
        parts: Vec<(usize, f32, f32, f32, usize, bool, [f32; 3], [f32; 3])>,
        // (part_idx, cx, cy, cz, verts, lm_uv_ok, vmin, vmax)
    }
    let mut by_mat: BTreeMap<usize, MatInfo> = BTreeMap::new();

    for (pi, part) in m.render_parts.iter().enumerate() {
        let mat = &m.materials[part.material];
        let Some(ref md) = expanded[part.mesh] else { continue };
        if md.pos.is_empty() { continue; }
        let (mut cx, mut cy, mut cz) = (0.0f32, 0.0f32, 0.0f32);
        let mut vmin = [f32::MAX; 3];
        let mut vmax = [f32::MIN; 3];
        for p in &md.pos {
            cx += p[0]; cy += p[1]; cz += p[2];
            for k in 0..3 {
                vmin[k] = vmin[k].min(p[k]);
                vmax[k] = vmax[k].max(p[k]);
            }
        }
        let n = md.pos.len() as f32;
        cx /= n; cy /= n; cz /= n;
        if cx.abs() < 8.0 && cz.abs() < 8.0 && cy > -2.0 && cy < 8.0 {
            let prelit = mat.shader_defines & 0x1000 != 0;
            let lm_uv_ok = md.lm_uv.iter().filter(|u| u[0] > 0.0).count();
            let has_tex = mat.tex_id >= 0 && m.tex_slot(mat.tex_id).is_some();
            let e = by_mat.entry(part.material).or_insert_with(|| MatInfo {
                prelit,
                lm_stage: mat.lightmap_stage(),
                lighting: mat.lighting_stage,
                diffuse: [mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]],
                spec: mat.specular_params,
                has_tex,
                tex_id: mat.tex_id,
                parts: Vec::new(),
            });
            e.parts.push((pi, cx, cy, cz, md.pos.len(), lm_uv_ok > 0, vmin, vmax));
        }
    }

    eprintln!("=== TABLE/BAR MATERIAL DIAGNOSTIC ===");
    eprintln!("  {} materials in center area\n", by_mat.len());

    for (&mi, info) in &by_mat {
        let span_x = if info.parts.iter().all(|p| p.7[0] > f32::MIN) {
            let xmin = info.parts.iter().map(|p| p.6[0]).fold(f32::MAX, f32::min);
            let xmax = info.parts.iter().map(|p| p.7[0]).fold(f32::MIN, f32::max);
            xmax - xmin
        } else { 0.0 };
        let span_z = if info.parts.iter().all(|p| p.7[2] > f32::MIN) {
            let zmin = info.parts.iter().map(|p| p.6[2]).fold(f32::MAX, f32::min);
            let zmax = info.parts.iter().map(|p| p.7[2]).fold(f32::MIN, f32::max);
            zmax - zmin
        } else { 0.0 };
        eprintln!(
            "mat[{mi}] prelit={} lm_stage={} lighting={} diffuse=({:.3},{:.3},{:.3}) spec=({:.3},{:.3},{:.3},{:.3}) tex_id={} has_tex={} parts={} span=({:.2}x{:.2})",
            info.prelit, info.lm_stage, info.lighting,
            info.diffuse[0], info.diffuse[1], info.diffuse[2],
            info.spec[0], info.spec[1], info.spec[2], info.spec[3],
            info.tex_id, info.has_tex, info.parts.len(), span_x, span_z,
        );
        // Show each part: center, verts, first 3 vertex positions.
        for &(pi, cx, cy, cz, verts, lm_ok, ref vmin, ref vmax) in &info.parts {
            let md = expanded[m.render_parts[pi].mesh].as_ref().unwrap();
            let pos_sample: Vec<String> = md.pos.iter().take(3)
                .map(|p| format!("({:.2},{:.2},{:.2})", p[0], p[1], p[2]))
                .collect();
            eprintln!(
                "  part[{pi}] mesh={} c=({cx:.2},{cy:.2},{cz:.2}) verts={verts} lm_ok={lm_ok} bbox=({:.2},{:.2},{:.2})-({:.2},{:.2},{:.2}) pos=[{}]",
                m.render_parts[pi].mesh,
                vmin[0], vmin[1], vmin[2], vmax[0], vmax[1], vmax[2],
                pos_sample.join(", "),
            );
        }
        eprintln!();
    }
}