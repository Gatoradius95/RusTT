use std::path::Path;

use anyhow::{Context, Result};
use png::BitDepth;

use rustt::ghg::Parsed;

fn save_png(path: &str, w: usize, h: usize, rgba: &[u8]) -> Result<()> {
    let mut enc = png::Encoder::new(std::fs::File::create(path)?, w as u32, h as u32);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(BitDepth::Eight);
    let mut wr = enc.write_header().context("png header")?;
    wr.write_image_data(rgba).context("png data")?;
    Ok(())
}

fn ref565(v: u16) -> [u8; 3] {
    let r = ((v >> 11) & 0x1f) as u32;
    let g = ((v >> 5) & 0x3f) as u32;
    let b = (v & 0x1f) as u32;
    [
        ((r * 255 + 15) / 31) as u8,
        ((g * 255 + 31) / 63) as u8,
        ((b * 255 + 15) / 31) as u8,
    ]
}

fn ref_lerp(a: u32, b: u32, num: u32, den: u32) -> u32 {
    (a * (den - num) + b * num) / den
}

fn reference_bc3(payload: &[u8], w: usize, h: usize, out: &mut [u8]) {
    let bw = (w + 3) / 4;
    let bh = (h + 3) / 4;
    for by in 0..bh {
        for bx in 0..bw {
            let bo = (by * bw + bx) * 16;
            let a0 = payload[bo] as u32;
            let a1 = payload[bo + 1] as u32;
            let mut alphas = [0u32; 8];
            if a0 > a1 {
                alphas[0] = a0;
                alphas[1] = a1;
                for i in 2u32..8 {
                    alphas[i as usize] = (a0 * (8 - i) + a1 * (i - 1)) / 7;
                }
            } else {
                alphas[0] = a0;
                alphas[1] = a1;
                for i in 2u32..6 {
                    alphas[i as usize] = (a0 * (6 - i) + a1 * (i - 1)) / 5;
                }
                alphas[6] = 0;
                alphas[7] = 255;
            }
            let c0 = u16::from_le_bytes(payload[bo + 8..bo + 10].try_into().unwrap());
            let c1 = u16::from_le_bytes(payload[bo + 10..bo + 12].try_into().unwrap());
            let rc0 = ref565(c0);
            let rc1 = ref565(c1);
            let colors = [
                rc0,
                rc1,
                [
                    ref_lerp(rc0[0] as u32, rc1[0] as u32, 1, 3) as u8,
                    ref_lerp(rc0[1] as u32, rc1[1] as u32, 1, 3) as u8,
                    ref_lerp(rc0[2] as u32, rc1[2] as u32, 1, 3) as u8,
                ],
                [
                    ref_lerp(rc0[0] as u32, rc1[0] as u32, 2, 3) as u8,
                    ref_lerp(rc0[1] as u32, rc1[1] as u32, 2, 3) as u8,
                    ref_lerp(rc0[2] as u32, rc1[2] as u32, 2, 3) as u8,
                ],
            ];
            let mut ab = [0u8; 8];
            ab[..6].copy_from_slice(&payload[bo + 2..bo + 8]);
            let abits = u64::from_le_bytes(ab);
            let bits = u32::from_le_bytes(payload[bo + 12..bo + 16].try_into().unwrap());
            for py in 0..4usize {
                for px in 0..4usize {
                    let ci = (bits >> (2 * (py * 4 + px))) & 3;
                    let ai = ((abits >> (3 * (py * 4 + px))) & 7) as usize;
                    let x = bx * 4 + px;
                    let y = by * 4 + py;
                    if x < w && y < h {
                        let o = (y * w + x) * 4;
                        out[o] = colors[ci as usize][0];
                        out[o + 1] = colors[ci as usize][1];
                        out[o + 2] = colors[ci as usize][2];
                        out[o + 3] = alphas[ai] as u8;
                    }
                }
            }
        }
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if let Some(s) = args.iter().position(|a| a == "--scan") {
        let dir = args.get(s + 1).expect("--scan needs a dir");
        let mut files: Vec<_> = Vec::new();
        fn collect(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
            let Ok(rd) = std::fs::read_dir(dir) else { return };
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    collect(&p, out);
                } else if p.extension().map(|e| e.to_string_lossy().to_uppercase()) == Some("GHG".into()) {
                    out.push(p);
                }
            }
        }
        collect(Path::new(dir), &mut files);
        let mut with_shapes = 0usize;
        for p in &files {
            let Ok(data) = std::fs::read(&p) else { continue };
            let Ok(parsed) = rustt::ghg::parse(&data) else { continue };
            let name = p.file_stem().unwrap().to_string_lossy().to_string();
            let shape_parts: Vec<String> = parsed
                .parts
                .iter()
                .enumerate()
                .filter(|(_, pt)| !pt.dynamic_buffers.is_empty())
                .map(|(i, pt)| {
                    let filled = pt.dynamic_buffers.iter().filter(|b| b.is_some()).count();
                    format!("p{i}:{}slots/{}filled", pt.dynamic_buffers.len(), filled)
                })
                .collect();
            if !shape_parts.is_empty() {
                with_shapes += 1;
                println!("{name}: SHAPES {}", shape_parts.join(" "));
            }
            for (i, t) in parsed.textures.iter().enumerate() {
                if t.fmt == rustt::ghg::TextureFmt::Dxt5 {
                    let Ok(rgba) = rustt::dxt::decode(t) else { continue };
                    let mut h = [0u32; 8];
                    for px in rgba.chunks_exact(4) {
                        let a = px[3];
                        if a < 32 {
                            h[0] += 1;
                        } else if a < 64 {
                            h[1] += 1;
                        } else if a < 128 {
                            h[2] += 1;
                        } else if a < 192 {
                            h[3] += 1;
                        } else if a < 224 {
                            h[4] += 1;
                        } else if a < 248 {
                            h[5] += 1;
                        } else if a < 255 {
                            h[6] += 1;
                        } else {
                            h[7] += 1;
                        }
                    }
                    let n = (rgba.len() / 4) as f64;
                    let pct: Vec<f64> = h.iter().map(|c| *c as f64 * 100.0 / n).collect();
                    println!(
                        "{name} tex{i} {w}x{h} a:<32 {p0:.0}% 32-63 {p1:.0}% 64-127 {p2:.0}% 128-191 {p3:.0}% 192-223 {p4:.0}% 224-247 {p5:.0}% 248-254 {p6:.0}% 255 {p7:.0}%",
                        name = name, i = i, w = t.w, h = t.h,
                        p0 = pct[0], p1 = pct[1], p2 = pct[2], p3 = pct[3],
                        p4 = pct[4], p5 = pct[5], p6 = pct[6], p7 = pct[7]
                    );
                }
            }
        }
        println!("--scan done, {} files scanned, {} with shape keys", files.len(), with_shapes);
        return Ok(());
    }

    let path = args.get(1).expect("usage: dump <model.ghg>");
    let data = std::fs::read(path).with_context(|| format!("reading {}", path))?;
    let parsed = rustt::ghg::parse(&data).with_context(|| format!("parsing {}", path))?;

    if let Some(a) = args.iter().position(|x| x == "--mesh") {
        let idx = args.get(a + 1).and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
        let md = rustt::glb::build_mesh(&parsed, idx);
        println!("part {idx}: {} verts, {} tris", md.pos.len(), md.idx.len() / 3);
        let mut mn = [f32::INFINITY; 3];
        let mut mx = [f32::NEG_INFINITY; 3];
        for (i, p) in md.pos.iter().enumerate() {
            for k in 0..3 {
                mn[k] = mn[k].min(p[k]);
                mx[k] = mx[k].max(p[k]);
            }
            if md.pos.len() <= 256 {
                println!("  v{i}: pos=({:.4},{:.4},{:.4}) uv=({:.3},{:.3})",
                    p[0], p[1], p[2], md.uv[i][0], md.uv[i][1]);
            }
        }
        println!("  bbox x:[{:.4},{:.4}] y:[{:.4},{:.4}] z:[{:.4},{:.4}]",
            mn[0], mx[0], mn[1], mx[1], mn[2], mx[2]);
        return Ok(());
    }

    if let Some(_) = args.iter().position(|x| x == "--shapes") {
        println!("=== shape keys (dynamic buffers) ===");
        for (i, p) in parsed.parts.iter().enumerate() {
            if p.dynamic_buffers.is_empty() {
                continue;
            }
            println!(
                "part {i}: stride={} verts={} slots={} filled={}",
                p.stride,
                p.num_v,
                p.dynamic_buffers.len(),
                p.dynamic_buffers.iter().filter(|b| b.is_some()).count()
            );
            for (j, buf) in p.dynamic_buffers.iter().enumerate() {
                match buf {
                    None => println!("  slot {j}: empty"),
                    Some(d) => {
                        let (mut mn, mut mx) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
                        let mut max_mag = 0.0f32;
                        let mut nonzero = 0usize;
                        for v in d {
                            let mag = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                            if mag > max_mag {
                                max_mag = mag;
                            }
                            if mag > 1e-6 {
                                nonzero += 1;
                            }
                            for k in 0..3 {
                                if v[k] < mn[k] { mn[k] = v[k]; }
                                if v[k] > mx[k] { mx[k] = v[k]; }
                            }
                        }
                        println!(
                            "  slot {j}: {}x3 min=({:.4},{:.4},{:.4}) max=({:.4},{:.4},{:.4}) max_mag={:.4} nonzero={nonzero}/{}",
                            d.len(),
                            mn[0], mn[1], mn[2], mx[0], mx[1], mx[2],
                            max_mag,
                            d.len()
                        );
                    }
                }
            }
        }
        return Ok(());
    }

    if let Some(a) = args.iter().position(|x| x == "--shapesobj") {
        let outdir = args.get(a + 1).expect("--shapesobj needs a dir");
        std::fs::create_dir_all(outdir)?;
        let stem = Path::new(path)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "model".into());
        for (i, p) in parsed.parts.iter().enumerate() {
            if p.dynamic_buffers.is_empty() {
                continue;
            }
            let md = rustt::glb::build_mesh(&parsed, i);
            if md.pos.len() != p.num_v {
                println!("part {i}: base mesh has {} verts, part expects {}; skipping", md.pos.len(), p.num_v);
                continue;
            }
            // OBJ: base pose, then each filled shape (base + offset).
            let mut out = String::new();
            out.push_str(&format!("# part {i}: {} verts, {} slots\n", p.num_v, p.dynamic_buffers.len()));
            for v in &md.pos {
                out.push_str(&format!("v {:.6} {:.6} {:.6}\n", v[0], v[1], v[2]));
            }
            for (j, buf) in p.dynamic_buffers.iter().enumerate() {
                let Some(d) = buf else { continue };
                out.push_str(&format!("# shape {j}\n"));
                for (k, v) in d.iter().enumerate() {
                    let b = md.pos.get(k).copied().unwrap_or([0.0; 3]);
                    out.push_str(&format!(
                        "v {:.6} {:.6} {:.6}\n",
                        b[0] + v[0],
                        b[1] + v[1],
                        b[2] + v[2]
                    ));
                }
            }
            for c in md.idx.chunks(3) {
                if c.len() != 3 {
                    continue;
                }
                out.push_str(&format!("f {} {} {}\n", c[0] + 1, c[1] + 1, c[2] + 1));
            }
            let p = format!("{outdir}/{stem}_part{i}.obj");
            std::fs::write(&p, out)?;
            println!("wrote {p}");
        }
        return Ok(());
    }

    if let Some(dir) = args.iter().position(|a| a == "--texpng") {
        let outdir = args.get(dir + 1).expect("--texpng needs a dir");
        std::fs::create_dir_all(outdir)?;
        for (i, t) in parsed.textures.iter().enumerate() {
            let rgba = rustt::dxt::decode(t)?;
            let p = format!("{}/tex_{}.png", outdir, i);
            save_png(&p, t.w, t.h, &rgba)?;
            println!("wrote {p}");
        }
        return Ok(());
    }

    if let Some(a) = args.iter().position(|x| x == "--verifydxt5") {
        let idx = args.get(a + 1).and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
        let t = parsed.textures.get(idx).context("texture index out of range")?;
        if t.fmt != rustt::ghg::TextureFmt::Dxt5 {
            println!("not a DXT5 texture");
            return Ok(());
        }
        let ours = rustt::dxt::decode(t)?;
        let mut refr = vec![0u8; t.w * t.h * 4];
        reference_bc3(&t.payload, t.w, t.h, &mut refr);
        let mut diffs = 0u32;
        let mut first = None;
        for (i, (a, b)) in ours.iter().zip(refr.iter()).enumerate() {
            if a != b {
                diffs += 1;
                if first.is_none() {
                    first = Some(i);
                }
            }
        }
        println!(
            "decode diff vs reference BC3: {} differing bytes / {} total (first at {})",
            diffs,
            ours.len(),
            first.map(|i| format!("byte {i}")).unwrap_or_else(|| "none".into())
        );
        if let Some(i) = first {
            let px = i / 4;
            let (x, y) = (px % t.w, px / t.w);
            let ch = i % 4;
            println!(
                "first mismatch at pixel ({x},{y}) channel {ch}: ours={} ref={}",
                ours[i], refr[i]
            );
        }
        return Ok(());
    }

    if let Some(a) = args.iter().position(|x| x == "--ascii") {
        let idx = args.get(a + 1).and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
        let t = parsed.textures.get(idx).context("texture index out of range")?;
        let rgba = rustt::dxt::decode(t)?;
        let w = t.w;
        let h = t.h;
        let cols = 128;
        let rows = (h as f32 * cols as f32 / w as f32) as usize;
        for ry in 0..rows {
            let mut line = String::new();
            for rx in 0..cols {
                let x0 = rx * w / cols;
                let x1 = ((rx + 1) * w + cols - 1) / cols;
                let y0 = ry * h / rows;
                let y1 = ((ry + 1) * h + rows - 1) / rows;
                let mut sa = 0u32;
                let mut sl = 0u32;
                let mut n = 0u32;
                for y in y0..y1 {
                    for x in x0..x1 {
                        let o = (y * w + x) * 4;
                        sa += rgba[o + 3] as u32;
                        sl += (rgba[o] as u32 + rgba[o + 1] as u32 + rgba[o + 2] as u32) / 3;
                        n += 1;
                    }
                }
                let a = sa / n.max(1);
                let l = sl / n.max(1);
                let c = if a < 24 {
                    ' '
                } else if a < 200 {
                    '+'
                } else if l < 100 {
                    '#'
                } else {
                    'o'
                };
                line.push(c);
            }
            println!("{line}");
        }
        return Ok(());
    }

    println!("file: {}", Path::new(path).file_name().unwrap().to_string_lossy());
    println!("parts: {}  render items: {}  materials: {}  textures: {}  bones: {}",
        parsed.parts.len(), parsed.render.len(), parsed.materials.len(),
        parsed.textures.len(), parsed.bones.len());
    println!();

    println!("=== render items ===");
    for (i, r) in parsed.render.iter().enumerate() {
        let layer = parsed.render_layer.get(i).copied().unwrap_or(u32::MAX);
        println!("item {i}: part={} mat={} bone={} layer={layer}", r.part, r.mat, r.bone);
    }
    println!();

    println!("=== parts (raw, untransformed) ===");
    for (i, p) in parsed.parts.iter().enumerate() {
        let md = rustt::glb::build_mesh(&parsed, i);
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        let mut centroid = [0.0f32; 3];
        let mut hash: u64 = 0;
        for (vi, v) in md.pos.iter().enumerate() {
            for k in 0..3 {
                if v[k] < min[k] { min[k] = v[k]; }
                if v[k] > max[k] { max[k] = v[k]; }
                centroid[k] += v[k];
            }
            hash = hash.wrapping_mul(31).wrapping_add(f32::to_bits(md.pos[vi][0]) as u64 ^ f32::to_bits(md.pos[vi][1]) as u64 ^ f32::to_bits(md.pos[vi][2]) as u64);
        }
        if !md.pos.is_empty() {
            for k in 0..3 { centroid[k] /= md.pos.len() as f32; }
        }
        let mut outward = 0usize;
        let mut total = 0usize;
        for c in md.idx.chunks(3) {
            let a = c[0] as usize;
            let b = c[1] as usize;
            let d = c[2] as usize;
            if a >= md.pos.len() || b >= md.pos.len() || d >= md.pos.len() { continue; }
            let ab = [md.pos[b][0]-md.pos[a][0], md.pos[b][1]-md.pos[a][1], md.pos[b][2]-md.pos[a][2]];
            let ac = [md.pos[d][0]-md.pos[a][0], md.pos[d][1]-md.pos[a][1], md.pos[d][2]-md.pos[a][2]];
            let n = [ab[1]*ac[2]-ab[2]*ac[1], ab[2]*ac[0]-ab[0]*ac[2], ab[0]*ac[1]-ab[1]*ac[0]];
            let tc = [(md.pos[a][0]+md.pos[b][0]+md.pos[d][0])/3.0 - centroid[0],
                      (md.pos[a][1]+md.pos[b][1]+md.pos[d][1])/3.0 - centroid[1],
                      (md.pos[a][2]+md.pos[b][2]+md.pos[d][2])/3.0 - centroid[2]];
            let dot = n[0]*tc[0] + n[1]*tc[1] + n[2]*tc[2];
            if dot > 0.0 { outward += 1; }
            total += 1;
        }
        println!(
            "part {i}: stride={} off_v={} num_v={} off_i={} num_i={} il={} vl={} verts={} tris={} hash={hash:016x} \
             outward={outward}/{total} \
             bbox min=({:.2},{:.2},{:.2}) max=({:.2},{:.2},{:.2}) c=({:.2},{:.2},{:.2})",
            p.stride, p.off_v, p.num_v, p.off_i, p.num_i, p.il, p.vl, md.pos.len(), md.idx.len() / 3,
            min[0], min[1], min[2], max[0], max[1], max[2],
            centroid[0], centroid[1], centroid[2]
        );
    }
    println!();

    println!("=== render items transformed by bone world ===");
    for (i, r) in parsed.render.iter().enumerate() {
        let md = rustt::glb::build_mesh(&parsed, r.part);
        if md.pos.is_empty() {
            continue;
        }
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        let m = if r.bone >= 0 {
            parsed.bones.get(r.bone as usize).map(|b| b.world)
        } else {
            None
        };
        let mut hash: u64 = 0;
        for v in &md.pos {
            let p = match m {
                Some(mat) => mat.transform_point3(glam::Vec3::from(*v)).to_array(),
                None => *v,
            };
            for k in 0..3 {
                if p[k] < min[k] { min[k] = p[k]; }
                if p[k] > max[k] { max[k] = p[k]; }
            }
            hash = hash.wrapping_mul(31).wrapping_add(f32::to_bits(p[0]) as u64 ^ f32::to_bits(p[1]) as u64 ^ f32::to_bits(p[2]) as u64);
        }
        println!(
            "item {i}: part={} bone={} moved={} bbox=({:.2},{:.2},{:.2})-({:.2},{:.2},{:.2}) hash={hash:016x}",
            r.part, r.bone, m.is_some(),
            min[0], min[1], min[2], max[0], max[1], max[2]
        );
    }

    println!("=== atlas sampling (what each item would actually sample) ===");
    let tex_rgba: Vec<Option<Vec<u8>>> = parsed
        .textures
        .iter()
        .map(|t| rustt::dxt::decode(t).ok())
        .collect();
    for (i, r) in parsed.render.iter().enumerate() {
        let md = rustt::glb::build_mesh(&parsed, r.part);
        if md.pos.is_empty() || md.uv.is_empty() {
            println!("item {i}: part={} no mesh", r.part);
            continue;
        }
        let tid = parsed.materials.get(r.mat as usize).map(|m| m.tex_id as usize);
        let rgba = tid.and_then(|t| tex_rgba.get(t).cloned().flatten());
        let (tw, th) = tid
            .and_then(|t| parsed.textures.get(t).map(|tx| (tx.w, tx.h)))
            .unwrap_or((1, 1));
        let mut sum = [0u64; 4];
        let mut tris = 0u64;
        let mut out_of_range = 0u64;
        let mut finite = 0u64;
        let mut uv_min = [f32::INFINITY; 2];
        let mut uv_max = [f32::NEG_INFINITY; 2];
        for c in md.idx.chunks(3) {
            let a = c[0] as usize;
            let b = c[1] as usize;
            let d = c[2] as usize;
            if a >= md.uv.len() || b >= md.uv.len() || d >= md.uv.len() {
                continue;
            }
            tris += 1;
            let uv = [
                (md.uv[a][0] + md.uv[b][0] + md.uv[d][0]) / 3.0,
                (md.uv[a][1] + md.uv[b][1] + md.uv[d][1]) / 3.0,
            ];
            for k in 0..2 {
                if uv[k] < uv_min[k] { uv_min[k] = uv[k]; }
                if uv[k] > uv_max[k] { uv_max[k] = uv[k]; }
            }
            if uv[0].is_finite() && uv[1].is_finite() {
                finite += 1;
            }
            if !(0.0..=1.0).contains(&uv[0]) || !(0.0..=1.0).contains(&uv[1]) {
                out_of_range += 1;
            }
            if let Some(px) = &rgba {
                if uv[0].is_finite() && uv[1].is_finite() {
                    let x = (((uv[0] * tw as f32).floor() as i64).rem_euclid(tw as i64)) as usize;
                    let y = (((uv[1] * th as f32).floor() as i64).rem_euclid(th as i64)) as usize;
                    let o = (y * tw + x) * 4;
                    for k in 0..4 {
                        sum[k] += px[o + k] as u64;
                    }
                }
            }
        }
        for v in &md.uv {
            for k in 0..2 {
                if v[k] < uv_min[k] { uv_min[k] = v[k]; }
                if v[k] > uv_max[k] { uv_max[k] = v[k]; }
            }
        }
        let t = tid.map(|v| v.to_string()).unwrap_or_else(|| "-".into());
        let s = if let Some(px) = &rgba {
            let n = if tris > 0 { tris } else { 1 };
            format!(
                "samples rgb=({:.0},{:.0},{:.0},{:.0})",
                sum[0] as f64 / n as f64,
                sum[1] as f64 / n as f64,
                sum[2] as f64 / n as f64,
                sum[3] as f64 / n as f64
            )
        } else {
            "no texture".into()
        };
        println!(
            "item {i}: part={} mat={} tex={t} {tw}x{th} stride={} {s} uv=[{:.3},{:.3}]-[{:.3},{:.3}] oob={out_of_range}/{tris} finite={finite}/{tris}",
            r.part, r.mat, parsed.parts[r.part].stride, uv_min[0], uv_min[1], uv_max[0], uv_max[1]
        );
    }
    println!();

    println!("=== raw vertex floats (v0 then v1) ===");
    for (pi, p) in parsed.parts.iter().enumerate() {
        if pi > 9 {
            break;
        }
        let vl = parsed.vertex_lists[p.vl];
        let base = p.off_v * p.stride;
        for v in 0..2usize.min(p.num_v) {
            let o = base + v * p.stride;
            let mut line = format!("part {pi} v{v}:");
            for f in 0..p.stride / 4 {
                let b = &vl[o + f * 4..o + f * 4 + 4];
                line.push_str(&format!(" [{f}]={:.3}", f32::from_le_bytes(b.try_into().unwrap())));
            }
            println!("{line}");
        }
    }
    println!();

    println!("=== materials ===");
    for (i, m) in parsed.materials.iter().enumerate() {
        println!(
            "mat {i}: id={} tex={} rgba={:02x}{:02x}{:02x}{:02x} diff=({:.2},{:.2},{:.2},{:.2})",
            m.id,
            m.tex_id,
            m.rgba[0],
            m.rgba[1],
            m.rgba[2],
            m.rgba[3],
            m.diffuse[0],
            m.diffuse[1],
            m.diffuse[2],
            m.diffuse[3]
        );
    }
    println!("=== textures ===");
    for (i, t) in parsed.textures.iter().enumerate() {
        let fmt = match t.fmt {
            rustt::ghg::TextureFmt::Dxt1 => "DXT1",
            rustt::ghg::TextureFmt::Dxt5 => "DXT5",
        };
        let mut line = format!("tex {i}: {}x{} fmt={} payload={}", t.w, t.h, fmt, t.payload.len());
        match rustt::dxt::decode(t) {
            Ok(rgba) => {
                let mut avg = [0u64; 4];
                let mut opaque = 0u64;
                let mut nonblack = 0u64;
                for px in rgba.chunks(4) {
                    for c in 0..4 {
                        avg[c] += px[c] as u64;
                    }
                    if px[3] > 128 {
                        opaque += 1;
                    }
                    if px[0] as u32 + px[1] as u32 + px[2] as u32 > 24 {
                        nonblack += 1;
                    }
                }
                let n = (rgba.len() / 4) as u64;
                line.push_str(&format!(
                    " decoded avg=({:.0},{:.0},{:.0},{:.0}) opaque={}/{} nonblack={}",
                    avg[0] as f64 / n as f64,
                    avg[1] as f64 / n as f64,
                    avg[2] as f64 / n as f64,
                    avg[3] as f64 / n as f64,
                    opaque,
                    n,
                    nonblack
                ));
            }
            Err(e) => line.push_str(&format!(" decode FAILED: {e}")),
        }
        println!("{line}");
    }

    println!("=== bones ===");
    for (i, b) in parsed.bones.iter().enumerate() {
        let m = b.world.to_cols_array();
        println!("bone {i}: name={} parent={}", b.name, b.parent);
        println!("   [ {:8.3} {:8.3} {:8.3} {:8.3} ]", m[0], m[4], m[8], m[12]);
        println!("   [ {:8.3} {:8.3} {:8.3} {:8.3} ]", m[1], m[5], m[9], m[13]);
        println!("   [ {:8.3} {:8.3} {:8.3} {:8.3} ]", m[2], m[6], m[10], m[14]);
        println!("   [ {:8.3} {:8.3} {:8.3} {:8.3} ]", m[3], m[7], m[11], m[15]);
    }
    Ok(())
}
