use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};

fn save_png(path: &str, w: usize, h: usize, rgba: &[u8]) -> Result<()> {
    let mut enc = png::Encoder::new(std::fs::File::create(path)?, w as u32, h as u32);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut wr = enc.write_header().context("png header")?;
    wr.write_image_data(rgba).context("png data")?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).context("usage: mapdump <map.gsc>")?;
    let data = std::fs::read(path).with_context(|| format!("reading {path}"))?;
    let map = rustt::map::parse(&data).with_context(|| format!("parsing {path}"))?;

    println!("file: {}", Path::new(path).file_name().unwrap().to_string_lossy());
    println!(
        "textures: {}  vertex buffers: {}  index buffers: {}  materials: {}  meshes: {}",
        map.textures.len(),
        map.vertex_buffers.len(),
        map.index_buffers.len(),
        map.materials.len(),
        map.meshes.len()
    );
    println!(
        "scene: texCount={} gsnhData=0x{:x} texMeta=0x{:x} materialList=0x{:x} meshList=0x{:x}",
        map.scene.tex_count,
        map.scene.gsnh_data,
        map.scene.tex_meta_ptr,
        map.scene.material_list_ptr,
        map.scene.mesh_list_start
    );
    println!();

    let vb_total: usize = map.vertex_buffers.iter().map(|b| b.len()).sum();
    let ib_total: usize = map.index_buffers.iter().map(|b| b.len()).sum();
    println!(
        "vertex buffers: {} buffers, {} bytes total",
        map.vertex_buffers.len(),
        vb_total
    );
    for (i, b) in map.vertex_buffers.iter().enumerate() {
        println!("  vb[{i}] {} bytes", b.len());
    }
    println!(
        "index buffers: {} buffers, {} bytes total",
        map.index_buffers.len(),
        ib_total
    );
    for (i, b) in map.index_buffers.iter().enumerate() {
        println!("  ib[{i}] {} bytes", b.len());
    }
    println!();

    let mut by_stride = BTreeMap::new();
    let mut by_vl = BTreeMap::new();
    let mut by_il = BTreeMap::new();
    let mut by_type = BTreeMap::new();
    let mut dyn_meshes = 0usize;
    let mut total_tris = 0u64;
    for m in &map.meshes {
        *by_stride.entry(m.vertex_size).or_insert(0u32) += 1;
        *by_vl.entry(m.vertex_list_id).or_insert(0u32) += 1;
        *by_il.entry(m.index_list_id).or_insert(0u32) += 1;
        *by_type.entry(m.mesh_type).or_insert(0u32) += 1;
        if m.use_dynamic_buffer != 0 || m.dynamic_buffer != 0 {
            dyn_meshes += 1;
        }
        total_tris += m.triangle_count as u64;
    }
    println!("mesh type distribution: {by_type:?}");
    println!("vertex stride distribution: {by_stride:?}");
    println!("vertex buffer usage: {by_vl:?}");
    println!("index buffer usage: {by_il:?}");
    println!("meshes using dynamic buffers: {dyn_meshes}");
    println!("total triangles: {total_tris}");
    println!();

    if let Some(a) = args.iter().position(|x| x == "--disp") {
        let n = args
            .get(a + 1)
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(20)
            .min(map.render_parts.len());
        println!(
            "render parts: {} (meshes: {}, materials: {})",
            map.render_parts.len(),
            map.meshes.len(),
            map.materials.len()
        );
        let mut mat_usage = std::collections::BTreeMap::new();
        let mut mesh_usage = std::collections::BTreeMap::new();
        let mut bad_mat = 0usize;
        for p in &map.render_parts {
            if p.material >= map.materials.len() {
                bad_mat += 1;
            }
            *mat_usage.entry(p.material).or_insert(0u32) += 1;
            *mesh_usage.entry(p.mesh).or_insert(0u32) += 1;
        }
        println!("parts with out-of-range material: {bad_mat}");
        println!("distinct materials used: {}  distinct meshes used: {}", mat_usage.len(), mesh_usage.len());
        println!("materials referenced more than once: {}", mat_usage.values().filter(|&&c| c > 1).count());
        println!("meshes referenced more than once: {}", mesh_usage.values().filter(|&&c| c > 1).count());
        for (i, p) in map.render_parts.iter().take(n).enumerate() {
            let m = &map.meshes[p.mesh];
            let mat = map.materials.get(p.material).map(|m| m.tex_id);
            println!(
                "part {i}: mesh {} (addr 0x{:x}, {} tris, stride {}) -> material {} (tex {mat:?})",
                p.mesh, m.address, m.triangle_count, m.vertex_size, p.material
            );
        }
        return Ok(());
    }

    if let Some(a) = args.iter().position(|x| x == "--desc") {
        let n = args
            .get(a + 1)
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(4)
            .min(map.meshes.len());
        println!("=== mesh descriptors (raw 0x40 bytes) ===");
        for i in 0..n {
            let m = &map.meshes[i];
            let d = &data[m.address..m.address + 0x40];
            let mut fields: Vec<String> = Vec::new();
            for f in 0..16 {
                let o = f * 4;
                let v = u32::from_le_bytes(d[o..o + 4].try_into().unwrap());
                fields.push(format!("+0x{o:02x}=0x{v:08x}({v})"));
            }
            println!("mesh {i} @0x{:x}: {}", m.address, fields.join(" "));
        }
        // Material pointer list: each entry is a relative pointer to a 0x2C4
        // material record (same layout as the MS00 block's).
        let mlp = map.scene.material_list_ptr;        println!();
        println!("=== material list @0x{mlp:x} (first 8 entries) ===");
        for i in 0..8usize.min(map.materials.len()) {
            let off = mlp + i * 4;
            if off + 4 > data.len() {
                break;
            }
            let relv = i32::from_le_bytes(data[off..off + 4].try_into().unwrap());
            let tgt = if relv == 0 {
                0
            } else {
                (off as i64 + relv as i64) as usize
            };
        println!(
            "  [{i}] rel=0x{relv:08x} -> 0x{tgt:x}  (mat id @+0x38: {})",
            if tgt + 0x3c <= data.len() {
                i32::from_le_bytes(data[tgt + 0x38..tgt + 0x3c].try_into().unwrap())
            } else {
                -1
            }
        );
        }
        // The renderable-list header (`gp`): mesh list + count were found at
        // +0x10/+0x14; dump the whole header to hunt for the material
        // association array.
        let gp = map.scene.gsc_renderable_list;
        println!();
        println!("=== gp (renderable list header) @0x{gp:x} ===");
        for f in 0..24 {
            let o = gp + f * 4;
            if o + 4 > data.len() {
                break;
            }
            let v = u32::from_le_bytes(data[o..o + 4].try_into().unwrap());
            println!("  gp+0x{:02x} = 0x{v:08x} ({v})", f * 4);
        }
        return Ok(());
    }

    if let Some(a) = args.iter().position(|x| x == "--uvcheck") {
        let _ = a;
        use std::collections::BTreeMap;
        let mut stats: BTreeMap<u16, (u64, u64, u64)> = BTreeMap::new();
        for m in &map.meshes {
            let stride = m.vertex_size as usize;
            let Some(uvo) = rustt::mapmesh::uv_offset(stride) else {
                continue;
            };
            let Some(vb) = map.vertex_buffers.get(m.vertex_list_id as usize) else {
                continue;
            };
            let base = m.vertex_offset as usize;
            let entry = stats.entry(m.vertex_size).or_insert((0, 0, 0));
            for v in 0..m.vertex_count as usize {
                let o = base + v * stride + uvo;
                if o + 8 > vb.len() {
                    continue;
                }
                let a = f32::from_le_bytes(vb[o..o + 4].try_into().unwrap());
                let b = f32::from_le_bytes(vb[o + 4..o + 8].try_into().unwrap());
                entry.0 += 1;
                if a.is_finite() && b.is_finite() {
                    entry.1 += 1;
                    if a >= -64.0 && a <= 64.0 && b >= -64.0 && b <= 64.0 {
                        entry.2 += 1;
                    }
                }
            }
        }
        for (stride, (total, finite, inrange)) in stats {
            let pct = if total > 0 {
                100.0 * inrange as f64 / total as f64
            } else {
                0.0
            };
            println!(
                "stride {stride}: vertices={total} finite={finite} in-range(+-64)={inrange} ({pct:.1}%)"
            );
        }
        return Ok(());
    }

    if let Some(_a) = args.iter().position(|x| x == "--layout") {
        let mut seen = std::collections::BTreeMap::new();
        for m in &map.meshes {
            if seen.contains_key(&m.vertex_size) {
                continue;
            }
            let Some(vb) = map.vertex_buffers.get(m.vertex_list_id as usize) else {
                continue;
            };
            let stride = m.vertex_size as usize;
            let base = m.vertex_offset as usize;
            if base + stride > vb.len() {
                continue;
            }
            seen.insert(m.vertex_size, (m, base, vb));
        }
        for (stride, (m, base, vb)) in seen {
            println!("stride {stride}: mesh @0x{:x} vl={} vtxOff={} vtxCount={}", m.address, m.vertex_list_id, m.vertex_offset, m.vertex_count);
            for f in 0..stride as usize / 4 {
                let o = base + f * 4;
                let b = &vb[o..o + 4];
                let fv = f32::from_le_bytes(b.try_into().unwrap());
                let uv_ = u32::from_le_bytes(b.try_into().unwrap());
                println!("  [{f}] f32={fv:.6} u32=0x{uv_:08x}");
            }
        }
        return Ok(());
    }

    if let Some(_a) = args.iter().position(|x| x == "--parts") {
        use std::collections::HashMap;
        let mut parts_per_mesh: HashMap<usize, usize> = HashMap::new();
        let mut oob_mesh = 0usize;
        for part in &map.render_parts {
            if part.mesh >= map.meshes.len() {
                oob_mesh += 1;
                continue;
            }
            *parts_per_mesh.entry(part.mesh).or_insert(0) += 1;
        }
        println!(
            "render parts: {} total, {} distinct mesh refs, {} out-of-range refs",
            map.render_parts.len(),
            parts_per_mesh.len(),
            oob_mesh
        );
        let mut reasons: HashMap<String, usize> = HashMap::new();
        let mut dyn_count = 0usize;
        let mut ok = 0usize;
        for (&mi, &count) in &parts_per_mesh {
            let m = &map.meshes[mi];
            let stride = m.vertex_size as usize;
            if m.use_dynamic_buffer != 0 {
                dyn_count += 1;
                *reasons.entry("dynamic buffer".into()).or_insert(0) += count;
                continue;
            }
            let reason = if rustt::mapmesh::uv_offset(stride).is_none() {                Some(format!("unknown stride {stride}"))
            } else {
                let vb_ok = map
                    .vertex_buffers
                    .get(m.vertex_list_id as usize)
                    .is_some_and(|vb| m.vertex_offset as usize + m.vertex_count as usize * stride <= vb.len());
                let ib_ok = map.index_buffers.get(m.index_list_id as usize).is_some_and(|ib| {
                    m.index_offset as usize * 2 + (m.triangle_count as usize + 2) * 2 <= ib.len()
                });
                match (vb_ok, ib_ok) {
                    (true, true) => {
                        match rustt::mapmesh::expand_mesh(&map, m) {
                            Some(md) => {
                                if md.pos.is_empty() || md.idx.is_empty() {
                                    Some("empty output".into())
                                } else {
                                    None
                                }
                            }
                            None => Some("expand failed".into()),
                        }
                    }
                    (false, _) => Some("vertex range oob".into()),
                    (_, false) => Some("index range oob".into()),
                }
            };
            match reason {
                Some(r) => *reasons.entry(r).or_insert(0) += count,
                None => ok += count,
            }
        }
        println!("ok parts: {ok}");
        println!("dynamic-buffer meshes referenced: {dyn_count}");
        let mut sorted: Vec<_> = reasons.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        for (r, c) in sorted {
            println!("  {r}: {c}");
        }
        return Ok(());
    }

    if let Some(a) = args.iter().position(|x| x == "--tex") {
        let outdir = args.get(a + 1).map(String::as_str).unwrap_or(".");
        std::fs::create_dir_all(outdir).context("creating texture output dir")?;
        let tex = &map.textures;
        for (i, t) in tex.iter().enumerate() {
            let tex = rustt::ghg::Texture {
                w: t.w,
                h: t.h,
                fmt: t.fmt,
                payload: t.payload.clone(),
            };
            let rgba = rustt::dxt::decode(&tex).with_context(|| format!("decoding tex {i}"))?;
            let p = format!("{outdir}/tex_{i:03}.png");
            save_png(&p, t.w, t.h, &rgba)?;
        }
        println!("wrote {} textures to {outdir}", tex.len());
        return Ok(());
    }

    if let Some(a) = args.iter().position(|x| x == "--fail") {
        let idx = args.get(a + 1).and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
        let mut seen = 0usize;
        for (mi, m) in map.meshes.iter().enumerate() {
            if rustt::mapmesh::expand_mesh(&map, m).is_some() {
                continue;
            }
            if seen != idx {
                seen += 1;
                continue;
            }
            let vb_len = map
                .vertex_buffers
                .get(m.vertex_list_id as usize)
                .map(|b| b.len())
                .unwrap_or(0);
            let ib_len = map
                .index_buffers
                .get(m.index_list_id as usize)
                .map(|b| b.len())
                .unwrap_or(0);
            let io = m.index_offset as usize * 2;
            let ic = m.triangle_count as usize + 2;
            println!(
                "failing mesh {mi} @0x{:x}: type={} triCount={} vtxSize={} vtxOff={} vtxCount={} idxOff={} vl={} vbLen={} ibLen={}",
                m.address, m.mesh_type, m.triangle_count, m.vertex_size,
                m.vertex_offset, m.vertex_count, m.index_offset,
                m.vertex_list_id, vb_len, ib_len
            );
            if io + ic * 2 <= ib_len {
                let ibuf = map.index_buffers[m.index_list_id as usize];
                let idx: Vec<u16> = ibuf[io..io + ic * 2]
                    .chunks(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect();
                let maxi = idx.iter().max().copied().unwrap_or(0);
                let minoob = idx
                    .iter()
                    .position(|&x| x as usize >= m.vertex_count as usize);
                println!(
                    "  strip indices: n={} max={} vtxCount={} first-oob@={:?}",
                    idx.len(),
                    maxi,
                    m.vertex_count,
                    minoob
                );
                println!("  first 24: {:?}", &idx[..24.min(idx.len())]);
                let inb = idx
                    .iter()
                    .filter(|&&x| (x as usize) < m.vertex_count as usize)
                    .count();
                println!("  indices < vtxCount: {inb}/{}", idx.len());
            }
            return Ok(());
        }
        println!("no failing mesh found for index {idx}");
        return Ok(());
    }

    if let Some(_a) = args.iter().position(|x| x == "--scan") {
        let max = args
            .iter()
            .position(|x| x == "--scan")
            .and_then(|a| args.get(a + 1))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(12);
        for (mi, m) in map.meshes.iter().take(max).enumerate() {
            let stride = m.vertex_size as usize;
            let vb_cap = map
                .vertex_buffers
                .get(m.vertex_list_id as usize)
                .map(|b| b.len() / stride.max(1))
                .unwrap_or(0);
            let io = m.index_offset as usize * 2;
            let ic = m.triangle_count as usize + 2;
            let (maxi, minidx) = map
                .index_buffers
                .get(m.index_list_id as usize)
                .filter(|ibuf| io + ic * 2 <= ibuf.len())
                .map(|ibuf| {
                    let idx: Vec<u16> = ibuf[io..io + ic * 2]
                        .chunks(2)
                        .map(|c| u16::from_le_bytes([c[0], c[1]]))
                        .collect();
                    (
                        idx.iter().max().copied().unwrap_or(0),
                        idx.iter().min().copied().unwrap_or(0),
                    )
                })
                .unwrap_or((0, 0));
            let ok = rustt::mapmesh::expand_mesh(&map, m).is_some();
            println!(
                "mesh {mi:3}: vtxOff={:5} vtxCount={:4} stride={:2} idxMin={:4} idxMax={:4} vbVertCap={:6} expand={ok}",
                m.vertex_offset, m.vertex_count, stride, minidx, maxi, vb_cap
            );
        }
        return Ok(());
    }

    if let Some(a) = args.iter().position(|x| x == "--mesh") {
        let idx = args.get(a + 1).and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
        let m = map.meshes.get(idx).context("mesh index out of range")?;
        let vb = map
            .vertex_buffers
            .get(m.vertex_list_id as usize)
            .context("mesh vertex buffer out of range")?;
        let ib = map
            .index_buffers
            .get(m.index_list_id as usize)
            .context("mesh index buffer out of range")?;
        let stride = m.vertex_size as usize;
        let base = m.vertex_offset as usize;
        println!(
            "mesh {idx} @0x{:x}: type={} triCount={} vtxSize={} vtxOff={} vtxCount={} idxOff={} il={} vl={} dyn={} dynBuf=0x{:x}",
            m.address, m.mesh_type, m.triangle_count, m.vertex_size,
            m.vertex_offset, m.vertex_count, m.index_offset,
            m.index_list_id, m.vertex_list_id, m.use_dynamic_buffer, m.dynamic_buffer
        );
        if m.use_dynamic_buffer != 0 {
            println!(
                "  (dynamic buffers: pointer 0x{:x} at vb offset {}, needs base-pose resolution)",
                m.dynamic_buffer, m.vertex_offset
            );
        }
        for v in 0..4usize.min(m.vertex_count as usize) {
            let o = base + v * stride;
            let line: Vec<String> = (0..(stride / 4))
                .map(|f| {
                    let b = &vb[o + f * 4..o + f * 4 + 4];
                    format!("{:.4}", f32::from_le_bytes(b.try_into().unwrap()))
                })
                .collect();
            println!("  v{v} [{o:#x}]: {}", line.join(" "));
        }
        println!("  indices (u16 strip, first 12):");
        let io = m.index_offset as usize * 2;
        let n = (12usize).min((m.triangle_count as usize + 2) * 2);
        let idx: Vec<String> = ib[io..io + n]
            .chunks(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]).to_string())
            .collect();
        println!("    {}", idx.join(" "));
        return Ok(());
    }

    println!("=== materials ===");
    for (i, m) in map.materials.iter().enumerate() {
        if i >= 12 {
            println!("  ... {} more", map.materials.len() - 12);
            break;
        }
        println!(
            "mat {i}: id={} tex={} diff=({:.2},{:.2},{:.2},{:.2}) texFlags=0x{:x} vformat=0x{:x}",
            m.id,
            m.tex_id,
            m.diffuse[0],
            m.diffuse[1],
            m.diffuse[2],
            m.diffuse[3],
            m.texture_flags,
            m.vertex_format_bits
        );
    }
    println!();

    println!("=== textures ===");
    for (i, t) in map.textures.iter().enumerate() {
        if i >= 16 {
            println!("  ... {} more", map.textures.len() - 16);
            break;
        }
        let fmt = match t.fmt {
            rustt::ghg::TextureFmt::Dxt1 => "DXT1",
            rustt::ghg::TextureFmt::Dxt5 => "DXT5",
        };
        println!("tex {i}: {}x{} fmt={fmt} payload={}", t.w, t.h, t.payload.len());
    }

    Ok(())
}
