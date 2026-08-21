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

    // --cmds: raw DISP command list + game-model command mappings
    if args.iter().position(|x| x == "--cmds").is_some() {
        // Re-find the DISP block from raw data.
        let nu20 = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize + 4;
        let mut disp_content: Option<usize> = None;
        {
            let mut pos = nu20 + 0x20;
            while pos + 8 <= data.len() {
                let id = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap());
                let sz = u32::from_le_bytes(data[pos+4..pos+8].try_into().unwrap()) as usize;
                if sz < 8 || pos + sz > data.len() { break; }
                if id == 0x50534944 { // "DISP"
                    disp_content = Some(pos + 8);
                    break;
                }
                pos += sz;
            }
        }
        let Some(disp) = disp_content else {
            println!("no DISP block found");
            return Ok(());
        };

        // Helper: self-relative pointer
        fn rel_cmd(d: &[u8], off: usize) -> usize {
            let v = i32::from_le_bytes(d[off..off+4].try_into().unwrap());
            if v == 0 { 0 } else { (off as i64 + v as i64) as usize }
        }

        // Command list at disp+8
        let cmd_start = rel_cmd(&data, disp + 8);
        let mut commands: Vec<(u8, u8, u32, usize)> = Vec::new(); // (type, flags, resource, file_offset)
        let mut pos = cmd_start;
        loop {
            if pos + 16 > data.len() { break; }
            let ty = data[pos];
            let flags = data[pos + 1];
            let resource = rel_cmd(&data, pos + 4) as u32;
            commands.push((ty, flags, resource, pos));
            if ty == 0x8e { break; } // END
            pos += 16;
        }
        println!("=== DISP command list ({} commands) ===", commands.len());
        for (i, &(ty, flags, resource, foff)) in commands.iter().enumerate() {
            let type_name = match ty {
                0x80 => "MTL",
                0x82 => "GEOMCALL",
                0x83 => "MTXLOAD",
                0x84 => "TERMINATE",
                0x85 => "MTL_CLIP",
                0x87 => "DUMMY",
                0x8b => "DYN_GEOM",
                0x8d => "NEXT",
                0x8e => "END",
                0x8f => "FACEON",
                0xb0 => "LIGHTMAP",
                _ => "OTHER",
            };
            // For MTXLOAD, try to read the matrix translation (row3 col0-2 = offsets 0x30,0x34,0x38)
            let xform_info = if ty == 0x83 && (resource as usize) + 64 <= data.len() {
                let a = resource as usize;
                let f = |o: usize| -> f32 { f32::from_le_bytes(data[a+o..a+o+4].try_into().unwrap()) };
                format!(" mat=[{:.2},{:.2},{:.2}]", f(0x30), f(0x34), f(0x38))
            } else if ty == 0x82 {
                // GEOMCALL: check if the resource is a known mesh address
                let mesh_idx = map.meshes.iter().position(|m| m.address as u32 == resource);
                match mesh_idx {
                    Some(mi) => format!(" mesh#{mi}(addr=0x{:x})", resource),
                    None => format!(" addr=0x{:x}(NOT IN MESH LIST)", resource),
                }
            } else {
                format!(" res=0x{:x}", resource)
            };
            println!("  [{i:3}] @0x{:06x} type=0x{:02x}({:<10}) flags={}{}",
                foff, ty, type_name, flags, xform_info);
        }

        // Game models
        let model_count = u32::from_le_bytes(data[disp+0x10..disp+0x14].try_into().unwrap()) as usize;
        let models_off = rel_cmd(&data, disp + 0x14);
        println!("\n=== Game models ({} total) ===", model_count);
        let mut total_parts = 0usize;
        let mut skipped_parts = 0usize;
        for i in 0..model_count {
            let a = models_off + i * 0xc;
            if a + 0xc > data.len() { break; }
            let cmd_count = u32::from_le_bytes(data[a..a+4].try_into().unwrap()) as usize;
            if cmd_count == 0 { continue; }
            let mat_off = rel_cmd(&data, a + 4);
            let mesh_off = rel_cmd(&data, a + 8);
            if mat_off == 0 || mesh_off == 0 { continue; }

            println!("  GM[{i:3}] cmdCount={cmd_count} matOff=0x{mat_off:x} meshOff=0x{mesh_off:x}");
            for k in 0..cmd_count {
                total_parts += 1;
                let mat_idx = u32::from_le_bytes(data[mat_off + k*4..mat_off + k*4 + 4].try_into().unwrap()) as usize;
                let cmd_idx = u32::from_le_bytes(data[mesh_off + k*4..mesh_off + k*4 + 4].try_into().unwrap()) as usize;

                if cmd_idx >= commands.len() {
                    println!("    part[{k}] cmdIdx={cmd_idx} OUT OF RANGE (cmd list has {} entries)", commands.len());
                    skipped_parts += 1;
                    continue;
                }
                let (ty, flags, resource, _) = commands[cmd_idx];
                let type_name = match ty {
                    0x82 => "GEOMCALL",
                    0x83 => "MTXLOAD",
                    0xb0 => "LIGHTMAP",
                    0x8e => "END",
                    _ => "OTHER",
                };
                let mesh_addr = if ty == 0x82 { resource } else { 0 };
                let mesh_info = if ty == 0x82 {
                    match map.meshes.iter().position(|m| m.address as u32 == mesh_addr) {
                        Some(mi) => format!("mesh#{mi}"),
                        None => format!("addr=0x{mesh_addr:x}(MISSING)"),
                    }
                } else {
                    format!("-")
                };
                let mat_info = if mat_idx < map.materials.len() {
                    let tex_id = map.materials[mat_idx].tex_id;
                    format!("mat#{mat_idx}(tex={tex_id})")
                } else {
                    format!("mat#{mat_idx}(OUT OF RANGE)")
                };
                let xform_info = if ty == 0x82 && (resource as usize) + 64 <= data.len() {
                    let a2 = resource as usize;
                    // Check if mesh vertices are in local space by reading first vertex pos
                    format!("") // We'll skip this for brevity
                } else {
                    format!("")
                };
                println!("    part[{k}] cmdIdx={cmd_idx} -> {type_name} {mesh_info} {mat_info}");
                if ty != 0x82 {
                    skipped_parts += 1;
                }
            }
        }
        println!("\n  total parts from GM: {total_parts}, non-GEOMCALL skipped: {skipped_parts}");
        println!("  render_parts in Map: {}", map.render_parts.len());
        return Ok(());
    }

    // --spec: parse and dump SpecialObjects from the DISP block
    if args.iter().position(|x| x == "--spec").is_some() {
        let nu20 = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize + 4;
        let mut disp_content: Option<usize> = None;
        {
            let mut pos = nu20 + 0x20;
            while pos + 8 <= data.len() {
                let id = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap());
                let sz = u32::from_le_bytes(data[pos+4..pos+8].try_into().unwrap()) as usize;
                if sz < 8 || pos + sz > data.len() { break; }
                if id == 0x50534944 { // "DISP"
                    disp_content = Some(pos + 8);
                    break;
                }
                pos += sz;
            }
        }
        let Some(disp) = disp_content else {
            println!("no DISP block found");
            return Ok(());
        };
        fn rel(d: &[u8], off: usize) -> usize {
            let v = i32::from_le_bytes(d[off..off+4].try_into().unwrap());
            if v == 0 { 0 } else { (off as i64 + v as i64) as usize }
        }

        let spec_count = u32::from_le_bytes(data[disp+0x6c..disp+0x70].try_into().unwrap()) as usize;
        let spec_ptr = rel(&data, disp + 0x70);
        println!("SpecialObjects: count={spec_count} ptr=0x{:x}", spec_ptr);

        // Game models
        let model_count = u32::from_le_bytes(data[disp+0x10..disp+0x14].try_into().unwrap()) as usize;
        let models_off = rel(&data, disp + 0x14);
        println!("GameModels: count={model_count} base=0x{:x}", models_off);

        // Also parse NTBL for names
        let mut names: std::collections::HashMap<usize, String> = std::collections::HashMap::new();
        let mut ntbl_start = 0usize;
        let mut ntbl_end = 0usize;
        {
            let mut pos = nu20 + 0x20;
            while pos + 8 <= data.len() {
                let id = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap());
                let sz = u32::from_le_bytes(data[pos+4..pos+8].try_into().unwrap()) as usize;
                if sz < 8 || pos + sz > data.len() { break; }
                if id == 0x4C42544E { // "NTBL"
                    let content = pos + 8;
                    let end = pos + sz;
                    ntbl_start = content + 4; // skip u32 size
                    ntbl_end = end;
                    let mut p = ntbl_start;
                    while p < end {
                        let start = p;
                        while p < end && data[p] != 0 { p += 1; }
                        let name = String::from_utf8_lossy(&data[start..p]).to_string();
                        names.insert(start, name);
                        if p < end { p += 1; }
                    }
                    break;
                }
                pos += sz;
            }
        }
        println!("NTBL names: {} range=0x{:x}-0x{:x}", names.len(), ntbl_start, ntbl_end);

        // STEP 1: Scan every 4-byte offset in the special object to find which one
        // resolves to a valid game model entry via self-relative pointer.
        println!("\n=== Scanning for model pointer offset ===");
        let mut found_model_offset: Option<usize> = None;
        let mut found_string_offset: Option<usize> = None;
        for trial_off in (0..0xd0u32).step_by(4) {
            let mut valid_count = 0usize;
            let mut name_match_count = 0usize;
            for i in 0..spec_count.min(50) {
                let a = spec_ptr + i * 0xd0;
                if a + trial_off as usize + 4 > data.len() { break; }
                let v = u32::from_le_bytes(data[a + trial_off as usize..a + trial_off as usize + 4].try_into().unwrap());
                let abs = rel(&data, a + trial_off as usize);
                // Check: does this resolve to a valid game model entry?
                if abs >= models_off && abs + 4 <= models_off + model_count * 0xc {
                    let off_in_models = abs - models_off;
                    if off_in_models % 0xc == 0 {
                        let cmd_count = u32::from_le_bytes(data[abs..abs+4].try_into().unwrap()) as usize;
                        if cmd_count > 0 && cmd_count < 100 {
                            valid_count += 1;
                        }
                    }
                }
            }
            if valid_count > 0 {
                println!("  +0x{:02x}: {} / 50 SOs resolve to valid game model entries", trial_off, valid_count);
                if found_model_offset.is_none() && valid_count > 25 {
                    found_model_offset = Some(trial_off as usize);
                }
            }
        }

        // STEP 2: Also scan for string pointer (absolute file offset into NTBL range)
        println!("\n=== Scanning for string pointer offset ===");
        for trial_off in (0..0xd0u32).step_by(4) {
            let mut valid_count = 0usize;
            let mut named_count = 0usize;
            for i in 0..spec_count.min(50) {
                let a = spec_ptr + i * 0xd0;
                if a + trial_off as usize + 4 > data.len() { break; }
                let v = u32::from_le_bytes(data[a + trial_off as usize..a + trial_off as usize + 4].try_into().unwrap()) as usize;
                // Check: is this a valid absolute offset into NTBL?
                if v >= ntbl_start && v < ntbl_end {
                    valid_count += 1;
                    if names.contains_key(&v) { named_count += 1; }
                }
            }
            if valid_count > 0 {
                println!("  +0x{:02x}: {} / 50 SOs in NTBL range ({} named)", trial_off, valid_count, named_count);
                if found_string_offset.is_none() && valid_count > 25 {
                    found_string_offset = Some(trial_off as usize);
                }
            }
        }

        // STEP 3: Also scan as self-relative for strings
        println!("\n=== Scanning for string pointer offset (self-relative) ===");
        for trial_off in (0..0xd0u32).step_by(4) {
            let mut valid_count = 0usize;
            let mut named_count = 0usize;
            for i in 0..spec_count.min(50) {
                let a = spec_ptr + i * 0xd0;
                if a + trial_off as usize + 4 > data.len() { break; }
                let abs = rel(&data, a + trial_off as usize);
                if abs >= ntbl_start && abs < ntbl_end {
                    valid_count += 1;
                    if names.contains_key(&abs) { named_count += 1; }
                }
            }
            if valid_count > 0 {
                println!("  +0x{:02x}: {} / 50 SOs in NTBL range via rel ({} named)", trial_off, valid_count, named_count);
                if found_string_offset.is_none() && valid_count > 25 {
                    found_string_offset = Some(trial_off as usize);
                }
            }
        }

        println!("\n=== Results ===");
        if let Some(o) = found_model_offset {
            println!("  Model pointer offset: +0x{:02x}", o);
        } else {
            println!("  Model pointer offset: NOT FOUND in 0x00-0xcc");
        }
        if let Some(o) = found_string_offset {
            println!("  String pointer offset: +0x{:02x}", o);
        } else {
            println!("  String pointer offset: NOT FOUND in 0x00-0xcc");
        }

        // STEP 4: Dump first 10 special objects with full hex and resolved fields
        if let Some(model_off) = found_model_offset {
            let string_off = found_string_offset.unwrap_or(model_off + 4);
            println!("\n=== First 10 SpecialObjects ===");
            for i in 0..spec_count.min(10) {
                let a = spec_ptr + i * 0xd0;
                if a + 0xd0 > data.len() { break; }
                let f = |o: usize| -> f32 { f32::from_le_bytes(data[a+o..a+o+4].try_into().unwrap()) };
                println!("\n  SO[{}] @0x{:x}:", i, a);
                for r in 0..4 {
                    let vals: Vec<String> = (0..4).map(|c| {
                        format!("{:.3}", f((r*4+c)*4))
                    }).collect();
                    println!("    mat{}: [{}]", r, vals.join(", "));
                }
                // Dump full hex from 0x40 to 0xd0
                let raw: Vec<String> = (0x40..0xd0).step_by(4).map(|o| {
                    let v = u32::from_le_bytes(data[a+o..a+o+4].try_into().unwrap());
                    format!("+0x{:02x}=0x{:08x}", o, v)
                }).collect();
                println!("    hex: {}", raw.join(" "));
                // Resolve model and string
                let model_abs = rel(&data, a + model_off);
                let model_valid = model_abs >= models_off && (model_abs - models_off) % 0xc == 0;
                let model_idx = if model_valid { Some((model_abs - models_off) / 0xc) } else { None };
                let string_abs = rel(&data, a + string_off);
                let string_name = names.get(&string_abs).map(|s| s.as_str()).unwrap_or("");
                println!("    model_off=+0x{:02x} -> 0x{:x} valid={} idx={:?}  string_off=+0x{:02x} -> 0x{:x} '{}'",
                    model_off, model_abs, model_valid, model_idx,
                    string_off, string_abs, string_name);
            }

            // Count unique models
            let mut model_counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
            let mut named = 0usize;
            for i in 0..spec_count {
                let a = spec_ptr + i * 0xd0;
                if a + 0xd0 > data.len() { break; }
                let model_abs = rel(&data, a + model_off);
                *model_counts.entry(model_abs).or_insert(0) += 1;
                let string_abs = rel(&data, a + string_off);
                if names.contains_key(&string_abs) { named += 1; }
            }
            println!("\n  {} total, {} named, {} distinct model addrs", spec_count, named, model_counts.len());
        }

        return Ok(());
    }

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
            let Some(uvo) = rustt::mapmesh::uv_offset(stride, 0) else {
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
            let reason = if rustt::mapmesh::uv_offset(stride, 0).is_none() {                Some(format!("unknown stride {stride}"))
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

    if let Some(a) = args.iter().position(|x| x == "--lmstats") {
        let idx: Vec<usize> = args
            .get(a + 1)
            .map(|s| {
                s.split(',')
                    .filter_map(|t| t.parse::<usize>().ok())
                    .collect()
            })
            .unwrap_or_else(|| Vec::new());
        let list: Vec<usize> = if idx.is_empty() {
            (0..map.textures.len().min(12)).collect()
        } else {
            idx
        };
        println!("tex: (idx) w x h minLum maxLum meanLum chroma dark% hi%");
        for i in list {
            let Some(t) = map.textures.get(i) else {
                continue;
            };
            let tex = rustt::ghg::Texture {
                w: t.w,
                h: t.h,
                fmt: t.fmt,
                payload: t.payload.clone(),
            };
            let Ok(rgba) = rustt::dxt::decode(&tex) else {
                println!("tex {i}: decode failed");
                continue;
            };
            let mut minl = 1.0f32;
            let mut maxl = 0.0f32;
            let mut sum = 0.0f64;
            let mut chroma = 0.0f64;
            let mut dark = 0usize;
            let mut hi = 0usize;
            let px = rgba.len() / 4;
            for p in 0..px {
                let (r, g, b) = (
                    rgba[p * 4] as f32 / 255.0,
                    rgba[p * 4 + 1] as f32 / 255.0,
                    rgba[p * 4 + 2] as f32 / 255.0,
                );
                let l = 0.299 * r + 0.587 * g + 0.114 * b;
                minl = minl.min(l);
                maxl = maxl.max(l);
                sum += l as f64;
                chroma += ((r - g).abs() + (g - b).abs() + (b - r).abs()) as f64 / 3.0;
                if l < 0.15 {
                    dark += 1;
                }
                if r > 0.9 && g > 0.9 && b > 0.9 {
                    hi += 1;
                }
            }
            let mean = sum / px as f64;
            println!(
                "tex {i}: {}x{} minL={:.3} maxL={:.3} meanL={:.3} chroma={:.4} dark={}% hi={}%",
                t.w,
                t.h,
                minl,
                maxl,
                mean,
                chroma / px as f64,
                dark * 100 / px.max(1),
                hi * 100 / px.max(1)
            );
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

    if let Some(_a) = args.iter().position(|x| x == "--stage") {
        use std::collections::BTreeMap;
        let mut by_stage: BTreeMap<(u8, bool), (usize, usize, usize)> = BTreeMap::new();
        for m in &map.meshes {
            let Some(p) = map
                .render_parts
                .iter()
                .find(|p| map.meshes.get(p.mesh).map(|mm| mm.address) == Some(m.address))
            else {
                continue;
            };
            let Some(mat) = map.materials.get(p.material) else {
                continue;
            };
            let e = by_stage
                .entry((mat.lighting_stage, mat.shader_defines & 0x1000 != 0))
                .or_insert((0, 0, 0));
            e.0 += 1;
            e.1 += 1;
            e.2 += m.vertex_count as usize;
        }
        println!(
            "lighting stage per (stage, prelit): (materials, meshes, verts)",
        );
        for ((stage, prelit), (nm, nmm, nv)) in &by_stage {
            println!(
                "stage {stage} prelit={prelit:<5} mats={nm:<4} meshes={nmm:<5} verts={nv}"
            );
        }
        let live = map
            .materials
            .iter()
            .filter(|m| m.shader_defines & 0x8000_0000 != 0)
            .count();
        println!("materials with LIVE bit (0x80000000): {live}/{}", map.materials.len());
        return Ok(());
    }

    if let Some(_a) = args.iter().position(|x| x == "--lm") {
        use rustt::ghg::TextureFmt;
        let mut enabled = 0usize;
        let mut smooth = 0usize;
        let mut set_hist: BTreeMap<u8, usize> = BTreeMap::new();
        for (i, m) in map.materials.iter().enumerate() {
            let lm_stage = m.lightmap_stage();
            if lm_stage == 0 {
                continue;
            }
            enabled += 1;
            if lm_stage == 1 {
                smooth += 1;
            }
            *set_hist.entry(m.lightmap_set_index).or_insert(0) += 1;
            let uvset = m.lightmap_uvset();
            if enabled <= 3 {
                let mlp = map.scene.material_list_ptr;
                let relv = i32::from_le_bytes(data[mlp + i * 4..mlp + i * 4 + 4].try_into().unwrap());
                let q = ((mlp + i * 4) as i64 + relv as i64) as usize;
                let o32 = |o: usize| {
                    data.get(q + o..q + o + 4)
                        .map(|b| format!("0x{:08x}", u32::from_le_bytes(b.try_into().unwrap())))
                        .unwrap_or_else(|| "??".into())
                };
                println!(
                    "  mat@{q:#x}: +0x70:{} +0x74:{} +0x78:{} +0x7C:{} +0x80:{} +0x84:{} +0xB0:{} +0xB4:{} +0x98:{} +0xFC:{} +0x100:{} +0x104:{} +0x108:{} +0x10C:{} +0x110:{} +0x114:{} +0x118:{} +0x11C:{}",
                    o32(0x70), o32(0x74), o32(0x78), o32(0x7C), o32(0x80), o32(0x84),
                    o32(0xB0), o32(0xB4), o32(0x98), o32(0xFC), o32(0x100), o32(0x104),
                    o32(0x108), o32(0x10C), o32(0x110), o32(0x114), o32(0x118), o32(0x11C)
                );
            }
            let cand = |idx: u8| -> String {
                match map.tex_slot(idx as i16) {
                    Some(s) => {
                        let t = &map.textures[s];
                        let fmt = match t.fmt {
                            TextureFmt::Dxt1 => "DXT1",
                            TextureFmt::Dxt5 => "DXT5",
                        };
                        format!("tex[{idx}]={}x{} {fmt}", t.w, t.h)
                    }
                    None => format!("tex[{idx}]=??"),
                }
            };
            println!(
                "mat {i}: id={} tex={} set={} stage={} uvset={} (coords=0x{:x}) diffuse:{} smooth:{} dir1:{} dir2:{}",
                m.id,
                m.tex_id,
                m.lightmap_set_index,
                lm_stage,
                uvset,
                m.uv_set_coords,
                cand(m.lightmap_set_index),
                cand(m.lightmap_set_index.wrapping_add(1)),
                cand(m.lightmap_set_index.wrapping_add(2)),
                cand(m.lightmap_set_index.wrapping_add(3)),
            );
        }
        println!();
        println!(
            "lightmap materials: {enabled}/{} (smooth={smooth}, directional={})",
            map.materials.len(),
            enabled - smooth
        );
        println!("set-index histogram: {:?}", set_hist);
        return Ok(());
    }

    if let Some(_a) = args.iter().position(|x| x == "--lmuv") {
        let mut by: BTreeMap<(u16, u8), (usize, usize, usize, usize, usize, f32, f32, f32, f32)> = BTreeMap::new();
        for m in &map.meshes {
            let Some(p) = map
                .render_parts
                .iter()
                .find(|p| map.meshes.get(p.mesh).map(|mm| mm.address) == Some(m.address))
            else {
                continue;
            };
            let Some(mat) = map.materials.get(p.material) else {
                continue;
            };
            if mat.lightmap_stage() == 0 {
                continue;
            }
            let stride = m.vertex_size as usize;
            let set = mat.lightmap_uvset();
            let Some(off) = rustt::mapmesh::uv_set_offset(stride, mat.vertex_format_bits, set as usize)
            else {
                continue;
            };
            let Some(vb) = map.vertex_buffers.get(m.vertex_list_id as usize) else {
                continue;
            };
            let base = m.vertex_offset as usize;
            let mut negx = 0usize;
            let mut posx = 0usize;
            let mut bad = 0usize;
            let mut umin = f32::MAX;
            let mut umax = f32::MIN;
            let mut vmin = f32::MAX;
            let mut vmax = f32::MIN;
            let n = (m.vertex_count as usize).min((vb.len() / stride).saturating_sub(base));
            for v in 0..n {
                let o = base * stride + v * stride + off;
                let u = f32::from_le_bytes(vb[o..o + 4].try_into().unwrap());
                let vv = f32::from_le_bytes(vb[o + 4..o + 8].try_into().unwrap());
                if !u.is_finite() || !vv.is_finite() {
                    bad += 1;
                    continue;
                }
                if u <= 0.0 {
                    negx += 1;
                    continue;
                }
                posx += 1;
                umin = umin.min(u);
                umax = umax.max(u);
                vmin = vmin.min(vv);
                vmax = vmax.max(vv);
            }
            let e = by
                .entry((m.vertex_size, set))
                .or_insert((0, 0, 0, 0, 0, f32::MAX, f32::MIN, f32::MAX, f32::MIN));
            e.0 += 1;
            e.1 += n;
            e.2 += negx;
            e.3 += posx;
            e.4 += bad;
            if posx > 0 {
                e.5 = e.5.min(umin);
                e.6 = e.6.max(umax);
                e.7 = e.7.min(vmin);
                e.8 = e.8.max(vmax);
            }
        }
        println!("(stride, uvset): meshes verts negX% posX% bad% | x>0: u[min..max] v[min..max]");
        for ((stride, set), (meshes, verts, negx, posx, bad, umin, umax, vmin, vmax)) in &by {
            let p = |n: usize| n * 100 / (*verts).max(1);
            println!(
                "stride {stride:<2} uvset {set}: meshes={meshes:<4} verts={verts:<7} negX={}% posX={}% bad={}% | u[{umin:.3}..{umax:.3}] v[{vmin:.3}..{vmax:.3}]",
                p(*negx),
                p(*posx),
                p(*bad)
            );
        }
        return Ok(());
    }

    if let Some(_a) = args.iter().position(|x| x == "--colorhist") {
        use std::collections::BTreeMap;
        let mut stat: BTreeMap<(u16, u32), (u64, [u64; 256], [u64; 256], [u64; 256], [u64; 256])> =
            BTreeMap::new();
        for m in &map.meshes {
            let vfbits = map
                .render_parts
                .iter()
                .find(|p| map.meshes.get(p.mesh).map(|mm| mm.address) == Some(m.address))
                .and_then(|p| map.materials.get(p.material))
                .map(|mat| mat.vertex_format_bits)
                .unwrap_or(0);
            let Some(md) = rustt::mapmesh::expand_mesh(&map, m) else {
                continue;
            };
            let e = stat.entry((m.vertex_size, vfbits)).or_insert((0, [0; 256], [0; 256], [0; 256], [0; 256]));
            for c in &md.color {
                e.0 += 1;
                e.1[c[0] as usize] += 1;
                e.2[c[1] as usize] += 1;
                e.3[c[2] as usize] += 1;
                e.4[c[3] as usize] += 1;
            }
        }
        let lo = |h: &[u64; 256]| h.iter().position(|&c| c > 0).unwrap_or(0);
        let hi = |h: &[u64; 256]| h.iter().rposition(|&c| c > 0).unwrap_or(0);
        let mode = |h: &[u64; 256]| {
            h.iter()
                .enumerate()
                .max_by_key(|&(_, &c)| c)
                .map(|(v, _)| v)
                .unwrap_or(0)
        };
        for ((stride, vfbits), (n, r, g, b, a)) in &stat {
            let (rl, rh, rm) = (lo(r), hi(r), mode(r));
            let (gl, gh, gm) = (lo(g), hi(g), mode(g));
            let (bl, bh, bm) = (lo(b), hi(b), mode(b));
            let (al, ah, am) = (lo(a), hi(a), mode(a));
            println!(
                "stride {stride:<2} vf 0x{vfbits:08x}: verts={n:<7} R[{rl}..{rh}]={rm} G[{gl}..{gh}]={gm} B[{bl}..{bh}]={bm} A[{al}..{ah}]={am}",
            );
        }
        let (mut tn, mut tr, mut tg, mut tb, mut ta) = (0u64, [0u64; 256], [0u64; 256], [0u64; 256], [0u64; 256]);
        for ((_, _), (n, r, g, b, a)) in &stat {
            tn += n;
            for i in 0..256 {
                tr[i] += r[i];
                tg[i] += g[i];
                tb[i] += b[i];
                ta[i] += a[i];
            }
        }
        let gt127 = |h: &[u64; 256]| h[128..].iter().sum::<u64>();
        let (rl, rh, rm) = (lo(&tr), hi(&tr), mode(&tr));
        let (gl, gh, gm) = (lo(&tg), hi(&tg), mode(&tg));
        let (bl, bh, bm) = (lo(&tb), hi(&tb), mode(&tb));
        let (al, ah, am) = (lo(&ta), hi(&ta), mode(&ta));
        println!(
            "TOTAL: verts={tn} R[{rl}..{rh}]={rm} G[{gl}..{gh}]={gm} B[{bl}..{bh}]={bm} A[{al}..{ah}]={am}",
        );
        println!(
            "  bytes >127 (would clamp to 1.0 after *2): R={}% G={}% B={}% A={}%",
            gt127(&tr) * 100 / tn.max(1),
            gt127(&tg) * 100 / tn.max(1),
            gt127(&tb) * 100 / tn.max(1),
            gt127(&ta) * 100 / tn.max(1),
        );
        return Ok(());
    }

    if let Some(_a) = args.iter().position(|x| x == "--findtex") {
    use std::collections::BTreeMap;
    let mut rows: Vec<(usize, usize, usize, f64, f64)> = Vec::new();
    for (i, t) in map.textures.iter().enumerate() {
        let Ok(rgba) = rustt::dxt::decode_rgba(t.w, t.h, t.fmt, &t.payload) else {
            continue;
        };
        let n = t.w * t.h;
        let mut black = 0usize;
        let mut color = 0usize;
        let mut ahi = 0usize;
        for px in rgba.chunks(4) {
            let (r, g, b, a) = (px[0] as i32, px[1] as i32, px[2] as i32, px[3] as i32);
            let mx = r.max(g).max(b);
            if mx < 12 {
                black += 1;
            }
            if mx - r.min(g).min(b) > 55 && mx > 30 {
                color += 1;
            }
            if a > 128 {
                ahi += 1;
            }
        }
        rows.push((i, t.w, t.h, black as f64 / n as f64, color as f64 / n as f64));
    }
    rows.sort_by(|a, b| {
        let ka = if a.3 > 0.4 { 1.0 } else { 0.0 };
        let kb = if b.3 > 0.4 { 1.0 } else { 0.0 };
        kb.partial_cmp(&ka)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal))
            .then((a.0).cmp(&b.0))
    });
    println!(
        "candidates (mostly-black textures with colorful pixels): tex, size, black%, colorful%"
    );
    let mut usage: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (pi, p) in map.render_parts.iter().enumerate() {
        let Some(mat) = map.materials.get(p.material) else {
            continue;
        };
        if let Some(s) = map.tex_slot(mat.tex_id) {
            usage.entry(s).or_default().push(pi);
        }
    }
    for (i, w, h, bf, cf) in rows.iter().take(30) {
        let fmt = match map.textures.get(*i).map(|t| t.fmt) {
            Some(rustt::ghg::TextureFmt::Dxt1) => "DXT1",
            Some(rustt::ghg::TextureFmt::Dxt5) => "DXT5",
            None => "??",
        };
        println!(
            "tex {i}: {w}x{h} {fmt} black={bf:.2} colorful={cf:.3} parts={}",
            usage.get(i).map(|v| v.len()).unwrap_or(0)
        );
        for pi in usage.get(i).map(|v| v.iter().take(3)).into_iter().flatten() {
            let p = &map.render_parts[*pi];
            let mat = &map.materials[p.material];
            let mlp = map.scene.material_list_ptr;
            let relv = i32::from_le_bytes(data[mlp + p.material * 4..mlp + p.material * 4 + 4].try_into().unwrap());
            let q = ((mlp + p.material * 4) as i64 + relv as i64) as usize;
            let o32 = |o: usize| {
                data.get(q + o..q + o + 4)
                    .map(|b| format!("0x{:08x}", u32::from_le_bytes(b.try_into().unwrap())))
                    .unwrap_or_else(|| "??".into())
            };
            let o16 = |o: usize| {
                data.get(q + o..q + o + 2)
                    .map(|b| format!("0x{:04x}", u16::from_le_bytes(b.try_into().unwrap())))
                    .unwrap_or_else(|| "??".into())
            };
            println!(
                "    part {pi}: mesh {} (0x{:x}) mat {} id={} texFlags=0x{:08x} vf=0x{:x} stage={} set={}",
                p.mesh,
                map.meshes.get(p.mesh).map(|m| m.address).unwrap_or(0),
                p.material,
                mat.id,
                mat.texture_flags,
                mat.vertex_format_bits,
                mat.lighting_stage,
                mat.lightmap_set_index
            );
            println!(
                "      rec@{q:#x}: +0x40:{} +0x44:{} +0x48:{} +0x4C:{} +0x50:{} +0x54:{} +0x58:{} +0x5C:{} +0xB0:{} +0xB4:{} +0x15C:{} +0x270:{} +0x26C:{} +0x2BC:{}",
                o32(0x40), o32(0x44), o32(0x48), o32(0x4C), o32(0x50), o32(0x54), o32(0x58), o32(0x5C),
                o32(0xB0), o32(0xB4), o16(0x15C), o32(0x270), o32(0x26C), o32(0x2BC)
            );
        }
    }
    return Ok(());
}

if let Some(_a) = args.iter().position(|x| x == "--alpha") {
    use std::collections::BTreeMap;
    let mut by: BTreeMap<(u8, u8), (usize, usize)> = BTreeMap::new();
    let mut mats_by_tex: BTreeMap<i16, Vec<usize>> = BTreeMap::new();
    for (i, m) in map.materials.iter().enumerate() {
        let k = (m.blend_mode(), m.depth_mode());
        let e = by.entry(k).or_insert((0, 0));
        e.0 += 1;
        e.1 += map
            .render_parts
            .iter()
            .filter(|p| p.material == i)
            .count();
        mats_by_tex.entry(m.tex_id).or_default().push(i);
    }
    println!("(blend, depth): materials, parts");
    for ((b, d), (nm, np)) in &by {
        println!("blend {b} depth {d}: mats={nm} parts={np}");
    }
    // materials whose alpha word has the alpha-cutoff format flag set
    let staged = map
        .materials
        .iter()
        .filter(|m| (m.alpha_type >> 0x14 & 0xff) == 5)
        .count();
    println!("materials with alpha-cutoff format flag (byte 5): {staged}/{}", map.materials.len());
    let _ = &mats_by_tex;
    return Ok(());
}

if let Some(a) = args.iter().position(|x| x == "--tex1") {
    let ti = args.get(a + 1).and_then(|s| s.parse::<usize>().ok()).context("--tex1 <index> [--texout <png>]")?;
    let t = map.textures.get(ti).context("texture index out of range")?;
    println!("tex {ti}: {}x{} payload={} bytes", t.w, t.h, t.payload.len());
    let rgba = rustt::dxt::decode_rgba(t.w, t.h, t.fmt, &t.payload)?;
    let n = (t.w * t.h).max(1);
    let (mut a0, mut a128, mut a250, mut black, mut vivid) = (0usize, 0usize, 0usize, 0usize, 0usize);
    for px in rgba.chunks(4) {
        match px[3] {
            0 => a0 += 1,
            v if v < 128 => a128 += 1,
            v if v >= 250 => a250 += 1,
            _ => {}
        }
        let (r, g, b) = (px[0] as i32, px[1] as i32, px[2] as i32);
        let mx = r.max(g).max(b);
        if mx < 12 {
            black += 1;
        }
        if mx - r.min(g).min(b) > 55 && mx > 30 {
            vivid += 1;
        }
    }
    println!(
        "alpha: 0={} ({:.1}%) 1..127={} ({:.1}%) >=250={} ({:.1}%) | black={:.1}% vivid={:.1}%",
        a0, a0 as f64 * 100.0 / n as f64,
        a128, a128 as f64 * 100.0 / n as f64,
        a250, a250 as f64 * 100.0 / n as f64,
        black as f64 * 100.0 / n as f64,
        vivid as f64 * 100.0 / n as f64
    );
    let out = args
        .iter()
        .position(|x| x == "--texout")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.clone());
    if let Some(path) = out {
        save_png(&path, t.w, t.h, &rgba)?;
        println!("saved {path}");
    }
    return Ok(());
}

if let Some(a) = args.iter().position(|x| x == "--hover") {
    // Props sitting above the floor: bounds y_min > miny with a small
    // footprint (< max_span). Catches machines/lamps on top of furniture.
    let (miny, max_span) = (
        args.get(a + 1).and_then(|s| s.parse::<f32>().ok()).unwrap_or(1.2),
        args.get(a + 2).and_then(|s| s.parse::<f32>().ok()).unwrap_or(2.0),
    );
    let mut hits: Vec<(f32, usize, [f32; 3], [f32; 3], [f32; 3])> = Vec::new();
    for (pi, part) in map.render_parts.iter().enumerate() {
        let Some(mesh) = map.meshes.get(part.mesh) else { continue };
        let Some(md) = rustt::mapmesh::expand_mesh(&map, mesh) else { continue };
        if md.pos.is_empty() {
            continue;
        }
        let mut mn = md.pos[0];
        let mut mx = md.pos[0];
        for p in &md.pos[1..] {
            for k in 0..3 {
                mn[k] = mn[k].min(p[k]);
                mx[k] = mx[k].max(p[k]);
            }
        }
        if mn[1] < miny {
            continue;
        }
        let span = [(mx[0] - mn[0]), (mx[1] - mn[1]), (mx[2] - mn[2])]
            .iter()
            .fold(0f32, |a, b| a.max(*b));
        if span > max_span {
            continue;
        }
        let c = [(mn[0] + mx[0]) * 0.5, (mn[1] + mx[1]) * 0.5, (mn[2] + mx[2]) * 0.5];
        hits.push((span, pi, c, mn, mx));
    }
    hits.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap());
    println!("props with y_min > {miny}, span <= {max_span}: {} hits", hits.len());
    for (span, pi, c, mn, mx) in &hits {
        let part = &map.render_parts[*pi];
        let Some(m) = map.materials.get(part.material) else { continue };
        let tex = map.tex_slot(m.tex_id);
        let texinfo = tex
            .and_then(|s| map.textures.get(s))
            .map(|t| format!("{}x{}", t.w, t.h))
            .unwrap_or("-".into());
        let fmt = tex
            .and_then(|s| map.textures.get(s))
            .map(|t| if t.fmt == rustt::ghg::TextureFmt::Dxt5 { "DXT5" } else { "DXT1" })
            .unwrap_or("-");
        println!(
            "part {:<4} span={:<5.2} c=({:6.2},{:5.2},{:6.2}) mat {:<3} id={:<3} tex={:<3} slot={:<3} {} {fmt:<4} bld={} dep={} defs=0x{:08x} light={} lm={} diff=({:.2},{:.2},{:.2},{:.2}) x[{:.2}..{:.2}] y[{:.2}..{:.2}] z[{:.2}..{:.2}]",
            pi, span, c[0], c[1], c[2],
            part.material, m.id, m.tex_id,
            tex.map(|t| t.to_string()).unwrap_or("-".into()),
            texinfo, m.blend_mode(), m.depth_mode(), m.shader_defines,
            m.lighting_stage, m.lightmap_stage(),
            m.diffuse[0], m.diffuse[1], m.diffuse[2], m.diffuse[3],
            mn[0], mx[0], mn[1], mx[1], mn[2], mx[2],
        );
    }
    return Ok(());
}

if let Some(a) = args.iter().position(|x| x == "--near") {
    let p = |k: usize| args.get(a + k).and_then(|s| s.parse::<f32>().ok());
    let (x, y, z, r) = match (p(1), p(2), p(3), p(4)) {
        (Some(x), Some(y), Some(z), Some(r)) => (x, y, z, r),
        _ => return Err(anyhow::anyhow!("usage: --near <x> <y> <z> <radius>")),
    };
    struct Hit {
        d: f32,
        part: usize,
        center: [f32; 3],
        lo: [f32; 3],
        hi: [f32; 3],
    }
    let mut hits: Vec<Hit> = Vec::new();
    for (pi, part) in map.render_parts.iter().enumerate() {
        let Some(mesh) = map.meshes.get(part.mesh) else { continue };
        let Some(md) = rustt::mapmesh::expand_mesh(&map, mesh) else { continue };
        if md.pos.is_empty() {
            continue;
        }
        let mut mn = md.pos[0];
        let mut mx = md.pos[0];
        for p in &md.pos[1..] {
            for k in 0..3 {
                mn[k] = mn[k].min(p[k]);
                mx[k] = mx[k].max(p[k]);
            }
        }
        let c = [
            (mn[0] + mx[0]) * 0.5,
            (mn[1] + mx[1]) * 0.5,
            (mn[2] + mx[2]) * 0.5,
        ];
        let d2 = (c[0] - x).powi(2) + (c[1] - y).powi(2) + (c[2] - z).powi(2);
        if d2 <= r * r {
            hits.push(Hit { d: d2.sqrt(), part: pi, center: c, lo: mn, hi: mx });
        }
    }
    hits.sort_by(|p, q| p.d.partial_cmp(&q.d).unwrap());
    println!("parts within {r} of ({x}, {y}, {z}): {} hits", hits.len());
    for h in &hits {
        let part = &map.render_parts[h.part];
        let Some(m) = map.materials.get(part.material) else { continue };
        let tex = map.tex_slot(m.tex_id);
        let texinfo = tex
            .and_then(|s| map.textures.get(s))
            .map(|t| format!("{}x{}", t.w, t.h));
        let fmt = tex
            .and_then(|s| map.textures.get(s))
            .map(|t| if t.fmt == rustt::ghg::TextureFmt::Dxt5 { "DXT5" } else { "DXT1" })
            .unwrap_or("-");
        let mesh = map.meshes.get(part.mesh);
        let cmod = mesh
            .and_then(|mm| {
                let md = rustt::mapmesh::expand_mesh(&map, mm)?;
                if md.color.is_empty() {
                    return None;
                }
                let mode = |k: usize| -> u8 {
                    let mut hist = [0usize; 256];
                    for c in &md.color {
                        hist[c[k] as usize] += 1;
                    }
                    hist.iter().enumerate().max_by_key(|&(_, &c)| c).map(|(v, _)| v as u8).unwrap_or(0)
                };
                Some((mode(0), mode(1), mode(2), mode(3)))
            })
            .map(|(r, g, b, a)| format!(" col=({r},{g},{b},{a})"))
            .unwrap_or_default();
        let (lo, hi) = (
            |k: usize| format!("{:.2}..{:.2}", h.lo[k], h.hi[k]),
            |k: usize| format!("{:.2}", h.hi[k] - h.lo[k]),
        );
        println!(
            "part {:<4} d={:<6.2} mat {:<3} id={:<3} tex={:<3} slot={:<3} {} {fmt:<4} bld={} dep={} alpha=0x{:08x} defs=0x{:08x} light={} lm={} uvsc=0x{:x} spec=({:.3},{:.3},{:.3},{:.3}){cmod} diff=({:.2},{:.2},{:.2},{:.2}) x[{},{}] y[{},{}] z[{},{}]",
            h.part, h.d, part.material, m.id, m.tex_id,
            tex.map(|t| t.to_string()).unwrap_or("-".into()),
            texinfo.as_deref().unwrap_or("-"),
            m.blend_mode(), m.depth_mode(), m.alpha_type, m.shader_defines,
            m.lighting_stage, m.lightmap_stage(), m.uv_set_coords,
            m.specular_params[0], m.specular_params[1], m.specular_params[2], m.specular_params[3],
            m.diffuse[0], m.diffuse[1], m.diffuse[2], m.diffuse[3],
            lo(0), hi(0), lo(1), hi(1), lo(2), hi(2),
        );
    }
    return Ok(());
}

if let Some(a) = args.iter().position(|x| x == "--lights") {
    let x: f32 = args.get(a + 1).and_then(|s| s.parse().ok()).context("--lights <x> <y> <z>")?;
    let y: f32 = args.get(a + 2).and_then(|s| s.parse().ok()).context("--lights <x> <y> <z>")?;
    let z: f32 = args.get(a + 3).and_then(|s| s.parse().ok()).context("--lights <x> <y> <z>")?;
    let rtl_path = path.replace(".GSC", ".RTL");
    // MAP_PC.GSC carries MAP.RTL (no platform tag); prefer the exact sibling
    // but fall back to the untagged name.
    let data = rustt::rtl::sibling_rtl_candidates(&path)
        .iter()
        .find_map(|p| std::fs::read(p).ok())
        .or_else(|| std::fs::read(&rtl_path).ok());
    for l in rustt::rtl::parse(data.as_deref().unwrap_or(&[])) {
        println!(
            "{} at ({:8.2},{:8.2},{:8.2}) dir ({:5.2},{:5.2},{:5.2}) col ({:5.2},{:5.2},{:5.2}) d={:7.2} fo={:7.2} m={:5.2}",
            format!("{:?}", l.kind),
            l.pos[0], l.pos[1], l.pos[2],
            l.dir[0], l.dir[1], l.dir[2],
            l.color[0], l.color[1], l.color[2],
            l.distance, l.falloff, l.multiplier
        );
    }
    let set = rustt::rtl::compute_light_set(&rustt::rtl::parse(data.as_deref().unwrap_or(&[])), [x, y, z]);
    println!("set for ({x:8.2},{y:8.2},{z:8.2}): ambient {:?}", set.scene_ambient);
    for i in 0..3 {
        println!("  light {i}: col {:?} pos {:?}", set.light_color[i], set.light_pos[i]);
    }
    return Ok(());
}

if let Some(a) = args.iter().position(|x| x == "--near") {
    let x: f32 = args.get(a + 1).and_then(|s| s.parse().ok()).context("--near <x> <y> <z> <radius>")?;
    let y: f32 = args.get(a + 2).and_then(|s| s.parse().ok()).context("--near <x> <y> <z> <radius>")?;
    let z: f32 = args.get(a + 3).and_then(|s| s.parse().ok()).context("--near <x> <y> <z> <radius>")?;
    let r: f32 = args.get(a + 4).and_then(|s| s.parse().ok()).context("--near <x> <y> <z> <radius>")?;
    for (i, part) in map.render_parts.iter().enumerate() {
        let Some(mesh) = map.meshes.get(part.mesh) else { continue };
        let Some(md) = rustt::mapmesh::expand_mesh(&map, mesh) else { continue };
        if md.pos.is_empty() {
            continue;
        }
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for p in &md.pos {
            for k in 0..3 {
                min[k] = min[k].min(p[k]);
                max[k] = max[k].max(p[k]);
            }
        }
        let cy = (min[0] + max[0]) * 0.5;
        let cz = (min[2] + max[2]) * 0.5;
        let hr = ((max[0] - min[0]).max(max[2] - min[2])) * 0.5;
        let dx = cy - x;
        let dz = cz - z;
        if dx * dx + dz * dz > (r + hr) * (r + hr) || !((min[1] - r) <= y && y <= (max[1] + r)) {
            continue;
        }
        let mut cmin = [255u8; 4];
        let mut cmax = [0u8; 4];
        for c in md.color.iter().take(16) {
            for k in 0..4 {
                cmin[k] = cmin[k].min(c[k]);
                cmax[k] = cmax[k].max(c[k]);
            }
        }
        match map.materials.get(part.material) {
            Some(m) => {
                let alpha = m.alpha_type;
                println!(
                    "part {i}: mesh {} @cx {cy:7.2} cz {cz:7.2} size {hr:5.2} | mat {} id={} tex={} diffuse=({:.1},{:.1},{:.1}) flags=0x{:08x} vf=0x{:08x} stride={} alpha=0x{:08x} blend={} depth={} defines=0x{:08x} stage={} prelit={} lmst={} set={} | vcol min={:02x}{:02x}{:02x} max={:02x}{:02x}{:02x}",
                    i,
                    part.mesh,
                    m.id,
                    m.tex_id,
                    m.diffuse[0], m.diffuse[1], m.diffuse[2],
                    m.texture_flags,
                    m.vertex_format_bits,
                    mesh.vertex_size,
                    alpha,
                    m.blend_mode(),
                    m.depth_mode(),
                    m.shader_defines,
                    m.lighting_stage,
                    m.shader_defines >> 12 & 1,
                    m.lightmap_stage(),
                    m.lightmap_set_index,
                    cmin[0], cmin[1], cmin[2],
                    cmax[0], cmax[1], cmax[2]
                );
            }
            None => {
                println!("part {i}: mesh {} @cx {cy:.2} cz {cz:.2} size {hr:.2} | (no material)", part.mesh);
            }
        }
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
            "mat {i}: id={} tex={} diff=({:.2},{:.2},{:.2},{:.2}) texFlags=0x{:08x} vformat=0x{:x}",
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
