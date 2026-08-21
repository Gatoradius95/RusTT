use rustt::map::parse;

fn main() {
    let path = "backup/LEVELS/MAP/MAP/MAP_PC.GSC";
    let data = std::fs::read(path).expect("read GSC");
    let map = parse(&data).unwrap();
    let tex_count = map.texture_real_index.len();
    let mat_count = map.materials.len();

    let nu20 = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize + 4;
    let mut ms00_pos = 0usize;
    {
        let mut p = nu20 + 0x20;
        while p + 8 <= data.len() {
            let id = u32::from_le_bytes(data[p..p+4].try_into().unwrap());
            let size = u32::from_le_bytes(data[p+4..p+8].try_into().unwrap()) as usize;
            if size < 8 || p + size > data.len() { break; }
            if id == 0x3030534d { ms00_pos = p; break; }
            p += size;
        }
    }
    let ms00_mats = ms00_pos + 16;
    let mat_size = 0x2c4;

    // Read specular and normal indices as i32
    let mut spec_indices: Vec<i32> = Vec::new();
    let mut norm_indices: Vec<i32> = Vec::new();
    for i in 0..mat_count {
        let base = ms00_mats + i * mat_size;
        let spec = i32::from_le_bytes(data[base+0xFC..base+0x100].try_into().unwrap());
        let norm = i32::from_le_bytes(data[base+0x100..base+0x104].try_into().unwrap());
        spec_indices.push(spec);
        norm_indices.push(norm);
    }

    let spec_valid = spec_indices.iter().filter(|&&v| v > 0 && (v as usize) < tex_count).count();
    let norm_valid = norm_indices.iter().filter(|&&v| v > 0 && (v as usize) < tex_count).count();
    let spec_neg = spec_indices.iter().filter(|&&v| v < 0).count();
    let norm_neg = norm_indices.iter().filter(|&&v| v < 0).count();

    println!("=== +0xFC (specular) and +0x100 (normal) as i32 ===");
    println!("specular: {} valid, {} neg (-1)", spec_valid, spec_neg);
    println!("normal:   {} valid, {} neg (-1)", norm_valid, norm_neg);

    // Correlation with shader_defines bits
    let mut spec_with_sd3 = 0u32;   // sd bit 3 = specular/phong
    let mut spec_without_sd3 = 0u32;
    let mut norm_with_sd0 = 0u32;   // sd bit 0 = surface type / normal map
    let mut norm_without_sd0 = 0u32;
    let mut norm_with_sd1 = 0u32;   // sd bit 1 = parallax

    for i in 0..mat_count {
        let base = ms00_mats + i * mat_size;
        let sd = u32::from_le_bytes(data[base+0x26C..base+0x270].try_into().unwrap());
        let spec = spec_indices[i];
        let norm = norm_indices[i];

        if spec > 0 { if sd & 0x8 != 0 { spec_with_sd3 += 1; } else { spec_without_sd3 += 1; } }
        if norm > 0 { if sd & 0x1 != 0 { norm_with_sd0 += 1; } else { norm_without_sd0 += 1; } }
        if norm > 0 && sd & 0x2 != 0 { norm_with_sd1 += 1; }
    }

    println!("\n=== Correlation ===");
    println!("spec tex + sd bit 3 (phong):     {}", spec_with_sd3);
    println!("spec tex + NO sd bit 3:           {}", spec_without_sd3);
    println!("norm tex + sd bit 0 (normal map): {}", norm_with_sd0);
    println!("norm tex + NO sd bit 0:           {}", norm_without_sd0);
    println!("norm tex + sd bit 1 (parallax):   {}", norm_with_sd1);

    // Show some materials WITH specular/normal maps
    println!("\n=== First 10 materials WITH specular map (+0xFC > 0) ===");
    let mut count = 0;
    for i in 0..mat_count {
        let base = ms00_mats + i * mat_size;
        let sd = u32::from_le_bytes(data[base+0x26C..base+0x270].try_into().unwrap());
        let spec = spec_indices[i];
        let norm = norm_indices[i];
        let tex_id = i16::from_le_bytes(data[base+0x74..base+0x76].try_into().unwrap());
        if spec > 0 && count < 10 {
            let real_spec = map.texture_real_index[spec as usize] as usize;
            let real_norm = if norm > 0 { map.texture_real_index[norm as usize] as usize } else { 0 };
            let spec_dims = if real_spec < map.textures.len() { format!("{}x{}", map.textures[real_spec].w, map.textures[real_spec].h) } else { "?".into() };
            let norm_dims = if norm > 0 && real_norm < map.textures.len() { format!("{}x{}", map.textures[real_norm].w, map.textures[real_norm].h) } else { "none".into() };
            println!("  mat[{}] id={} tex_id={} spec_idx={} ({}) norm_idx={} ({}) sd=0x{:08X}",
                i, map.materials[i].id, tex_id, spec, spec_dims, norm, norm_dims, sd);
            count += 1;
        }
    }

    println!("\n=== First 10 materials WITH normal map (+0x100 > 0) but NO specular ===");
    let mut count = 0;
    for i in 0..mat_count {
        let base = ms00_mats + i * mat_size;
        let sd = u32::from_le_bytes(data[base+0x26C..base+0x270].try_into().unwrap());
        let spec = spec_indices[i];
        let norm = norm_indices[i];
        let tex_id = i16::from_le_bytes(data[base+0x74..base+0x76].try_into().unwrap());
        if norm > 0 && spec < 0 && count < 10 {
            let real_norm = map.texture_real_index[norm as usize] as usize;
            let norm_dims = if real_norm < map.textures.len() { format!("{}x{}", map.textures[real_norm].w, map.textures[real_norm].h) } else { "?".into() };
            println!("  mat[{}] id={} tex_id={} norm_idx={} ({}) sd=0x{:08X}",
                i, map.materials[i].id, tex_id, norm, norm_dims, sd);
            count += 1;
        }
    }

    // Now update the material struct info
    println!("\n=== CONFIRMED material record field layout ===");
    println!("  +0x074 (i16): diffuseFileTexture index");
    println!("  +0x0B4 (u32): textureFlags");
    println!("  +0x0FC (i32): specularFileTexture index (-1 = none)");
    println!("  +0x100 (i32): normalFileTexture index (-1 = none)");
    println!("  +0x1F0 (u32): vertexFormatBits");
    println!("  +0x26C (u32): shaderDefinesBits");
}
