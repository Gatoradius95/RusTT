use crate::glb::MeshData;
use crate::map::{Map, Mesh};

/// Baked-light (RGBA8, 0..127 grayscale) byte offset within a map vertex,
/// per (stride, material vertex-format bits). Verified byte-for-byte against
/// MAP_PC.GSC for all 1847 meshes / 375 materials:
///
/// Layout families (pos is always the first 3 floats; the pre-uv fields are
/// byte-packed normals/`0x80`-centered, a `0x00..0x7F` grayscale scalar and a
/// 4-byte flags/scalar field — the *baked light* is the 0x7F-grayscale slot):
///  28 vf 0x02000901: pos + [scalar] + color + uv                    (+16)
///  28 vf 0x00000000: pos + [scalar] + color + uv                    (+16)
///  28 vf 0x02000129: pos + [nrm] + [nrm] + [flags] + color          (+24, no uv)
///  32 vf 0x02000909: pos + [nrm] + [flags] + color + uv             (+20)
///  36 vf 0x02000929: pos + [nrm] + [nrm] + [flags] + color + uv     (+24, uv@28)
///  36 vf 0x02001101: pos + [flags] + color + uv + uv                (+16)
///  40 vf 0x02001301: pos + [flags] + color + [pad] + uv + uv        (+16)
///  40 vf 0x06000909: pos + [nrm] + [flags] + color + uv + uv        (+20, AO)
///  48 vf 0x06001109: pos + [nrm] + [flags] + color + uv x3          (+20, AO)
///  52 vf 0x06001129: pos + [nrm] + [nrm] + [flags] + color + uv x3  (+24, AO)
///  56 vf 0x060013a9: pos + [nrm] + [nrm] + [flags] + color + pad + uv x3 (+24)
///  64 vf 0x06001b/ba9: like 56 with 4 uv sets                       (+24, AO)
pub fn color_offset(stride: usize, vfbits: u32) -> usize {
    match (stride, vfbits) {
        (28, 0x02000129) => 24,
        (32, _) => 20,
        (36, 0x02000929) => 24,
        (36, 0x02001101) => 16,
        (40, 0x02001301) => 16,
        (40, 0x06000909) => 20,
        (48, 0x06001109) => 20,
        (52, 0x06001129) => 24,
        (56, 0x060013a9) => 24,
        (64, 0x06001b29) | (64, 0x06001ba9) => 24,
        _ => 16,
    }
}

/// UV (diffuse set 0) byte offset within a map vertex, per (stride,
/// vertex-format bits). Same source as `color_offset`.
pub fn uv_offset(stride: usize, vfbits: u32) -> Option<usize> {
    match (stride, vfbits) {
        (28, _) => Some(20),
        (32, _) => Some(24),
        (36, 0x02000929) => Some(28),
        (36, _) => Some(20),
        (40, _) => Some(24),
        (48, _) => Some(24),
        (52, _) => Some(28),
        (56, _) => Some(32),
        (64, _) => Some(32),
        _ => None,
    }
}

/// Byte offset of UV set `set` within a map vertex. Map UV sets are
/// contiguous 2-float slots (`uv_offset + 8 * set`), which the documented
/// layouts confirm (e.g. 52 vf 0x06001129: uv x3 @28/36/44, 56: @32/40/48,
/// 64: 4 sets @32/40/48/56, 36/40: uv+uv @+8). Returns `None` when the set
/// would fall outside the stride.
pub fn uv_set_offset(stride: usize, vfbits: u32, set: usize) -> Option<usize> {
    let base = uv_offset(stride, vfbits)?;
    let off = base + set * 8;
    (off + 8 <= stride).then_some(off)
}

/// Expand every map mesh into CPU-side `MeshData`. Meshes that fail to expand
/// (missing buffers, unknown stride, empty triangles) are skipped.
pub fn expand_all(map: &Map) -> Vec<MeshData> {
    map.meshes
        .iter()
        .filter_map(|m| expand_mesh(map, m))
        .collect()
}

/// Expand a single triangle-strip mesh into `MeshData`. The index buffer holds
/// u16 triangle strips; like the game (and BrickBench's `GSCMesh`, which draws
/// with `setBaseVertex(vertexOffset)`), the real vertex id is
/// `index + vertex_offset`, absolute within the mesh's vertex buffer. Strips
/// are unwound on the CPU and vertices are deduplicated per mesh. `nrm` is
/// computed from geometry (area-weighted, like the ghg exporter). Skin data is
/// left empty: map vertices are rigid (identity skin matrix).
pub fn expand_mesh(map: &Map, m: &Mesh) -> Option<MeshData> {
    // mesh_type 0 ("none") is used by modded minifigs to hide specific parts.
    if m.mesh_type == 0 {
        return None;
    }
    let stride = m.vertex_size as usize;
    let mat = map
        .render_parts
        .iter()
        .find(|p| map.meshes.get(p.mesh).map(|mm| mm.address) == Some(m.address))
        .and_then(|p| map.materials.get(p.material));
    let vfbits = mat.map(|mat| mat.vertex_format_bits).unwrap_or(0);
    // Lightmap UV set offset (raw file UVs: u in [0..1], v in [-1..0];
    // u <= 0 selects the vertex-lit fallback in the shader). None for
    // materials without a lightmap stage -> the renderer leaves lm_uv empty.
    let lmuvo = mat
        .filter(|mat| mat.lightmap_stage() != 0)
        .and_then(|mat| {
            let declared = mat.lightmap_uvset() as usize;
            // The shipped vertex shader adds lightmapOffset to the highest UV
            // set the mesh actually declares (`USE_vs_uvSet3..uvSet0` chain),
            // so a material's declared lightmap set absent from this mesh's
            // layout falls back to the highest set that fits — never empty.
            (0..=declared)
                .rev()
                .find_map(|set| uv_set_offset(stride, vfbits, set))
        });
    let co = color_offset(stride, vfbits);
    let uvo = uv_offset(stride, vfbits)?;
    let vb = map.vertex_buffers.get(m.vertex_list_id as usize)?;
    let ib = map.index_buffers.get(m.index_list_id as usize)?;
    let vb_verts = vb.len() / stride;
    if stride == 0 || vb_verts == 0 {
        return None;
    }

    // Triangle strips store `triangle_count + 2` u16 indices. `index_offset` is
    // in elements (u16s), matching BrickBench's `GSCMesh.setBaseElement`, so the
    // byte offset is `index_offset * 2`. Index id is relative to the mesh's base
    // vertex `vertex_offset`.
    let idx_off = m.index_offset as usize * 2;
    let idx_count = m.triangle_count as usize + 2;
    if idx_off + idx_count * 2 > ib.len() {
        return None;
    }
    let rd = |k: usize| u16::from_le_bytes(ib[idx_off + k * 2..idx_off + k * 2 + 2].try_into().unwrap());
    let base = m.vertex_offset as usize;
    let mut ids: Vec<usize> = Vec::with_capacity(idx_count);
    for k in 0..idx_count {
        let id = rd(k) as usize + base;
        if id < vb_verts {
            ids.push(id);
        } else {
            ids.push(0);
        }
    }

    // Deduplicate per mesh: map absolute vertex ids to a compact local range
    // while pulling position, baked vertex color and UV out of the shared vertex
    // buffer. The color field (RGBA8, 0..127 scale) sits at `co` (per-family,
    // see `color_offset`); the original multiplies it into the textured surface
    // color (baked prelit light), so it must survive into the renderer.
    let mut pos: Vec<[f32; 3]> = Vec::new();
    let mut color: Vec<[u8; 4]> = Vec::new();
    let mut uv: Vec<[f32; 2]> = Vec::new();
    let mut lm_uv: Vec<[f32; 2]> = Vec::new();
    let mut remap: std::collections::HashMap<usize, u16> = std::collections::HashMap::new();
    let mut idx = Vec::with_capacity(idx_count - 2);
    #[allow(clippy::too_many_arguments)]
    let local = |id: usize, pos: &mut Vec<[f32; 3]>, color: &mut Vec<[u8; 4]>, uv: &mut Vec<[f32; 2]>, lm_uv: &mut Vec<[f32; 2]>, remap: &mut std::collections::HashMap<usize, u16>| -> u16 {
        if let Some(&l) = remap.get(&id) {
            return l;
        }
        let l = pos.len() as u16;
        let o = id * stride;
        pos.push([
            f32::from_le_bytes(vb[o..o + 4].try_into().unwrap()),
            f32::from_le_bytes(vb[o + 4..o + 8].try_into().unwrap()),
            f32::from_le_bytes(vb[o + 8..o + 12].try_into().unwrap()),
        ]);
        color.push(vb[o + co..o + co + 4].try_into().unwrap());
        let a = f32::from_le_bytes(vb[o + uvo..o + uvo + 4].try_into().unwrap());
        let b = f32::from_le_bytes(vb[o + uvo + 4..o + uvo + 8].try_into().unwrap());
        uv.push(if a.is_finite() && b.is_finite() { [a, b] } else { [0.0, 0.0] });
        if let Some(lo) = lmuvo {
            let c = f32::from_le_bytes(vb[o + lo..o + lo + 4].try_into().unwrap());
            let d = f32::from_le_bytes(vb[o + lo + 4..o + lo + 8].try_into().unwrap());
            // Shader gate: lm_uv.x <= 0 selects vertex-lit (lm_diffuse = 1).
            lm_uv.push(if c > 0.0 && c.is_finite() && d.is_finite() { [c, d] } else { [0.0, 0.0] });
        }
        remap.insert(id, l);
        l
    };
    for k in 2..idx_count {
        // Triangle (k-2, k-1, k), reversed on odd k to keep the winding of a
        // GPU triangle strip consistent.
        let (a, b, c) = if k % 2 == 0 {
            (ids[k - 2], ids[k - 1], ids[k])
        } else {
            (ids[k - 1], ids[k - 2], ids[k])
        };
        if a == b || b == c || a == c {
            continue;
        }
        let la = local(a, &mut pos, &mut color, &mut uv, &mut lm_uv, &mut remap);
        let lb = local(b, &mut pos, &mut color, &mut uv, &mut lm_uv, &mut remap);
        let lc = local(c, &mut pos, &mut color, &mut uv, &mut lm_uv, &mut remap);
        idx.push(la);
        idx.push(lb);
        idx.push(lc);
    }
    if idx.is_empty() {
        return None;
    }

    // Per-vertex normals from triangle winding (area-weighted).
    let n = pos.len();
    let mut nrm = vec![[0f32; 3]; n];
    for c in idx.chunks(3) {
        let a = c[0] as usize;
        let b = c[1] as usize;
        let d = c[2] as usize;
        let (abx, aby, abz) = (
            pos[b][0] - pos[a][0],
            pos[b][1] - pos[a][1],
            pos[b][2] - pos[a][2],
        );
        let (acx, acy, acz) = (
            pos[d][0] - pos[a][0],
            pos[d][1] - pos[a][1],
            pos[d][2] - pos[a][2],
        );
        let (nx, ny, nz) = (aby * acz - abz * acy, abz * acx - abx * acz, abx * acy - aby * acx);
        for v in [a, b, d] {
            nrm[v][0] += nx;
            nrm[v][1] += ny;
            nrm[v][2] += nz;
        }
    }
    for v in nrm.iter_mut() {
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if l > 1e-12 {
            v[0] /= l;
            v[1] /= l;
            v[2] /= l;
        } else {
            v[0] = 0.0;
            v[1] = 1.0;
            v[2] = 0.0;
        }
    }

    Some(MeshData {
        pos,
        nrm,
        uv,
        lm_uv,
        color,
        tangent: Vec::new(),
        idx,
        skin: Vec::new(),
        skin_bones: Vec::new(),
    })
}

/// Apply a 4×4 model matrix to every vertex position and normal in the mesh.
///
/// The matrix is stored row-major: `m[row][col]`. Positions get the full
/// affine transform (rows 0–2, column 3 = translation). Normals use only the
/// upper-left 3×3 rotation/scale (ignoring translation).
pub fn apply_transform(md: &mut MeshData, m: &[[f32; 4]; 4]) {
    // Fast-path: skip identity matrices (avoids a full scan of every mesh).
    if m[0][0] == 1.0 && m[0][1] == 0.0 && m[0][2] == 0.0 && m[0][3] == 0.0
        && m[1][0] == 0.0 && m[1][1] == 1.0 && m[1][2] == 0.0 && m[1][3] == 0.0
        && m[2][0] == 0.0 && m[2][1] == 0.0 && m[2][2] == 1.0 && m[2][3] == 0.0
        && m[3][0] == 0.0 && m[3][1] == 0.0 && m[3][2] == 0.0 && m[3][3] == 1.0
    {
        return;
    }
    for p in &mut md.pos {
        let (x, y, z) = (p[0], p[1], p[2]);
        p[0] = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3];
        p[1] = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3];
        p[2] = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3];
    }
    for n in &mut md.nrm {
        let (x, y, z) = (n[0], n[1], n[2]);
        n[0] = m[0][0] * x + m[0][1] * y + m[0][2] * z;
        n[1] = m[1][0] * x + m[1][1] * y + m[1][2] * z;
        n[2] = m[2][0] * x + m[2][1] * y + m[2][2] * z;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_stride_offsets() {
        assert_eq!(uv_offset(28, 0x02000901), Some(20));
        assert_eq!(uv_offset(28, 0x02000129), Some(20));
        assert_eq!(uv_offset(32, 0x02000909), Some(24));
        assert_eq!(uv_offset(36, 0x02001101), Some(20));
        assert_eq!(uv_offset(36, 0x02000929), Some(28));
        assert_eq!(uv_offset(40, 0x02001301), Some(24));
        assert_eq!(uv_offset(40, 0x06000909), Some(24));
        assert_eq!(uv_offset(48, 0x06001109), Some(24));
        assert_eq!(uv_offset(52, 0x06001129), Some(28));
        assert_eq!(uv_offset(56, 0x060013a9), Some(32));
        assert_eq!(uv_offset(64, 0x06001ba9), Some(32));
        assert_eq!(uv_offset(30, 0), None);
    }

    #[test]
    fn known_color_offsets() {
        assert_eq!(color_offset(28, 0x02000901), 16);
        assert_eq!(color_offset(28, 0x02000129), 24);
        assert_eq!(color_offset(32, 0x02000909), 20);
        assert_eq!(color_offset(36, 0x02001101), 16);
        assert_eq!(color_offset(36, 0x02000929), 24);
        assert_eq!(color_offset(40, 0x02001301), 16);
        assert_eq!(color_offset(40, 0x06000909), 20);
        assert_eq!(color_offset(48, 0x06001109), 20);
        assert_eq!(color_offset(52, 0x06001129), 24);
        assert_eq!(color_offset(56, 0x060013a9), 24);
        assert_eq!(color_offset(64, 0x06001ba9), 24);
        assert_eq!(color_offset(64, 0x06001b29), 24);
        assert_eq!(color_offset(30, 0), 16);
    }
}
