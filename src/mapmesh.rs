use crate::glb::MeshData;
use crate::map::{Map, Mesh};

/// UV (diffuse set 0) byte offset within a map vertex, per vertex stride
/// (bytes). Verified against MAP_PC.GSC across all 1847 meshes.
///
/// Layout families (pos is always the first 3 floats):
///  28: pos + color + nrm + uv                       (1 set)
///  32: pos + color + flags + nrm + uv               (1 set)
///  36: pos + color + nrm + uv + uv                  (2 sets)
///  40: pos + color + flags + nrm + uv + uv          (2 sets)
///  48: pos + color + flags + nrm + uv x3            (3 sets)
///  52: pos + color + ? + flags + nrm + uv x3        (3 sets)
///  56: pos + color + ? + flags + nrm + pad + uv x3  (3 sets)
///  64: pos + color + ? + flags + nrm + pad + uv x4  (4 sets)
pub fn uv_offset(stride: usize) -> Option<usize> {
    match stride {
        28 => Some(20),
        32 => Some(24),
        36 => Some(20),
        40 => Some(24),
        48 => Some(24),
        52 => Some(28),
        56 => Some(32),
        64 => Some(32),
        _ => None,
    }
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
    let stride = m.vertex_size as usize;
    let uvo = uv_offset(stride)?;
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
    // while pulling position and UV out of the shared vertex buffer.
    let mut pos: Vec<[f32; 3]> = Vec::new();
    let mut uv: Vec<[f32; 2]> = Vec::new();
    let mut remap: std::collections::HashMap<usize, u16> = std::collections::HashMap::new();
    let mut idx = Vec::with_capacity(idx_count - 2);
    let mut local = |id: usize, pos: &mut Vec<[f32; 3]>, uv: &mut Vec<[f32; 2]>, remap: &mut std::collections::HashMap<usize, u16>| -> u16 {
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
        let a = f32::from_le_bytes(vb[o + uvo..o + uvo + 4].try_into().unwrap());
        let b = f32::from_le_bytes(vb[o + uvo + 4..o + uvo + 8].try_into().unwrap());
        uv.push(if a.is_finite() && b.is_finite() { [a, b] } else { [0.0, 0.0] });
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
        let la = local(a, &mut pos, &mut uv, &mut remap);
        let lb = local(b, &mut pos, &mut uv, &mut remap);
        let lc = local(c, &mut pos, &mut uv, &mut remap);
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
        idx,
        skin: Vec::new(),
        skin_bones: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_stride_offsets() {
        assert_eq!(uv_offset(28), Some(20));
        assert_eq!(uv_offset(32), Some(24));
        assert_eq!(uv_offset(36), Some(20));
        assert_eq!(uv_offset(40), Some(24));
        assert_eq!(uv_offset(48), Some(24));
        assert_eq!(uv_offset(52), Some(28));
        assert_eq!(uv_offset(56), Some(32));
        assert_eq!(uv_offset(64), Some(32));
        assert_eq!(uv_offset(30), None);
    }
}
