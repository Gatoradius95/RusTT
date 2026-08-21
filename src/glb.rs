use anyhow::{Context, Result};
use png::BitDepth;

use crate::dxt;
use crate::ghg::{Parsed, TextureFmt, uv_offset};

/// Flip textures vertically to convert D3D (top-down) storage to GL/glTF sampling.
const FLIP_TEXTURE_V: bool = true;

#[derive(Clone)]
pub struct MeshData {
    pub pos: Vec<[f32; 3]>,
    pub nrm: Vec<[f32; 3]>,
    pub uv: Vec<[f32; 2]>,
    pub idx: Vec<u16>,
    /// Per-vertex skin block, 8 bytes per vertex: 4 weights (u8) followed by
    /// 4 local bone indices (u8; 255 = no influence). Empty when the part
    /// carries no per-vertex skinning.
    pub skin: Vec<u8>,
    /// Global skin bones of the part (descriptor +0x0a). Empty when the part
    /// is rigidly bound to its render item's single bone.
    pub skin_bones: Vec<u16>,
}

pub fn build_meshes(p: &Parsed) -> Vec<MeshData> {
    let mut out = Vec::with_capacity(p.render.len());
    for item in &p.render {
        out.push(build_mesh(p, item.part));
    }
    out
}

pub fn build_mesh(p: &Parsed, part_idx: usize) -> MeshData {
    let empty = MeshData {
        pos: Vec::new(),
        nrm: Vec::new(),
        uv: Vec::new(),
        idx: Vec::new(),
        skin: Vec::new(),
        skin_bones: Vec::new(),
    };
    let part = match p.parts.get(part_idx) {
        Some(x) => x,
        None => return empty,
    };
    if part.num_v == 0 || part.num_i < 3 {
        return empty;
    }
    let vl = match p.vertex_lists.get(part.vl) {
        Some(x) => *x,
        None => return empty,
    };
    let base = part.off_v * part.stride;
    let n = part.num_v;
    if base + n * part.stride > vl.len() {
        return empty;
    }
    let uvo = uv_offset(part.stride).or_else(|| scan_uv(vl, base, n, part.stride));

    let mut pos = Vec::with_capacity(n);
    let mut uv = Vec::with_capacity(n);
    let mut got_uv = false;
    // Per-vertex skin block offsets: (weights, indices) by stride.
    // Stride 44: [pos 12][nrm 12][pad 4][uv 8][skin 8] -> skin at 36.
    // Stride 40: [pos 12][nrm 12][uv 8][skin 8] -> skin at 32.
    let (w_off, i_off) = match part.stride {
        44 => (Some(36), Some(40)),
        40 => (Some(32), Some(36)),
        _ => (None, None),
    };
    let mut skin = Vec::with_capacity(if w_off.is_some() { n * 8 } else { 0 });
    for v in 0..n {
        let o = base + v * part.stride;
        pos.push([
            f32::from_le_bytes(vl[o..o + 4].try_into().unwrap()),
            f32::from_le_bytes(vl[o + 4..o + 8].try_into().unwrap()),
            f32::from_le_bytes(vl[o + 8..o + 12].try_into().unwrap()),
        ]);
        if let (Some(w), Some(i)) = (w_off, i_off) {
            skin.extend_from_slice(&vl[o + w..o + w + 4]);
            skin.extend_from_slice(&vl[o + i..o + i + 4]);
        }
        if let Some(u) = uvo {
            let a = f32::from_le_bytes(vl[o + u..o + u + 4].try_into().unwrap());
            let b = f32::from_le_bytes(vl[o + u + 4..o + u + 8].try_into().unwrap());
            if a.is_finite() && b.is_finite() {
                uv.push([a, b]);
                got_uv = true;
            } else {
                uv.push([0.0, 0.0]);
            }
        } else {
            uv.push([0.0, 0.0]);
        }
    }

    let il = match p.index_lists.get(part.il) {
        Some(x) => *x,
        None => return empty,
    };
    let istart = part.off_i * 2;
    if istart + part.num_i * 2 > il.len() {
        return empty;
    }
    let mut idx = Vec::new();
    let mut tri = |a: usize, b: usize, c: usize| {
        if a < n && b < n && c < n && a != b && b != c && a != c {
            idx.push(a as u16);
            idx.push(b as u16);
            idx.push(c as u16);
        }
    };
    // Index lists store u16 indices. `off_i` is a u16 offset, so `istart`
    // points at the first byte of the part's indices.
    let rd_u16 = |byte_off: usize| -> usize {
        u16::from_le_bytes(il[byte_off..byte_off + 2].try_into().unwrap()) as usize
    };
    for k in 2..part.num_i {
        let i = rd_u16(istart + k * 2);
        let j = rd_u16(istart + (k - 1) * 2);
        let h = rd_u16(istart + (k - 2) * 2);
        if k % 2 == 0 {
            tri(h, j, i);
        } else {
            tri(j, h, i);
        }
    }

    // per-vertex normals from triangle winding (area-weighted)
    let mut nrm = vec![[0f32; 3]; n];
    for c in idx.chunks(3) {
        let a = c[0] as usize;
        let b = c[1] as usize;
        let d = c[2] as usize;
        if a >= n || b >= n || d >= n {
            continue;
        }
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
            *v = [0.0, 1.0, 0.0];
        }
    }
    let _ = got_uv;
    MeshData {
        pos,
        nrm,
        uv,
        idx,
        skin,
        skin_bones: part.skin_bones.iter().map(|&b| b as u16).collect(),
    }
}

fn scan_uv(vl: &[u8], base: usize, n: usize, stride: usize) -> Option<usize> {
    let max = stride.saturating_sub(8);
    'off: for u in (0..=max).step_by(4) {
        if u + 8 > stride {
            continue;
        }
        let mut good = 0;
        for v in 0..n {
            let o = base + v * stride + u;
            if o + 8 > vl.len() {
                break;
            }
            let a = f32::from_le_bytes(vl[o..o + 4].try_into().unwrap());
            let b = f32::from_le_bytes(vl[o + 4..o + 8].try_into().unwrap());
            if a.is_finite() && b.is_finite() && a.abs() <= 2.0 && b.abs() <= 2.0 {
                good += 1;
            } else {
                continue 'off;
            }
        }
        if good == n {
            return Some(u);
        }
    }
    None
}

fn align4(buf: &mut Vec<u8>) {
    while buf.len() % 4 != 0 {
        buf.push(0);
    }
}

struct Bv {
    byte_offset: usize,
    byte_length: usize,
}

fn encode_png(texture: &crate::ghg::Texture) -> Result<Vec<u8>> {
    let rgba = dxt::decode(texture)?;
    let w = texture.w as u32;
    let h = texture.h as u32;
    let mut buf = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut buf, w, h);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(BitDepth::Eight);
        let mut wr = enc.write_header().context("png header")?;
        if FLIP_TEXTURE_V {
            let row = w as usize * 4;
            let mut flipped = Vec::with_capacity(rgba.len());
            for y in (0..h as usize).rev() {
                flipped.extend_from_slice(&rgba[y * row..(y + 1) * row]);
            }
            wr.write_image_data(&flipped).context("png data")?;
        } else {
            wr.write_image_data(&rgba).context("png data")?;
        }
        wr.finish().context("png finish")?;
    }
    Ok(buf)
}

fn jf(v: f32) -> String {
    if v == v.trunc() && v.abs() < 1e6 {
        format!("{:.0}", v)
    } else {
        format!("{}", v)
    }
}

pub fn build_glb(p: &Parsed, meshes: &[MeshData]) -> Result<Vec<u8>> {
    // encode textures
    let mut pngs = Vec::with_capacity(p.textures.len());
    for t in &p.textures {
        pngs.push(encode_png(t)?);
    }

    let n_mesh = meshes.len();
    let n_tex = pngs.len();

    // BIN assembly
    let mut bin: Vec<u8> = Vec::new();
    let mut bvs: Vec<Bv> = Vec::new();
    let mut v_off = Vec::with_capacity(n_mesh);
    let mut i_off = Vec::with_capacity(n_mesh);
    for m in meshes {
        align4(&mut bin);
        v_off.push(bin.len());
        for v in 0..m.pos.len() {
            bin.extend_from_slice(&m.pos[v][0].to_le_bytes());
            bin.extend_from_slice(&m.pos[v][1].to_le_bytes());
            bin.extend_from_slice(&m.pos[v][2].to_le_bytes());
            bin.extend_from_slice(&m.nrm[v][0].to_le_bytes());
            bin.extend_from_slice(&m.nrm[v][1].to_le_bytes());
            bin.extend_from_slice(&m.nrm[v][2].to_le_bytes());
            bin.extend_from_slice(&m.uv[v][0].to_le_bytes());
            bin.extend_from_slice(&m.uv[v][1].to_le_bytes());
        }
        bvs.push(Bv {
            byte_offset: v_off[v_off.len() - 1],
            byte_length: m.pos.len() * 32,
        });
        align4(&mut bin);
        i_off.push(bin.len());
        for i in &m.idx {
            bin.extend_from_slice(&i.to_le_bytes());
        }
        bvs.push(Bv {
            byte_offset: i_off[i_off.len() - 1],
            byte_length: m.idx.len() * 2,
        });
    }
    let mut img_off = Vec::with_capacity(n_tex);
    for png in &pngs {
        align4(&mut bin);
        img_off.push(bin.len());
        bin.extend_from_slice(png);
        bvs.push(Bv {
            byte_offset: img_off[img_off.len() - 1],
            byte_length: png.len(),
        });
    }

    // JSON
    let mut j = String::new();
    j.push_str("{\"asset\":{\"version\":\"2.0\",\"generator\":\"rustt\"},");
    j.push_str(&format!("\"scene\":0,\"scenes\":[{{\"nodes\":[0]}}],"));
    // nodes
    j.push_str("\"nodes\":[");
    if n_mesh == 0 {
        j.push_str("{}");
    } else {
        j.push_str("{\"name\":\"root\",\"children\":[");
        for i in 1..=n_mesh {
            j.push_str(&format!("{i},"));
        }
        if n_mesh > 1 {
            j.truncate(j.len() - 1);
        }
        j.push_str("]}");
        for i in 0..n_mesh {
            j.push_str(&format!(",{{\"name\":\"Part {i}\",\"mesh\":{i}}}"));
        }
    }
    j.push_str("],");

    // meshes
    j.push_str("\"meshes\":[");
    for i in 0..n_mesh {
        if i > 0 {
            j.push(',');
        }
        let a0 = 4 * i;
        let mat = mesh_material(p, i);
        j.push_str(&format!(
            "{{\"primitives\":[{{\"attributes\":{{\"POSITION\":{a0},\"NORMAL\":{},\"TEXCOORD_0\":{}}},\"indices\":{},\"material\":{mat}}}]}}",
            a0 + 1,
            a0 + 2,
            a0 + 3
        ));
    }
    j.push_str("],");

    // materials
    j.push_str("\"materials\":[");
    for (i, m) in p.materials.iter().enumerate() {
        if i > 0 {
            j.push(',');
        }
        let mut s = String::from("{\"pbrMetallicRoughness\":{");
        // The float diffuse alpha in ghg data is a leftover lighting value
        // (often 0.5 even for fully opaque parts; the byte alpha in `rgba` is
        // 255). Keep baseColor opaque and let DXT5-textured materials carry
        // real alpha through BLEND instead.
        s.push_str("\"baseColorFactor\":[");
        s.push_str(&jf(m.diffuse[0].clamp(0.0, 1.0)));
        s.push(',');
        s.push_str(&jf(m.diffuse[1].clamp(0.0, 1.0)));
        s.push(',');
        s.push_str(&jf(m.diffuse[2].clamp(0.0, 1.0)));
        s.push_str(",1.0]");
        s.push_str(",\"metallicFactor\":0.0,\"roughnessFactor\":1.0");
        if m.tex_id >= 0 && (m.tex_id as usize) < n_tex {
            s.push_str(",\"baseColorTexture\":{\"index\":");
            s.push_str(&format!("{}", m.tex_id));
            s.push('}');
        }
        s.push('}');
        let dxt5 = m.tex_id >= 0
            && (m.tex_id as usize) < n_tex
            && p.textures[m.tex_id as usize].fmt == TextureFmt::Dxt5;
        if dxt5 {
            s.push_str(",\"alphaMode\":\"BLEND\"");
        }
        s.push('}');
        j.push_str(&s);
    }
    j.push_str("],");

    // textures / images / samplers
    j.push_str("\"textures\":[");
    for i in 0..n_tex {
        if i > 0 {
            j.push(',');
        }
        j.push_str(&format!("{{\"sampler\":0,\"source\":{i}}}"));
    }
    j.push_str("],");
    j.push_str("\"images\":[");
    for i in 0..n_tex {
        if i > 0 {
            j.push(',');
        }
        j.push_str(&format!(
            "{{\"mimeType\":\"image/png\",\"bufferView\":{}}}",
            2 * n_mesh + i
        ));
    }
    j.push_str("],");
    j.push_str("\"samplers\":[{\"magFilter\":9729,\"minFilter\":9987,\"wrapS\":10497,\"wrapT\":10497}],");

    // accessors
    j.push_str("\"accessors\":[");
    for i in 0..n_mesh {
        if i > 0 {
            j.push(',');
        }
        let m = &meshes[i];
        let bvv = 2 * i;
        let mut mn = [0f32; 3];
        let mut mx = [0f32; 3];
        if !m.pos.is_empty() {
            mn = m.pos[0];
            mx = m.pos[0];
            for p in &m.pos {
                for k in 0..3 {
                    if p[k] < mn[k] {
                        mn[k] = p[k];
                    }
                    if p[k] > mx[k] {
                        mx[k] = p[k];
                    }
                }
            }
        }
        j.push_str(&format!(
            "{{\"bufferView\":{bvv},\"byteOffset\":0,\"componentType\":5126,\"count\":{},\"type\":\"VEC3\",\"min\":[{},{},{}],\"max\":[{},{},{}]}}",
            m.pos.len(),
            jf(mn[0]),
            jf(mn[1]),
            jf(mn[2]),
            jf(mx[0]),
            jf(mx[1]),
            jf(mx[2])
        ));
        j.push(',');
        j.push_str(&format!(
            "{{\"bufferView\":{bvv},\"byteOffset\":12,\"componentType\":5126,\"count\":{},\"type\":\"VEC3\"}}",
            m.nrm.len()
        ));
        j.push(',');
        j.push_str(&format!(
            "{{\"bufferView\":{bvv},\"byteOffset\":24,\"componentType\":5126,\"count\":{},\"type\":\"VEC2\"}}",
            m.uv.len()
        ));
        j.push(',');
        j.push_str(&format!(
            "{{\"bufferView\":{},\"byteOffset\":0,\"componentType\":5123,\"count\":{},\"type\":\"SCALAR\"}}",
            2 * i + 1,
            m.idx.len()
        ));
    }
    j.push_str("],");

    // bufferViews
    j.push_str("\"bufferViews\":[");
    for (i, bv) in bvs.iter().enumerate() {
        if i > 0 {
            j.push(',');
        }
        let target = if i < 2 * n_mesh {
            if i % 2 == 0 {
                34962
            } else {
                34963
            }
        } else {
            0
        };
        if target == 0 {
            j.push_str(&format!(
                "{{\"buffer\":0,\"byteOffset\":{},\"byteLength\":{}}}",
                bv.byte_offset, bv.byte_length
            ));
        } else if target == 34962 {
            j.push_str(&format!(
                "{{\"buffer\":0,\"byteOffset\":{},\"byteLength\":{},\"byteStride\":32,\"target\":{target}}}",
                bv.byte_offset, bv.byte_length
            ));
        } else {
            j.push_str(&format!(
                "{{\"buffer\":0,\"byteOffset\":{},\"byteLength\":{},\"target\":{target}}}",
                bv.byte_offset, bv.byte_length
            ));
        }
    }
    j.push_str("],");
    j.push_str(&format!(
        "\"buffers\":[{{\"byteLength\":{}}}]",
        bin.len()
    ));
    j.push_str("}");

    // GLB container
    let json = j.into_bytes();
    let json_len = pad4(json.len());
    let bin_len = pad4(bin.len());
    let total = 12 + 8 + json_len + 8 + bin_len;
    let mut glb = Vec::with_capacity(total);
    glb.extend_from_slice(b"glTF");
    glb.extend_from_slice(&2u32.to_le_bytes());
    glb.extend_from_slice(&(total as u32).to_le_bytes());
    glb.extend_from_slice(&(json_len as u32).to_le_bytes());
    glb.extend_from_slice(&0x4e4f534au32.to_le_bytes()); // "JSON"
    glb.extend_from_slice(&json);
    while glb.len() < 20 + json_len {
        glb.push(0x20);
    }
    glb.extend_from_slice(&(bin_len as u32).to_le_bytes());
    glb.extend_from_slice(&0x004e4942u32.to_le_bytes()); // "BIN\0"
    glb.extend_from_slice(&bin);
    while glb.len() < total {
        glb.push(0);
    }
    Ok(glb)
}

fn mesh_material(p: &Parsed, i: usize) -> i32 {
    match p.render.get(i) {
        Some(item) if item.mat >= 0 && (item.mat as usize) < p.materials.len() => item.mat,
        _ => 0,
    }
}

fn pad4(n: usize) -> usize {
    (n + 3) & !3
}
