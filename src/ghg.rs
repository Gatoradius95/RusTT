use anyhow::{ensure, Context, Result};
use glam::Mat4;

pub struct Parsed<'a> {
    pub materials: Vec<Material>,
    pub textures: Vec<Texture>,
    pub parts: Vec<Part>,
    pub render: Vec<RenderItem>,
    /// Layer index of each render item (parallel to `render`). Layers are
    /// LOD/quality variants: the game draws only the layers selected by the
    /// sibling `.TXT` (`layers_special`, `layers_high`, ...).
    pub render_layer: Vec<u32>,
    pub bones: Vec<Bone>,
    pub vertex_lists: Vec<&'a [u8]>,
    pub index_lists: Vec<&'a [u8]>,
}

pub struct Material {
    pub id: i32,
    pub diffuse: [f32; 4],
    pub tex_id: i16,
    pub rgba: [u8; 4],
}

pub struct Texture {
    pub w: usize,
    pub h: usize,
    pub fmt: TextureFmt,
    pub payload: Vec<u8>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TextureFmt {
    Dxt1,
    Dxt5,
}

pub struct Part {
    pub stride: usize,
    pub off_v: usize,
    pub num_v: usize,
    pub off_i: usize,
    pub num_i: usize,
    pub il: usize,
    pub vl: usize,
    /// Skin bones from the part descriptor at +0x0a (0xff-terminated, local
    /// indices into this list appear in the vertex skin block). Empty when the
    /// part is rigidly bound to a single render-item bone.
    pub skin_bones: Vec<u8>,
    /// Shape keys ("dynamic buffers"): per slot, `num_v` per-vertex [x,y,z]
    /// offsets from the base pose. `None` for an empty slot (pointer 0).
    /// Driven by BSA channel weights, one shape per channel (by index).
    pub dynamic_buffers: Vec<Option<Vec<[f32; 3]>>>,
}

pub struct RenderItem {
    pub part: usize,
    pub mat: i32,
    pub bone: i32,
}

pub struct Bone {
    pub name: String,
    pub parent: i32,
    /// identity matrix (first matrix per bone) — used for AN3 0x20 rotations.
    pub identity: Mat4,
    /// local matrix (consecutive block after the bone structs) — the model
    /// rest pose world skeleton.
    pub local: Mat4,
    /// identity * local (world) for the model's own bind pose.
    pub world: Mat4,
    /// the `abs_bones3` world-rest matrix (wide-stance) — the true model
    /// binding skeleton.
    pub bind: Mat4,
}

#[inline]
fn at_i32(d: &[u8], o: usize) -> i32 {
    i32::from_le_bytes(d[o..o + 4].try_into().unwrap())
}

#[inline]
fn at_u32(d: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(d[o..o + 4].try_into().unwrap())
}

#[inline]
fn at_u16(d: &[u8], o: usize) -> u16 {
    u16::from_le_bytes(d[o..o + 2].try_into().unwrap())
}

/// readInt at offset q, then seek(value - 4) relative to the post-read position
/// yields q + value (signed math).
fn rel(d: &[u8], q: i64) -> Result<i64> {
    let qq = usize::try_from(q).context("negative offset")?;
    Ok(q + at_i32(d, qq) as i64)
}

fn check_range(d: &[u8], o: usize, n: usize) -> Result<()> {
    ensure!(o + n <= d.len(), "read past end (offset {o:#x}, len {n})");
    Ok(())
}

pub fn parse(data: &[u8]) -> Result<Parsed<'_>> {
    let num20 = at_i32(data, 0);
    ensure!(
        num20 != 0x3032554e,
        "Batman variant (NU20 first) is not supported by this exporter"
    );
    ensure!(
        num20 > 0,
        "unrecognized file: first int 0x{:08x}",
        num20 as u32
    );
    let header = num20 as i64;

    // Skip to HEAD chunk.
    let head = header + 4 + 4 + 0xc;
    let mut p = head as usize;
    check_range(data, p, 8)?;
    let _fourcc = at_u32(data, p);
    let chunk_size = at_u32(data, p + 4) as usize;
    let _abs_pntr = p as i64 + 8 + at_i32(data, p + 8) as i64;
    let abs_gsnh = p as i64 + 16 + at_i32(data, p + 12) as i64;
    p += 16;
    ensure!(
        chunk_size >= 8,
        "HEAD chunk size too small ({chunk_size})"
    );

    // NTBL
    check_range(data, p, 8)?;
    let _fourcc = at_u32(data, p);
    let chunk_size = at_u32(data, p + 4) as usize;
    ensure!(chunk_size >= 8, "NTBL chunk size too small ({chunk_size})");
    p += chunk_size;

    // optional TREF / TST0
    loop {
        check_range(data, p, 8)?;
        let fourcc = at_u32(data, p);
        if fourcc == 1178948180 || fourcc == 810832724 {
            let chunk_size = at_u32(data, p + 4) as usize;
            ensure!(chunk_size >= 8, "chunk size too small ({chunk_size})");
            p += chunk_size;
        } else {
            break;
        }
    }

    // MS00 materials
    check_range(data, p, 8)?;
    let fourcc = at_u32(data, p);
    ensure!(fourcc == 0x3030534d, "expected MS00 at {p:#x}, got {fourcc:#08x}");
    let chunk_size = at_u32(data, p + 4) as usize;
    let number_materials = at_i32(data, p + 8) as usize;
    p += 16;
    let mut materials = Vec::with_capacity(number_materials);
    for _ in 0..number_materials {
        check_range(data, p, 0x38 + 4 + 0x18 + 16 + 0x10 + 2 + 0x52 + 4 + 0x1f8)?;
        let mut q = p;
        q += 0x38;
        let id = at_i32(data, q);
        q += 4 + 0x18;
        let diffuse = [
            f32::from_le_bytes(data[q..q + 4].try_into().unwrap()),
            f32::from_le_bytes(data[q + 4..q + 8].try_into().unwrap()),
            f32::from_le_bytes(data[q + 8..q + 12].try_into().unwrap()),
            f32::from_le_bytes(data[q + 12..q + 16].try_into().unwrap()),
        ];
        q += 16 + 0x10;
        let tex_id = i16::from_le_bytes(data[q..q + 2].try_into().unwrap());
        q += 2 + 0x52;
        let rgba = [data[q], data[q + 1], data[q + 2], data[q + 3]];
        materials.push(Material {
            id,
            diffuse,
            tex_id,
            rgba,
        });
        p += 0x38 + 4 + 0x18 + 16 + 0x10 + 2 + 0x52 + 4 + 0x1f8;
    }
    let _ = chunk_size;

    // GSNH
    let gsnh = (abs_gsnh - 12) as usize;
    check_range(data, gsnh, 20)?;
    let _fourcc = at_u32(data, gsnh);
    let _chunk_size = at_u32(data, gsnh + 4);
    let number_images = at_i32(data, gsnh + 12) as usize;
    let images_meta = gsnh as i64 + 16 + at_i32(data, gsnh + 16) as i64;
    let mesh_meta = rel(data, gsnh as i64 + 16 + 4 + 0x28)?;
    let bones_base = gsnh as i64 + 16 + 4 + 0x28 + 4 + 0x130;
    let number_bones = at_i32(data, bones_base as usize) as usize;
    let abs_bones = rel(data, bones_base + 4)?;
    let abs_bones2 = rel(data, bones_base + 8)?;
    let abs_bones3 = rel(data, bones_base + 12)?;
    let number_layer = at_i32(data, (bones_base + 16 + 24) as usize) as usize;
    let abs_layer = rel(data, bones_base + 20 + 24)?;

    // image metas -> count real images
    let mut number_real = 0usize;
    let mut q = images_meta;
    for _ in 0..number_images {
        let save = q;
        q = rel(data, q)?;
        let w = at_i32(data, q as usize);
        let h = at_i32(data, q as usize + 4);
        if w != 0 && h != 0 {
            number_real += 1;
        }
        q = save + 4;
    }

    // textures: descriptor table at file offset 6
    let mut textures = Vec::with_capacity(number_real);
    let mut t = 6usize;
    for _ in 0..number_real {
        check_range(data, t, 24)?;
        let w = at_i32(data, t) as usize;
        let h = at_i32(data, t + 4) as usize;
        let fmt_raw = at_i32(data, t + 12);
        let size = at_i32(data, t + 20) as usize;
        check_range(data, t + 24, size)?;
        let blk = &data[t + 24..t + 24 + size];
        // blk starts with a 128-byte DDS header; payload follows.
        ensure!(blk.len() >= 128, "texture data too short ({})", blk.len());
        let fourcc = &blk[84..88];
        let fmt = if fourcc == b"DXT1" {
            TextureFmt::Dxt1
        } else if fourcc == b"DXT5" {
            TextureFmt::Dxt5
        } else {
            // fall back to the descriptor format field
            if fmt_raw == 15 {
                TextureFmt::Dxt5
            } else {
                TextureFmt::Dxt1
            }
        };
        let payload = blk[128..].to_vec();
        textures.push(Texture {
            w,
            h,
            fmt,
            payload,
        });
        t += 24 + size;
    }

    // vertex lists
    check_range(data, t, 2)?;
    let num_vl = at_u16(data, t) as usize;
    t += 2;
    let mut vertex_lists = Vec::with_capacity(num_vl);
    for _ in 0..num_vl {
        check_range(data, t, 4)?;
        let size = at_u32(data, t) as usize;
        t += 4;
        check_range(data, t, size)?;
        vertex_lists.push(&data[t..t + size]);
        t += size;
    }

    // index lists
    check_range(data, t, 2)?;
    let num_il = at_u16(data, t) as usize;
    t += 2;
    let mut index_lists = Vec::with_capacity(num_il);
    for _ in 0..num_il {
        check_range(data, t, 4)?;
        let size = at_u32(data, t) as usize;
        t += 4;
        check_range(data, t, size)?;
        index_lists.push(&data[t..t + size]);
        t += size;
    }

    // parts
    let mm = mesh_meta as usize;
    check_range(data, mm + 0x14, 4)?;
    let number_parts = at_i32(data, mm + 0x14) as usize;
    let mut part_pos = mm + 0x14 + 4 + 0x08;
    let mut parts = Vec::with_capacity(number_parts);
    for _ in 0..number_parts {
        check_range(data, part_pos, 4)?;
        let offset_part = at_i32(data, part_pos) as i64;
        let desc = part_pos as i64 + offset_part;
        let desc = usize::try_from(desc).context("negative part descriptor offset")?;
        check_range(data, desc, 0x30)?;
        let num_i = at_i32(data, desc + 4) + 2;
        let stride = at_u16(data, desc + 8) as usize;
        let off_v = at_i32(data, desc + 0x14) as usize;
        let num_v = at_i32(data, desc + 0x18) as usize;
        let off_i = at_i32(data, desc + 0x1c) as usize;
        let il = at_i32(data, desc + 0x20) as usize;
        let vl = at_i32(data, desc + 0x24) as usize;
        // +0x0a: 0xff-terminated list of the part's skin bones. With 8 u8s the
        // list can hold up to 9 bones; the header row is `[stride u16]` so the
        // first list byte sits right after the stride.
        let mut skin_bones = Vec::new();
        for k in 0..9 {
            let b = data[desc + 0x0a + k];
            if b == 0xff {
                break;
            }
            skin_bones.push(b);
        }
        // Shape keys ("dynamic buffers"): after vertexBufferID (+0x24) the
        // descriptor holds `dynamicBufferCount` (+0x28) and a rel pointer
        // (+0x2c) to an array of `dynamicBufferCount` rel pointers. Each
        // nonzero pointer targets `num_v*3` f32 per-vertex offsets; 0 = empty
        // slot. (Matches BactaTank's BactaTankMesh.read.)
        let dynamic_buffers = parse_dynamic_buffers(data, desc, num_v)?;
        ensure!(num_i >= 2, "part has fewer than 2 indices");
        parts.push(Part {
            stride,
            off_v,
            num_v,
            off_i,
            num_i: num_i as usize,
            il,
            vl,
            skin_bones,
            dynamic_buffers,
        });
        part_pos += 4;
    }

    // layers -> render items
    let mut render = Vec::new();
    let mut render_layer = Vec::new();
    let mut layer_pos = abs_layer;
    for li in 0..number_layer as u32 {
        let _text = rel(data, layer_pos)?;
        let mut lp = [0i64; 4];
        let mut q = layer_pos + 4;
        for slot in lp.iter_mut() {
            let tmp = at_i32(data, q as usize);
            if tmp != 0 {
                *slot = rel(data, q)?;
            }
            q += 4;
        }
        layer_pos = q;
        for (slot, per_bone) in [(lp[0], true), (lp[1], false), (lp[2], true), (lp[3], false)] {
            let before = render.len();
            read_bpp_pairs(data, number_bones, slot, per_bone, &mut render)?;
            render_layer.extend(std::iter::repeat(li).take(render.len() - before));
        }
    }
    ensure!(
        render.len() == number_parts,
        "render item count {} != part count {}",
        render.len(),
        number_parts
    );
    ensure!(
        render_layer.len() == render.len(),
        "layer bookkeeping mismatch"
    );

    // bones. Layout (per bone struct, 0x60 bytes):
    //   [identity mat44 @ 0x00][0xC pad][name rel @ 0x4C][parent @ 0x50][3 pad][0xC pad]
    // then after all bone structs a consecutive block of `local` mat44s
    // (the model rest-pose skeleton).
    let mut bones = Vec::with_capacity(number_bones);
    let mut q = abs_bones;
    for _ in 0..number_bones {
        let identity = mat_at(data, q)?;
        let b = q + 0x40 + 0xc;
        let name_rel = at_i32(data, b as usize);
        let name_pos = b + name_rel as i64;
        let np = usize::try_from(name_pos).context("negative bone name offset")?;
        let name = {
            let rest = &data[np.min(data.len())..];
            let end = rest.iter().position(|&c| c == 0).unwrap_or(rest.len());
            String::from_utf8_lossy(&rest[..end]).into_owned()
        };
        let parent = data[(b + 4) as usize] as i32;
        bones.push(Bone {
            name,
            parent: if parent >= 0x40 { -1 } else { parent },
            identity,
            local: Mat4::IDENTITY,
            world: Mat4::IDENTITY,
            bind: Mat4::IDENTITY,
        });
        q += 0x60;
    }

    // local matrices (parent-relative offsets) — consecutive, right after the
    // bone structs, at `abs_bones2` (the addon reads `local_mats` here).
    // `abs_bones3` holds the world-rest (model binding) matrices.
    for i in 0..number_bones {
        let local = mat_at(data, abs_bones2 + i as i64 * 0x40)?;
        let bind = mat_at(data, abs_bones3 + i as i64 * 0x40)?;
        bones[i].local = local;
        bones[i].bind = bind;
        let parent = bones[i].parent;
        // GHSS world = chained local (as the addon builds its armature).
        if parent == -1 {
            bones[i].world = local;
        } else {
            bones[i].world = bones[parent as usize].world * local;
        }
    }

    Ok(Parsed {
        materials,
        textures,
        parts,
        render,
        render_layer,
        bones,
        vertex_lists,
        index_lists,
    })
}

fn mat_at(d: &[u8], o: i64) -> Result<Mat4> {
    let o = usize::try_from(o).context("negative matrix offset")?;
    check_range(d, o, 0x40)?;
    let mut a = [0f32; 16];
    for (i, slot) in a.iter_mut().enumerate() {
        *slot = f32::from_le_bytes(d[o + i * 4..o + i * 4 + 4].try_into().unwrap());
    }
    Ok(Mat4::from_cols_array(&a))
}

fn read_bpp_pairs(
    d: &[u8],
    number_bones: usize,
    pos0: i64,
    per_bone: bool,
    out: &mut Vec<RenderItem>,
) -> Result<()> {
    if pos0 == 0 {
        return Ok(());
    }
    if per_bone {
        let mut p = pos0;
        for bone in 0..number_bones {
            let tmp = at_i32(d, p as usize);
            p += 4;
            if tmp != 0 {
                let mut q = p + tmp as i64 - 4;
                q += 8;
                q = rel(d, q)?;
                q += 0xb0;
                q = rel(d, q)?;
                let pn = at_i32(d, q as usize) as usize;
                q += 4;
                q = rel(d, q)?;
                for j in 0..pn {
                    let m = at_i32(d, (q + (pn - 1 - j) as i64 * 4) as usize);
                    out.push(RenderItem {
                        part: out.len(),
                        mat: m,
                        bone: bone as i32,
                    });
                }
            }
        }
    } else {
        let mut q = pos0 + 8;
        q = rel(d, q)?;
        q += 0xb0;
        q = rel(d, q)?;
        let pn = at_i32(d, q as usize) as usize;
        q += 4;
        q = rel(d, q)?;
        for j in 0..pn {
            let m = at_i32(d, (q + (pn - 1 - j) as i64 * 4) as usize);
            out.push(RenderItem {
                part: out.len(),
                mat: m,
                bone: -1,
            });
        }
    }
    Ok(())
}

/// Shape keys ("dynamic buffers") for a part descriptor at `desc`.
///
/// Layout (after `vertexBufferID` at +0x24):
///   +0x28 u32 dynamicBufferCount
///   +0x2c i32 rel-ptr → [dbc × i32 rel-ptr], each pointing at `num_v*3` f32
///         per-vertex [x,y,z] offsets from the base pose. A pointer of 0 means
///         the slot is empty (`None`).
///
/// Pointers are relative to their own field position (read value then seek
/// `value - 4`), i.e. `data = field_pos + value`, matching BactaTank.
fn parse_dynamic_buffers(data: &[u8], desc: usize, num_v: usize) -> Result<Vec<Option<Vec<[f32; 3]>>>> {
    let count = at_i32(data, desc + 0x28);
    ensure!(count >= 0, "negative dynamic buffer count ({count})");
    ensure!(count as usize <= 4096, "implausible dynamic buffer count ({count})");
    if count == 0 {
        return Ok(Vec::new());
    }
    let array = rel(data, (desc + 0x2c) as i64)?;
    let array = usize::try_from(array).context("negative dynamic buffer array offset")?;
    check_range(data, array, count as usize * 4)?;
    let mut buffers = Vec::with_capacity(count as usize);
    for i in 0..count as usize {
        let field = array + i * 4;
        let value = at_i32(data, field);
        if value == 0 {
            buffers.push(None);
            continue;
        }
        let block = rel(data, field as i64)?;
        let block = usize::try_from(block).context("negative dynamic buffer block offset")?;
        let floats = num_v * 3;
        check_range(data, block, floats * 4)?;
        let mut vec = Vec::with_capacity(num_v);
        for j in 0..num_v {
            vec.push([
                f32::from_le_bytes(data[block + j * 12..block + j * 12 + 4].try_into().unwrap()),
                f32::from_le_bytes(data[block + j * 12 + 4..block + j * 12 + 8].try_into().unwrap()),
                f32::from_le_bytes(data[block + j * 12 + 8..block + j * 12 + 12].try_into().unwrap()),
            ]);
        }
        buffers.push(Some(vec));
    }
    Ok(buffers)
}

/// UV byte offset for a given vertex stride (Star Wars variant).
pub fn uv_offset(stride: usize) -> Option<usize> {
    match stride {
        44 => Some(28),
        40 => Some(24),
        36 => Some(28),
        32 => Some(24),
        _ => None,
    }
}
