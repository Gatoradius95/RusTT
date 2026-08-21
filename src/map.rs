use std::collections::HashMap;

use anyhow::{ensure, Result};

use crate::ghg::TextureFmt;

/// A parsed LEGO TCS level `.GSC` bundle (the big scene file that holds the
/// whole map: textures, shared vertex/index buffers, materials and the
/// renderable mesh list).
///
/// Layout (verified against BrickBench's `SceneFileLoader` /
/// `GameSceneHeaderBlock`):
///
/// ```text
/// [0x000] u32 -> nu20 block start minus 4 (i.e. `nu20 = getInt(0) + 4`)
/// [0x006] texture blobs, one per real texture:
///         24-byte meta [w][h][?][?][?][size] + `size` bytes of DDS data
/// [..]    u16 vertexBufferCount, then count×(u32 size + bytes)
/// [..]    u16 indexBufferCount,  then count×(u32 size + bytes)
/// [..]    'NU20' block, then a block stream:
///         NTBL, MS00 (materials), TSS0, DINI, …, PSID (mesh descriptors +
///         vertex/index data), …, GSNH, PNTR.
///         Every block is [u32 fourcc][u32 size] where size INCLUDES the
///         8-byte header, so the stream walks with `pos += size`.
/// [nu20+0x18] self-relative pointer -> PNTR relocation table
/// [nu20+0x1C] self-relative pointer -> GSNH block content
/// GSNH   scene header (content at `gsnh`):
///         +0x00 texIndexList, +0x04 texCount, +0x08 texMetaPtr,
///         +0x0C materialPtrList, +0x1D0 gp (renderable list).
///         gp+0x10 self-relative mesh list start, gp+0x14 mesh count.
/// PNTR   relocation table: `numPntr` self-relative entries; each fixes the
///         pointer value at `entry + rel` to `ptr + i32(ptr)` (absolute).
/// ```
pub struct Map<'a> {
    /// Textures from the leading region (DDS payloads).
    pub textures: Vec<Texture>,
    /// The descriptor (real) index of each entry in `textures` (parallel).
    /// Materials store their texture as this real index, not a `textures`
    /// position, so lookups go through `Map::tex_slot`.
    pub texture_real_index: Vec<u32>,
    /// Shared vertex buffers; meshes index them via `Mesh::vertex_list_id`.
    pub vertex_buffers: Vec<&'a [u8]>,
    /// Shared index buffers; meshes index them via `Mesh::index_list_id`.
    pub index_buffers: Vec<&'a [u8]>,
    /// Materials parsed from the MS00 block (0x2C4 bytes each).
    pub materials: Vec<Material>,
    /// Renderable meshes, in the order the game draws them.
    pub meshes: Vec<Mesh>,
    /// The material→mesh pairing from the DISP block's game models, flattened
    /// in game-model order. The mesh list itself carries no material, so this
    /// is the mapping a renderer needs. Empty when the file has no DISP block.
    pub render_parts: Vec<RenderPart>,
    /// Raw scene-header bookkeeping.
    pub scene: SceneInfo,
}

/// One drawable (mesh + material) resolved from the DISP game-model records.
pub struct RenderPart {
    /// Index into `Map::meshes`.
    pub mesh: usize,
    /// Index into `Map::materials`.
    pub material: usize,
}

impl Map<'_> {
    /// Map a material's stored texture real index to a position in `textures`.
    pub fn tex_slot(&self, real_index: i16) -> Option<usize> {
        if real_index < 0 {
            return None;
        }
        self.texture_real_index
            .iter()
            .position(|&r| r as i16 == real_index)
    }
}

pub struct SceneInfo {
    pub gsnh_data: usize,
    pub tex_count: u32,
    pub tex_meta_ptr: usize,
    pub material_list_ptr: usize,
    /// Renderable list header location (the `gp` in BrickBench's reader).
    pub gsc_renderable_list: usize,
    pub mesh_list_start: usize,
    pub mesh_count: u32,
}

pub struct Texture {
    pub w: usize,
    pub h: usize,
    pub fmt: TextureFmt,
    /// Raw DDS payload (the 128-byte DDS header is stripped).
    pub payload: Vec<u8>,
}

pub struct Material {
    pub id: i32,
    pub diffuse: [f32; 4],
    /// Real index into the texture list (`diffuseFileTexture` in BrickBench).
    pub tex_id: i16,
    /// 0xB4 in the material record (texture flags).
    pub texture_flags: u32,
    /// 0x1F0 (vertex format bits).
    pub vertex_format_bits: u32,
}

/// A renderable mesh (triangle strip, `mesh_type == 6`).
///
/// Vertices live in `vertex_buffers[vertex_list_id]`; `vertex_offset` is a
/// vertex index into that buffer (relative base vertex) and `vertex_size` is
/// the per-vertex stride. Triangles are u16 strips in
/// `index_buffers[index_list_id]`; `index_offset` is an element (u16) offset,
/// so the byte offset is `index_offset * 2`.
pub struct Mesh {
    pub address: usize,
    pub mesh_type: u32,
    pub triangle_count: u32,
    pub vertex_size: u16,
    pub vertex_offset: u32,
    pub vertex_count: u32,
    pub index_offset: u32,
    pub index_list_id: u32,
    pub vertex_list_id: u32,
    pub use_dynamic_buffer: u32,
    pub dynamic_buffer: u32,
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

#[inline]
fn check_range(d: &[u8], o: usize, n: usize) -> Result<()> {
    ensure!(o + n <= d.len(), "read past end (offset {o:#x}, len {n})");
    Ok(())
}

const BLOCK_MS00: u32 = 0x3030534d; // "MS00"
const BLOCK_DISP: u32 = 0x50534944; // "DISP"

/// Display-command types from BrickBench's `DisplayCommand.CommandType`.
const CMD_GEOMCALL: u8 = 0x82;
const CMD_END: u8 = 0x8e;

/// BrickBench `readPointer`: a self-relative signed offset from the field's
/// own address. A zero value is a null pointer.
#[inline]
fn rel(d: &[u8], off: usize) -> usize {
    let v = at_i32(d, off);
    if v == 0 {
        0
    } else {
        (off as i64 + v as i64) as usize
    }
}

pub fn parse(data: &[u8]) -> Result<Map<'_>> {
    // --- nu20 header: relative pointers to PNTR count and GSNH data --------
    let nu20 = at_u32(data, 0) as usize + 4;
    check_range(data, nu20, 0x20)?;
    ensure!(&data[nu20..nu20 + 4] == b"NU20", "not a Nu2 map (bad 'NU20' magic)");
    let pntr_loc = rel(data, nu20 + 0x18);
    let gsnh_data = rel(data, nu20 + 0x1c);
    check_range(data, pntr_loc, 4)?;
    check_range(data, gsnh_data, 0x20)?;

    // --- PNTR relocation table ---------------------------------------------
    // Entries are self-relative signed offsets from their own address. Each
    // non-zero target value `off` at `ptr` is replaced by the absolute
    // `ptr + off`.
    let num_pntr = at_u32(data, pntr_loc);
    let mut reloc = HashMap::with_capacity(num_pntr as usize);
    for i in 0..num_pntr {
        let x = pntr_loc + 4 + i as usize * 4;
        if x + 4 > data.len() {
            break;
        }
        let rel = at_i32(data, x);
        let ptr = x as i64 + rel as i64;
        if ptr < 0 || (ptr as usize) + 4 > data.len() {
            continue;
        }
        let ptr = ptr as usize;
        let off = at_i32(data, ptr);
        if off == 0 {
            continue;
        }
        let tgt = ptr as i64 + off as i64;
        if tgt < 0 || tgt > data.len() as i64 {
            continue;
        }
        reloc.insert(ptr, tgt as u32);
    }

    // --- GSNH scene header --------------------------------------------------
    let _tex_index_list = rel(data, gsnh_data);
    let tex_count = at_u32(data, gsnh_data + 4);
    let tex_meta_ptr = rel(data, gsnh_data + 8);
    let material_list_ptr = rel(data, gsnh_data + 0xc);

    let gp = rel(data, gsnh_data + 0x1d0);
    check_range(data, gp, 0x1c)?;
    let mesh_list_start = rel(data, gp + 0x10);
    let mesh_count = at_u32(data, gp + 0x14);

    // --- textures -----------------------------------------------------------
    // Real textures are the descriptors with non-zero width/height; their
    // blobs are laid out sequentially in the leading region starting at 0x6.
    let mut real_indices: Vec<u32> = Vec::new();
    for i in 0..tex_count as usize {
        let tgt = rel(data, tex_meta_ptr + i * 4);
        if tgt + 8 <= data.len() {
            let w = at_i32(data, tgt);
            let h = at_i32(data, tgt + 4);
            if w > 0 && h > 0 {
                real_indices.push(i as u32);
            }
        }
    }
    let texture_real_index = real_indices;
    let mut textures = Vec::with_capacity(texture_real_index.len());
    let mut t = 6usize;
    for _ in 0..texture_real_index.len() {
        check_range(data, t, 24)?;
        let mut w = at_i32(data, t) as usize;
        let mut h = at_i32(data, t + 4) as usize;
        let fmt_raw = at_i32(data, t + 12);
        let size = at_i32(data, t + 20) as usize;
        check_range(data, t + 24, size)?;
        let blk = &data[t + 24..t + 24 + size];
        ensure!(blk.len() >= 128, "texture data too short ({})", blk.len());
        let fmt = if &blk[84..88] == b"DXT5" || fmt_raw == 15 {
            TextureFmt::Dxt5
        } else {
            TextureFmt::Dxt1
        };
        // Some blobs carry garbage in their own meta (e.g. w = -128). The
        // descriptor list (texIndexList) is authoritative for what counts as a
        // real texture, so fall back to the DDS header dimensions in that case.
        if w == 0 || h == 0 || w > 8192 || h > 8192 {
            w = at_u32(blk, 16) as usize;
            h = at_u32(blk, 12) as usize;
        }
        textures.push(Texture {
            w,
            h,
            fmt,
            payload: blk[128..].to_vec(),
        });
        t += 24 + size;
    }

    // --- vertex buffers (right after the texture blobs) ---------------------
    check_range(data, t, 2)?;
    let num_vl = at_u16(data, t) as usize;
    t += 2;
    let mut vertex_buffers = Vec::with_capacity(num_vl);
    for _ in 0..num_vl {
        check_range(data, t, 4)?;
        let size = at_u32(data, t) as usize;
        t += 4;
        check_range(data, t, size)?;
        vertex_buffers.push(&data[t..t + size]);
        t += size;
    }

    // --- index buffers -------------------------------------------------------
    check_range(data, t, 2)?;
    let num_il = at_u16(data, t) as usize;
    t += 2;
    let mut index_buffers = Vec::with_capacity(num_il);
    for _ in 0..num_il {
        check_range(data, t, 4)?;
        let size = at_u32(data, t) as usize;
        t += 4;
        check_range(data, t, size)?;
        index_buffers.push(&data[t..t + size]);
        t += size;
    }

    // --- materials (MS00 block, 0x2C4 bytes each) ----------------------------
    let mut materials = Vec::new();
    walk_blocks(data, nu20 + 0x20, &mut |pos, id| {
        if id == BLOCK_MS00 && pos + 0x10 <= data.len() {
            // MS00: [u32 count][u32 skip][materials]
            let count = at_u32(data, pos + 8) as usize;
            ensure!(pos + 16 + count * 0x2c4 <= data.len(), "MS00 block too small");
            let mut q = pos + 16;
            for _ in 0..count {
                let id = at_i32(data, q + 0x38);
                let diffuse = [
                    f32::from_le_bytes(data[q + 0x54..q + 0x58].try_into().unwrap()),
                    f32::from_le_bytes(data[q + 0x58..q + 0x5c].try_into().unwrap()),
                    f32::from_le_bytes(data[q + 0x5c..q + 0x60].try_into().unwrap()),
                    f32::from_le_bytes(data[q + 0x60..q + 0x64].try_into().unwrap()),
                ];
                let tex_id = i16::from_le_bytes(data[q + 0x74..q + 0x76].try_into().unwrap());
                let texture_flags = at_u32(data, q + 0xb4);
                let vertex_format_bits = at_u32(data, q + 0x1f0);
                materials.push(Material {
                    id,
                    diffuse,
                    tex_id,
                    texture_flags,
                    vertex_format_bits,
                });
                q += 0x2c4;
            }
            return Ok(true);
        }
        Ok(false)
    })?;

    // --- meshes --------------------------------------------------------------
    // Mesh addresses in the list are self-relative. Descriptor fields are
    // plain values (BrickBench reads them with raw getInt/getShort).
    let mut meshes = Vec::with_capacity(mesh_count as usize);
    for i in 0..mesh_count as usize {
        let m = rel(data, mesh_list_start + i * 4);
        check_range(data, m, 0x38)?;
        meshes.push(Mesh {
            address: m,
            mesh_type: at_u32(data, m),
            triangle_count: at_u32(data, m + 4),
            vertex_size: at_u16(data, m + 8),
            vertex_offset: at_u32(data, m + 0x14),
            vertex_count: at_u32(data, m + 0x18),
            index_offset: at_u32(data, m + 0x1c),
            index_list_id: at_u32(data, m + 0x20),
            vertex_list_id: at_u32(data, m + 0x24),
            use_dynamic_buffer: at_u32(data, m + 0x28),
            dynamic_buffer: at_u32(data, m + 0x34),
        });
    }

    // --- render parts (DISP game-model records) ------------------------------
    // The GSNH mesh list carries no material; the DISP block's game models
    // pair each drawable with a material index (MS00 order) and a display
    // command whose GEOMCALL points back at a mesh address.
    let render_parts = if let Some(disp) = find_block_content(data, nu20 + 0x20, BLOCK_DISP)? {
        parse_render_parts(data, disp, &meshes)?
    } else {
        Vec::new()
    };

    Ok(Map {
        textures,
        texture_real_index,
        vertex_buffers,
        index_buffers,
        materials,
        meshes,
        render_parts,
        scene: SceneInfo {
            gsnh_data,
            tex_count,
            tex_meta_ptr,
            material_list_ptr,
            gsc_renderable_list: gp,
            mesh_list_start,
            mesh_count,
        },
    })
}

/// Return the content offset of the first block with the given id, or `None`.
fn find_block_content(data: &[u8], start: usize, id: u32) -> Result<Option<usize>> {
    let mut found: Option<usize> = None;
    walk_blocks(data, start, &mut |pos, blk_id| {
        if blk_id == id {
            found = Some(pos + 8);
            return Ok(true);
        }
        Ok(false)
    })?;
    Ok(found)
}

/// Resolve the DISP game-model records into flat (mesh, material) parts.
fn parse_render_parts(data: &[u8], disp: usize, meshes: &[Mesh]) -> Result<Vec<RenderPart>> {
    // Display command list: 16-byte commands [u8 type][u8 flags][2 pad][rel ptr],
    // terminated by an END command.
    let cmd_start = rel(data, disp + 8);
    let mut commands: Vec<(u8, u32)> = Vec::new();
    let mut pos = cmd_start;
    loop {
        check_range(data, pos, 16)?;
        let ty = data[pos];
        let resource = rel(data, pos + 4);
        commands.push((ty, resource as u32));
        if ty == CMD_END {
            break;
        }
        pos += 16;
    }

    // Mesh address -> mesh index (the mesh list is authoritative).
    let mut mesh_by_addr = HashMap::with_capacity(meshes.len());
    for (i, m) in meshes.iter().enumerate() {
        mesh_by_addr.insert(m.address as u32, i);
    }

    // Game models: each is a 0x0C record pairing a material index list with a
    // display-command index list.
    let model_count = at_u32(data, disp + 0x10) as usize;
    let models = rel(data, disp + 0x14);
    let mut parts = Vec::new();
    for i in 0..model_count {
        let a = models + i * 0xc;
        check_range(data, a, 0xc)?;
        let cmd_count = at_u32(data, a) as usize;
        if cmd_count == 0 {
            continue;
        }
        let mat_off = rel(data, a + 4);
        let mesh_off = rel(data, a + 8);
        if mat_off == 0 || mesh_off == 0 {
            continue;
        }
        check_range(data, mat_off, cmd_count * 4)?;
        check_range(data, mesh_off, cmd_count * 4)?;
        for k in 0..cmd_count {
            let material = at_u32(data, mat_off + k * 4) as usize;
            let cmd_idx = at_u32(data, mesh_off + k * 4) as usize;
            let Some(&(ty, addr)) = commands.get(cmd_idx) else {
                continue;
            };
            if ty != CMD_GEOMCALL {
                continue;
            }
            let Some(&mesh) = mesh_by_addr.get(&addr) else {
                continue;
            };
            parts.push(RenderPart { mesh, material });
        }
    }
    Ok(parts)
}

/// Walk the block stream. `pos` starts at the first block (right after the
/// nu20 header). Every block is `[u32 id][u32 size]` with `size` including the
/// 8-byte header; `f` is called with each block id; return true to stop.
fn walk_blocks(
    data: &[u8],
    start: usize,
    f: &mut dyn FnMut(usize, u32) -> Result<bool>,
) -> Result<()> {
    let mut pos = start;
    while pos + 8 <= data.len() {
        let id = at_u32(data, pos);
        let size = at_u32(data, pos + 4) as usize;
        if size < 8 || pos + size > data.len() {
            break;
        }
        if f(pos, id)? {
            break;
        }
        pos += size;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal synthetic map: no textures, no vertex/index buffers,
    /// one empty MS00 material block, a GSNH scene header with a single mesh,
    /// and an empty PNTR table. Mirrors BrickBench's layout rules exactly.
    fn synthetic_map() -> Vec<u8> {
        let mut b = Vec::new();
        macro_rules! u16p { ($v:expr) => { b.extend_from_slice(&($v as u16).to_le_bytes()) } }
        macro_rules! u32p { ($v:expr) => { b.extend_from_slice(&($v as u32).to_le_bytes()) } }
        macro_rules! pad { ($n:expr) => { b.resize(b.len() + $n, 0) } }

        // Leading region: nu20 = getInt(0) + 4 -> 0x0C.
        u32p!(8);
        pad!(2); // 0x04..0x06 (texture blob region is empty)
        u16p!(0); // 0x06 vertexBufferCount
        u16p!(0); // 0x08 indexBufferCount
        pad!(2); // 0x0A..0x0C

        // NU20 header at 0x0C.
        b.extend_from_slice(b"NU20");
        u32p!(0);
        u32p!(0);
        u32p!(0);
        b.extend_from_slice(b"HEAD");
        u32p!(16);
        u32p!(0x1fc); // 0x24 rel -> pntr_loc (0x220)
        u32p!(0x1c); // 0x28 rel -> gsnh content (0x44)

        // Block stream from 0x2C.
        b.extend_from_slice(b"MS00"); // 0x2C
        u32p!(16);
        u32p!(0); // material count
        u32p!(0);

        b.extend_from_slice(b"GSNH"); // 0x3C
        u32p!(0x1dc); // block size: content 0x44..0x218
        // GSNH content at 0x44.
        u32p!(0); // +0x00 texIndexList (null)
        u32p!(0); // +0x04 texCount
        u32p!(0); // +0x08 texMetaPtr (null)
        u32p!(0); // +0x0C materialPtrList (null)
        pad!(0x1d0 - 0x10); // 0x54..0x214
        u32p!(0x14); // +0x1D0 (0x214) rel -> gp (0x228)

        b.extend_from_slice(b"PNTR"); // 0x218
        u32p!(12);
        u32p!(0); // num_pntr (pntr_loc = 0x220)

        // gp content at 0x228.
        pad!(4); // 0x224..0x228 (PNTR block ends at 0x224)
        pad!(0x10); // gp+0x00..0x10 (gsc_renderable_list area)
        u32p!(0xc); // gp+0x10 rel -> mesh list (0x244)
        u32p!(1); // gp+0x14 mesh count
        u32p!(0); // gp+0x18

        // Mesh list at 0x244.
        u32p!(4); // rel -> mesh descriptor (0x248)
        // Mesh descriptor at 0x248.
        u32p!(6); // type (triangle strip)
        u32p!(3); // triangle count
        u16p!(32); // vertex size
        pad!(0x14 - 0x0a); // 0x252..0x25C
        u32p!(0); // +0x14 vertex offset
        u32p!(4); // +0x18 vertex count
        u32p!(0); // +0x1C index offset
        u32p!(0); // +0x20 index list id
        u32p!(0); // +0x24 vertex list id
        u32p!(0); // +0x28 use dynamic buffer
        pad!(0x34 - 0x2c); // 0x270..0x27C
        u32p!(0); // +0x34 dynamic buffer

        debug_assert_eq!(b.len(), 0x280, "layout drift in synthetic map");
        b
    }

    #[test]
    fn parses_minimal_map() {
        let data = synthetic_map();
        let map = parse(&data).expect("synthetic map should parse");

        assert_eq!(map.textures.len(), 0);
        assert_eq!(map.vertex_buffers.len(), 0);
        assert_eq!(map.index_buffers.len(), 0);
        assert_eq!(map.materials.len(), 0);

        assert_eq!(map.scene.tex_count, 0);
        assert_eq!(map.scene.mesh_count, 1);
        assert_eq!(map.scene.gsnh_data, 0x44);
        assert_eq!(map.scene.mesh_list_start, 0x244);

        assert_eq!(map.meshes.len(), 1);
        let m = &map.meshes[0];
        assert_eq!(m.mesh_type, 6);
        assert_eq!(m.triangle_count, 3);
        assert_eq!(m.vertex_size, 32);
        assert_eq!(m.vertex_offset, 0);
        assert_eq!(m.vertex_count, 4);
        assert_eq!(m.address, 0x248);
    }

    #[test]
    fn rejects_non_map_file() {
        let data = [0u8; 64];
        match parse(&data) {
            Ok(_) => panic!("garbage should be rejected"),
            Err(e) => assert!(e.to_string().contains("NU20"), "unexpected error: {e}"),
        }
    }
}
