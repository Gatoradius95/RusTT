use rustt::map::parse;
use rustt::mapmesh::expand_mesh;

fn main() {
    let path = "backup/LEVELS/MAP/MAP/MAP_PC.GSC";
    let data = std::fs::read(path).unwrap();
    let map = parse(&data).unwrap();
    let lt = map.scene.material_list_ptr as i64;
    let rel = |q: i64| -> i64 {
        let at = i32::from_le_bytes(data[q as usize..q as usize + 4].try_into().unwrap()) as i64;
        q + at
    };
    let tsp = |mi: usize| -> i16 {
        let r = rel(lt + mi as i64 * 4);
        i16::from_le_bytes(
            data[r as usize + 0xfc..r as usize + 0xfe]
                .try_into()
                .unwrap(),
        )
    };
    for (pi, part) in map.render_parts.iter().enumerate() {
        let Some(m) = map.meshes.get(part.mesh) else { continue };
        let Some(mat) = map.materials.get(part.material) else { continue };
        let Some(md) = expand_mesh(&map, m) else { continue };
        if md.pos.len() < 4 {
            continue;
        }
        let mut cs = std::collections::BTreeMap::<[u8; 4], usize>::new();
        for c in &md.color {
            *cs.entry(*c).or_insert(0) += 1;
        }
        let top: Vec<([u8; 4], usize)> = cs.into_iter().collect();
        let top: Vec<String> = top
            .iter()
            .take(3)
            .map(|(c, n)| format!("({:3},{:3},{:3},{:3})x{}", c[0], c[1], c[2], c[3], n))
            .collect();
        let x0 = md.pos.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
        let x1 = md.pos.iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max);
        let z0 = md.pos.iter().map(|p| p[2]).fold(f32::INFINITY, f32::min);
        let z1 = md.pos.iter().map(|p| p[2]).fold(f32::NEG_INFINITY, f32::max);
        println!(
            "part {pi:3} mesh={:3} matid={:4} defs=0x{:08x} lmset={} tf=0x{:08x} tex={} tsp={} vcol[{}] x={:.1}..{:.1} z={:.1}..{:.1}",
            part.mesh,
            mat.id,
            mat.shader_defines,
            mat.lightmap_set_index,
            mat.texture_flags,
            mat.tex_id,
            tsp(part.material),
            top.join(","),
            x0,
            x1,
            z0,
            z1,
        );
    }
}