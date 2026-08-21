use rustt::map::parse;
use rustt::mapmesh::expand_mesh;

fn main() {
    let path = "backup/LEVELS/MAP/MAP/MAP_PC.GSC";
    let data = std::fs::read(path).unwrap();
    let map = parse(&data).unwrap();
    let mut total_prelit_lm0 = 0usize;
    let mut mats: std::collections::BTreeMap<String, usize> = Default::default();
    for (pi, part) in map.render_parts.iter().enumerate() {
        let Some(m) = map.meshes.get(part.mesh) else { continue };
        let Some(mat) = map.materials.get(part.material) else { continue };
        let Some(md) = expand_mesh(&map, m) else { continue };
        if md.pos.len() < 4 {
            continue;
        }
        let prelit = mat.shader_defines & 0x1000 != 0;
        let live31 = mat.shader_defines & 0x8000_0000 != 0;
        let lm = mat.lightmap_stage();
        let key = (mat.shader_defines, lm, -1, false);
        let _ = key;
        if !(prelit && lm == 0) {
            continue;
        }
        total_prelit_lm0 += 1;
        let mut cs = std::collections::BTreeMap::<[u8; 4], usize>::new();
        for c in &md.color {
            *cs.entry(*c).or_insert(0) += 1;
        }
        let top: Vec<String> = cs
            .iter()
            .take(3)
            .map(|(c, n)| format!("({:3},{:3},{:3},{:3})x{}", c[0], c[1], c[2], c[3], n))
            .collect();
        let e = mats.entry(format!("defs=0x{:08x} lm={} tex={}", mat.shader_defines, lm, mat.tex_id)).or_insert(0);
        *e += 1;
        let x0 = md.pos.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
        let x1 = md.pos.iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max);
        let z0 = md.pos.iter().map(|p| p[2]).fold(f32::INFINITY, f32::min);
        let z1 = md.pos.iter().map(|p| p[2]).fold(f32::NEG_INFINITY, f32::max);
        println!(
            "part {pi:3} mat={:4} defs=0x{:08x} ls={} tex={:4} vcol[{}] x={:.1}..{:.1} z={:.1}..{:.1}",
            mat.id,
            mat.shader_defines,
            mat.lighting_stage,
            mat.tex_id,
            top.join(","),
            x0,
            x1,
            z0,
            z1,
        );
    }
    println!("\n== summary per material ==");
    for (k, n) in &mats {
        println!("{k}: parts={n}");
    }
    println!("\ntotal prelit+lm0 parts: {total_prelit_lm0}");
}