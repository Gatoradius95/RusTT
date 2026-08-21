use rustt::map::parse;

fn main() {
    let path = "backup/LEVELS/MAP/MAP/MAP_PC.GSC";
    let data = std::fs::read(path).unwrap();
    let map = parse(&data).unwrap();
    println!("{} materials", map.materials.len());
    let mut tex_hit = 0usize;
    let mut tex_miss = 0usize;
    for m in map.materials.iter() {
        if m.tex_id < 0 {
            continue;
        }
        match map.tex_slot(m.tex_id) {
            Some(_) => tex_hit += 1,
            None => tex_miss += 1,
        }
    }
    println!(
        "tex_slot: {} hit, {} miss (of {} materials with tex_id >= 0); real_index list len = {}",
        tex_hit,
        tex_miss,
        tex_hit + tex_miss,
        map.texture_real_index.len()
    );
    for (i, m) in map.materials.iter().enumerate() {
        // Optional mat-id filter: `dbg_mats 12` prints only material id 12.
        if let Some(want) = std::env::args().nth(1).and_then(|s| s.parse::<i32>().ok()) {
            if m.id != want {
                continue;
            }
        } else if m.id != 310 && m.id != 309 && m.id != 311 && m.id != 221 && m.id != 232
            && m.id != 136
            && m.id != 131 && m.id != 135 && m.id != 132 && m.id != 303 && m.id != 58 && m.id != 151
            && m.id != 102 && m.id != 24 && m.id != 38
        {
            continue;
        }
        println!(
            "idx={:3} mat {:4} dif=[{:6.3},{:6.3},{:6.3}] a={:.3} tex={:4} tf=0x{:08x} lmset={} uvsc=0x{:x} defs=0x{:08x} ls={} sp=[{:7.3},{:7.3},{:7.3},{:7.3}] lmstage={}",
            i,
            m.id,
            m.diffuse[0],
            m.diffuse[1],
            m.diffuse[2],
            m.diffuse[3],
            m.tex_id,
            m.texture_flags,
            m.lightmap_set_index,
            m.uv_set_coords,
            m.shader_defines,
            m.lighting_stage,
            m.specular_params[0],
            m.specular_params[1],
            m.specular_params[2],
            m.specular_params[3],
            m.lightmap_stage(),
        );
    }
}