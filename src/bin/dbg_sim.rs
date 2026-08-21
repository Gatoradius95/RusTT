// CPU mirror of viewer fs_model (shaders.wgsl) for a chosen part, with the
// actual per-mesh light set and a fixed camera. Prints the resulting colors.
use rustt::map::parse;
use rustt::mapmesh::expand_mesh;
use rustt::rtl::{compute_light_set, LightSet, RtlLight};
use glam::Vec3;

fn main() {
    let path = "backup/LEVELS/MAP/MAP/MAP_PC.GSC";
    let data = std::fs::read(path).unwrap();
    let map = parse(&data).unwrap();
    let rtl_path = "backup/LEVELS/MAP/MAP/MAP.RTL";
    let lights: Vec<RtlLight> = if let Ok(d) = std::fs::read(rtl_path) {
        rustt::rtl::parse(&d)
    } else {
        Vec::new()
    };
    println!("map lights: {} (parse of {rtl_path})", lights.len());

    let part_idx = std::env::args().nth(1).unwrap_or("579".into()).parse::<usize>().unwrap();
    let cam = std::env::args()
        .nth(2)
        .map(|s| {
            let v: Vec<f32> = s.split(',').filter_map(|x| x.parse().ok()).collect();
            Vec3::new(v[0], v[1], v[2])
        })
        .unwrap_or(Vec3::new(-42.0, 5.0, -54.0));

    let part = &map.render_parts[part_idx];
    println!(
        "part lightmap key: {} ; lm state: {}",
        part.lightmap,
        map.lightmaps.get(&part.lightmap).map(|st| {
            format!(
                "ty={} tex=[{} {} {} {}] off={:?}",
                st.ty, st.tex[0], st.tex[1], st.tex[2], st.tex[3], st.off
            )
        }).unwrap_or("none".into())
    );
    let mesh = &map.meshes[part.mesh];
    let mat = &map.materials[part.material];
    let md = expand_mesh(&map, mesh).unwrap();

    let center = {
        let mut c = Vec3::ZERO;
        for p in &md.pos {
            c += Vec3::from(*p);
        }
        c / md.pos.len() as f32
    };
    let set = compute_light_set(&lights, center.to_array());
    println!(
        "part {part_idx} mesh {} mat {} defs=0x{:08x} tf=0x{:08x} lmset={} ls={} sp={:?} center=({:.1},{:.1},{:.1})",
        part.mesh, mat.id, mat.shader_defines, mat.texture_flags, mat.lightmap_set_index,
        mat.lighting_stage, mat.specular_params, center.x, center.y, center.z
    );
    println!("lights: amb={:?} cols={:?} poss={:?}", set.scene_ambient, set.light_color, set.light_pos);

    let prelit = (mat.shader_defines & 0x1000) != 0;
    let has_lm = !prelit;
    let lm_stage = (prelit && mat.lightmap_stage() != 0) as u32;
    let _ = (has_lm, lm_stage);
    let base = [mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]];
    let sp = mat.specular_params;
    let ambient = [0.10f32, 0.11, 0.13];
    // WGSL fs_model clone
    let nverts = md.pos.len();
    let mut avg = Vec3::ZERO;
    let mut alpha_out = 1.0f32;
    let mut hist: std::collections::BTreeMap<[u8; 3], usize> = Default::default();
    for v in 0..nverts {
        let p = Vec3::from(md.pos[v]);
        let n = Vec3::from(md.nrm[v]).normalize();
        let view_dir = (cam - p).normalize();
        let mut color = Vec3::new(base[0], base[1], base[2]);
        let vc = md.color[v];
        let mut alpha = 1.0f32;
        let mut baked = Vec3::ONE;
        if prelit {
            baked = Vec3::new(
                (vc[0] as f32 / 255.0).min(1.0),
                (vc[1] as f32 / 255.0).min(1.0),
                (vc[2] as f32 / 255.0).min(1.0),
            );
            color *= baked;
            alpha = (vc[3] as f32 / 255.0 * 2.0).min(1.0);
            alpha_out = alpha;
        }
        let mut lm_diffuse = Vec3::ONE;
        // no lightmap sample available -> 1.0 (has_lm=0 path)
        let mut diffuse = Vec3::ONE;
        if prelit {
            diffuse = lm_diffuse;
        }
        let mut specular = Vec3::ZERO;
        if mat.lighting_stage != 0 {
            if !prelit {
                let mut d = Vec3::ZERO;
                for i in 0..3 {
                    let l = Vec3::from(set.light_pos[i].to_vec_slice()).normalize();
                    let ndl = n.dot(l).max(0.0);
                    let li = set.light_pos[i][3];
                    let lc = Vec3::from(set.light_color[i].to_vec_slice());
                    d += lc * ndl * li;
                }
                d += Vec3::from(ambient) + Vec3::from(set.scene_ambient.to_vec_slice());
                diffuse = d;
            }
            if mat.lighting_stage == 6 {
                for i in 0..3 {
                    let l = Vec3::from(set.light_pos[i].to_vec_slice()).normalize();
                    let r = (-l).reflect(n);
                    let lc = Vec3::from(set.light_color[i].to_vec_slice());
                    specular += lc * view_dir.dot(r).max(0.0).powf(sp[0]) * set.light_color[i][3];
                }
                specular *= sp[1];
                if prelit {
                    specular *= baked;
                }
            }
        }
        let mut lit = color * diffuse + specular;
        lit = lit.min(Vec3::ONE);
        avg += lit;
        let px = [
            (lit.x * 255.0).round() as u8,
            (lit.y * 255.0).round() as u8,
            (lit.z * 255.0).round() as u8,
        ];
        *hist.entry(px).or_insert(0) += 1;
    }
    avg /= nverts as f32;
    println!("avg out rgb = ({:.3},{:.3},{:.3}) -> {:?} alpha={:.3}", avg.x, avg.y, avg.z, [(avg.x*255.0) as u8, (avg.y*255.0) as u8, (avg.z*255.0) as u8], alpha_out);
    println!("vertex color byte: {:?}", md.color[0]);
    let top: Vec<_> = hist.into_iter().rev().take(5).collect();
    for (k, n) in top {
        let pct = n as f32 * 100.0 / nverts as f32;
        if pct > 0.5 {
            println!("  {:?} x{n} ({pct:.1}%)", k);
        }
    }
}

trait ToSlice {
    fn to_vec_slice(&self) -> [f32; 3];
}
impl ToSlice for [f32; 4] {
    fn to_vec_slice(&self) -> [f32; 3] {
        [self[0], self[1], self[2]]
    }
}
impl ToSlice for [f32; 3] {
    fn to_vec_slice(&self) -> [f32; 3] {
        *self
    }
}
