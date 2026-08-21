// seamcheck: does the map's source geometry have watertight seams between
// parts? Loads MAP_PC.GSC, expands every part mesh, then finds pairs of
// parts that share vertex positions EXACTLY (quantized to 1/4096) or nearly
// (min distance < 0.002).
// Usage: cargo run --release --bin dbg_seams
use rustt::map::parse;
use rustt::mapmesh::expand_mesh;
use std::collections::HashMap;

fn main() {
    let path = "backup/LEVELS/MAP/MAP/MAP_PC.GSC";
    let data = std::fs::read(path).unwrap();
    let map = parse(&data).unwrap();

    const SCALE: i64 = 4096;
    let np = map.render_parts.len();
    println!("parts: {np}");
    let mut cell: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    let mut quant: Vec<Vec<(i64, i64, i64)>> = Vec::with_capacity(np);
    let mut any_too_big = 0usize;
    for (i, part) in map.render_parts.iter().enumerate() {
        let mesh = &map.meshes[part.mesh];
        let md = expand_mesh(&map, mesh).unwrap();
        let qs: Vec<(i64, i64, i64)> = md
            .pos
            .iter()
            .map(|p| {
                (
                    (p[0] as f64 * SCALE as f64).round() as i64,
                    (p[1] as f64 * SCALE as f64).round() as i64,
                    (p[2] as f64 * SCALE as f64).round() as i64,
                )
            })
            .collect();
        for &q in &qs {
            cell.entry(q).or_default().push(i);
        }
        if md.pos.len() > 1000 {
            any_too_big += 1;
        }
        quant.push(qs);
    }
    println!("parts with >1000 verts: {any_too_big}");

    // Floor/wall profile dump for the cantina region: flat parts (tiny Y
    // extent, larger XZ) with their vcol alpha min/max and material blend.
    println!("--- planar parts in z<=-18 (floor/wall seam candidates) ---");
    for (i, part) in map.render_parts.iter().enumerate() {
        let mesh = &map.meshes[part.mesh];
        let md = expand_mesh(&map, mesh).unwrap();
        let mut lo = [f32::MAX; 3];
        let mut hi = [f32::MIN; 3];
        let mut amin = 255u8;
        let mut amax = 0u8;
        let mut rmin = [255u8; 3];
        let mut rmax = [0u8; 3];
        for (p, c) in md.pos.iter().zip(md.color.iter()) {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
                rmin[k] = rmin[k].min(c[k]);
                rmax[k] = rmax[k].max(c[k]);
            }
            amin = amin.min(c[3]);
            amax = amax.max(c[3]);
        }
        let ext = [
            hi[0] - lo[0],
            hi[1] - lo[1],
            hi[2] - lo[2],
        ];
        let flat = ext[1] < 0.6 && (ext[0] * ext[2]) > 4.0;
        if flat && lo[2] <= -18.0 && hi[0] <= 5.0 && hi[0] >= -40.0 {
            let mat = &map.materials[part.material];
            println!(
                "part {i}: mesh {} mat {} blend_mode={} alpha_type=0x{:x} y=({:.2},{:.2}) xz=({:.1}x{:.1}) center=({:.1},{:.1}) vcol_a={}..{} vcol_rgb=[{}..{}, {}..{}, {}..{}]",
                part.mesh,
                mat.id,
                mat.blend_mode(),
                mat.alpha_type,
                lo[1],
                hi[1],
                ext[0],
                ext[2],
                (lo[0] + hi[0]) * 0.5,
                (lo[2] + hi[2]) * 0.5,
                amin,
                amax,
                rmin[0],
                rmax[0],
                rmin[1],
                rmax[1],
                rmin[2],
                rmax[2],
            );
        }
    }
    println!("--- end planar dump ---");
    println!("--- wall-like parts (Y extent > 0.4) in xz=[-40..5] (vertical faces) ---");
    let mut walls: Vec<(usize, f32, f32, f32, f32)> = Vec::new();
    for (i, part) in map.render_parts.iter().enumerate() {
        let mesh = &map.meshes[part.mesh];
        let md = expand_mesh(&map, mesh).unwrap();
        let mut lo = [f32::MAX; 3];
        let mut hi = [f32::MIN; 3];
        for p in &md.pos {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let ext = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
        if ext[1] > 0.4 && lo[0] <= 5.0 && lo[0] >= -40.0 && lo[2] >= -90.0 && lo[2] <= 5.0 {
            walls.push((i, (lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, (lo[2] + hi[2]) * 0.5, ext[1]));
        }
    }
    println!("wall-like parts: {}", walls.len());
    for (i, cx, cy, cz, ey) in walls.iter().take(40) {
        println!("part {i}: cy={cy:.2} y-ext={ey:.2} at ({cx:.2},*,{cz:.2})");
    }
    println!("--- end wall dump ---");

    // Region query: `dbg_seams box <x> <z>` — parts whose AABB covers the
    // point (x,z) with y in [-0.5, 1.5], plus near-miss neighbors.
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 4 && args[1] == "box" {
        let qx: f32 = args[2].parse().unwrap();
        let qz: f32 = args[3].parse().unwrap();
        println!("--- parts around ({qx},{qz}) ---");
        let mut hits: Vec<(usize, [f32; 3], [f32; 3], [f32; 3])> = Vec::new();
        for (i, part) in map.render_parts.iter().enumerate() {
            let mesh = &map.meshes[part.mesh];
            let md = expand_mesh(&map, mesh).unwrap();
            let mut lo = [f32::MAX; 3];
            let mut hi = [f32::MIN; 3];
            for p in &md.pos {
                for k in 0..3 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
            }
            let ext = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
            let near = (qx >= lo[0] - 0.1 && qx <= hi[0] + 0.1 && qz >= lo[2] - 0.1 && qz <= hi[2] + 0.1)
                && hi[1] > -0.5 && lo[1] < 1.5;
            let covers = qx >= lo[0] && qx <= hi[0] && qz >= lo[2] && qz <= hi[2];
            if covers || near {
                hits.push((
                    i,
                    lo,
                    hi,
                    ext,
                ));
            }
        }
        for (i, lo, hi, ext) in hits {
            let mat = &map.materials[map.render_parts[i].material];
            println!(
                "part {i}: mesh {} mat {} blend_mode={} x={:.3}..{:.3} y={:.3}..{:.3} z={:.3}..{:.3} (ext {:.3}x{:.3}x{:.3})",
                map.render_parts[i].mesh,
                mat.id,
                mat.blend_mode(),
                lo[0],
                hi[0],
                lo[1],
                hi[1],
                lo[2],
                hi[2],
                ext[0],
                ext[1],
                ext[2],
            );
        }
        println!("--- end box ---");
    }
    // Tile scan: `dbg_seams tiles` — all parts whose y range spans the floor
    // band (lo[1] < 0.05 && hi[1] > 0.05) inside x[-36..-18], z[-62..-44],
    // sorted by x then z, so the floor tiling around a corner is visible.
    if args.len() >= 2 && args[1] == "tiles" {
        println!("--- floor-band parts x[-36,-18] z[-62,-44] ---");
        let mut list: Vec<(usize, f32, f32, f32, f32, f32, f32)> = Vec::new();
        for (i, part) in map.render_parts.iter().enumerate() {
            let mesh = &map.meshes[part.mesh];
            let md = expand_mesh(&map, mesh).unwrap();
            let mut lo = [f32::MAX; 3];
            let mut hi = [f32::MIN; 3];
            for p in &md.pos {
                for k in 0..3 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
            }
            if lo[1] < 0.05 && hi[1] > 0.05 && hi[0] <= -18.0 && lo[0] >= -36.0
                && hi[2] <= -44.0 && lo[2] >= -62.0
            {
                list.push((
                    i,
                    lo[0],
                    hi[0],
                    lo[1],
                    hi[1],
                    lo[2],
                    hi[2],
                ));
            }
        }
        list.sort_by(|a, b| {
            (a.5 + a.6).partial_cmp(&(b.5 + b.6)).unwrap()
        });
        list.sort_by(|a, b| (a.1 + a.2).partial_cmp(&(b.1 + b.2)).unwrap());
        for (i, x0, x1, y0, y1, z0, z1) in list {
            let mat = &map.materials[map.render_parts[i].material];
            println!(
                "part {i}: mat {} blend={} x={:8.3}..{:8.3} y={:7.3}..{:7.3} z={:8.3}..{:8.3}",
                mat.id,
                mat.blend_mode(),
                x0,
                x1,
                y0,
                y1,
                z0,
                z1
            );
        }
        println!("--- end tiles ---");
    }

    let mut pairs: HashMap<(usize, usize), usize> = HashMap::new();
    for (i, qs) in quant.iter().enumerate() {
        if qs.len() > 2000 {
            continue;
        }
        let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for &q in qs {
            if let Some(pl) = cell.get(&q) {
                for &j in pl {
                    if j != i && !seen.contains(&j) {
                        seen.insert(j);
                        let key = if i < j { (i, j) } else { (j, i) };
                        *pairs.entry(key).or_insert(0) += 1;
                    }
                }
            }
        }
        // dedupe: count common verts, not common cells
    }
    // pairs map holds counts of shared QUANTIZED CELLS per pair; dedupe per vert:
    let mut shared_counts: HashMap<(usize, usize), usize> = HashMap::new();
    let mut near: Vec<((usize, usize), f64)> = Vec::new();
    for ((i, j), _cells) in &pairs {
        let qi = &quant[*i];
        let qj = &quant[*j];
        let si: std::collections::HashSet<(i64, i64, i64)> = qi.iter().copied().collect();
        let sj: std::collections::HashSet<(i64, i64, i64)> = qj.iter().copied().collect();
        let shared = si.intersection(&sj).count();
        if shared > 0 {
            *shared_counts.entry((*i, *j)).or_insert(0) += shared;
        } else {
            // near-miss: min distance between any two verts
            let mut best = f64::MAX;
            for a in qi {
                for b in qj {
                    let d = ((a.0 - b.0) as f64).powi(2)
                        + ((a.1 - b.1) as f64).powi(2)
                        + ((a.2 - b.2) as f64).powi(2);
                    if d < best {
                        best = d;
                    }
                }
            }
            let dist = (best as f64).sqrt() / SCALE as f64;
            if dist < 0.002 {
                near.push(((*i, *j), dist));
            }
        }
    }

    let exact = shared_counts.len();
    let mut exact_list: Vec<((usize, usize), usize)> =
        shared_counts.iter().map(|(k, v)| (*k, *v)).collect();
    exact_list.sort_by(|a, b| b.1.cmp(&a.1));
    println!("pairs sharing exact verts: {exact}");
    println!("top exact-shared pairs (i,j,count):");
    for ((i, j), c) in exact_list.iter().take(15) {
        println!("  {i},{j} shared={c}");
    }
    near.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    println!("near-miss pairs (min dist < 0.002, no exact): {}", near.len());
    for ((i, j), d) in near.iter().take(10) {
        println!("  {i},{j} min_dist={d:.5}");
    }
    if exact == 0 && near.is_empty() {
        println!("NO shared seams at all — every part is standalone");
    }
}