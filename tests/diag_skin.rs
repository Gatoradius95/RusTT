use anyhow::Result;
use rustt::ghg;

fn at_i32(d: &[u8], o: i64) -> i32 {
    i32::from_le_bytes(d[o as usize..o as usize + 4].try_into().unwrap())
}

fn rel(d: &[u8], q: i64) -> Result<i64> {
    Ok(q + at_i32(d, q) as i64)
}

/// Extract the 0xff-terminated skin-bone list at descriptor +0x0a.
fn part_bones(data: &[u8], desc: usize) -> Vec<usize> {
    let mut v = Vec::new();
    for k in 0..10 {
        let b = data[desc + 0x0a + k];
        if b == 0xff {
            break;
        }
        v.push(b as usize);
    }
    v
}

/// Per-vertex skin decoding. Returns per-vertex (weights, local indices).
/// stride 44: weights @36, indices @40. stride 40: weights @32, indices @36.
fn decode_skin(
    vlist: &[u8],
    stride: usize,
    off_v: usize,
    num_v: usize,
) -> Vec<([u8; 4], [u8; 4])> {
    let (w_off, i_off) = match stride {
        44 => (36, 40),
        40 => (32, 36),
        _ => return Vec::new(),
    };
    let mut out = Vec::with_capacity(num_v);
    for v in 0..num_v {
        let base = off_v * stride + v * stride;
        let mut w = [0u8; 4];
        let mut idx = [0u8; 4];
        for k in 0..4 {
            w[k] = vlist[base + w_off + k];
            idx[k] = vlist[base + i_off + k];
        }
        out.push((w, idx));
    }
    out
}

#[test]
fn dump_skin_bones() -> Result<()> {
    let m = "backup/CHARS/ANAKIN/ANAKIN_PADAWAN_PC.GHG";
    let data = std::fs::read(m)?;
    let parsed = ghg::parse(&data)?;

    let num20 = at_i32(&data, 0) as i64;
    let head = num20 + 4 + 4 + 0xc;
    let abs_gsnh = head + 16 + at_i32(&data, head + 12) as i64;
    let gsnh = (abs_gsnh - 12) as usize;
    let mesh_meta = rel(&data, gsnh as i64 + 16 + 4 + 0x28)?;
    let mm = mesh_meta as usize;
    let number_parts = at_i32(&data, (mm + 0x14) as i64) as usize;
    let mut part_pos = mm + 0x14 + 4 + 0x08;

    // rest-world bone translations (chained local) = the frame parts live in
    let mut rest_pos = Vec::new();
    for b in &parsed.bones {
        rest_pos.push(b.world.w_axis.truncate());
    }

    let mut part_meta = Vec::new();
    for i in 0..number_parts {
        let offset_part = at_i32(&data, part_pos as i64) as i64;
        let desc = (part_pos as i64 + offset_part) as usize;
        let stride = i16::from_le_bytes([data[desc + 8], data[desc + 9]]) as usize;
        let off_v = at_i32(&data, (desc + 0x14) as i64) as usize;
        let num_v = at_i32(&data, (desc + 0x18) as i64) as usize;
        let vl = at_i32(&data, (desc + 0x24) as i64) as usize;
        part_meta.push((desc, stride, off_v, num_v, vl));
        part_pos += 4;
    }

    println!("bone rest-world positions:");
    for (i, b) in parsed.bones.iter().enumerate() {
        let p = rest_pos[i];
        println!(
            "  {i:2} {:<20} ({:+.3},{:+.3},{:+.3})",
            b.name, p.x, p.y, p.z
        );
    }

    for (i, (desc, stride, off_v, num_v, vl)) in part_meta.iter().enumerate() {
        let bones = part_bones(&data, *desc);
        if bones.is_empty() || (stride != &44 && stride != &40) {
            continue;
        }
        let vlist = parsed.vertex_lists[*vl];
        let skin = decode_skin(vlist, *stride, *off_v, *num_v);

        // scan position alignment: which offset within the vertex yields
        // plausible float triples for (nearly) all vertices?
        let mut best_off = 0usize;
        let mut best_cnt = 0u32;
        for off in (0..8).step_by(4) {
            let mut cnt = 0u32;
            for v in 0..*num_v {
                let base = *off_v * *stride + v * *stride + off;
                let x = f32::from_le_bytes(vlist[base..base + 4].try_into().unwrap());
                let y = f32::from_le_bytes(vlist[base + 4..base + 8].try_into().unwrap());
                let z = f32::from_le_bytes(vlist[base + 8..base + 12].try_into().unwrap());
                if x.is_finite() && y.is_finite() && z.is_finite()
                    && x.abs() < 1.0 && y.abs() < 2.0 && z.abs() < 2.0
                {
                    cnt += 1;
                }
            }
            if cnt > best_cnt {
                best_cnt = cnt;
                best_off = off;
            }
        }

        // mean position per decoded bone
        let mut sums = std::collections::HashMap::<usize, (f32, f32, f32, u32)>::new();
        let mut matched = 0u32;
        let mut sum_ok = 0u32;
        for v in 0..*num_v {
            let base = *off_v * *stride + v * *stride + best_off;
            let pos = (
                f32::from_le_bytes(vlist[base..base + 4].try_into().unwrap()),
                f32::from_le_bytes(vlist[base + 4..base + 8].try_into().unwrap()),
                f32::from_le_bytes(vlist[base + 8..base + 12].try_into().unwrap()),
            );
            let (w, idx) = skin[v];
            // dominant decoded bone: max weight among VALID (non-null) indices
            let mut dom = None;
            let mut dom_w = 0i32;
            let mut valid_w = 0u32;
            for k in 0..4 {
                let l = idx[k];
                if l != 0xff && (l as usize) < bones.len() {
                    valid_w += w[k] as u32;
                    if w[k] as i32 > dom_w {
                        dom_w = w[k] as i32;
                        dom = Some(l);
                    }
                }
            }
            if valid_w != 0 && valid_w <= 255 {
                sum_ok += 1;
            }
            if let Some(l) = dom {
                if l != 0xff && (l as usize) < bones.len() {
                    let g = bones[l as usize];
                    let e = sums.entry(g).or_insert((0.0, 0.0, 0.0, 0));
                    e.0 += pos.0;
                    e.1 += pos.1;
                    e.2 += pos.2;
                    e.3 += 1;
                    // nearest rest-world bone
                    let mut best = 0usize;
                    let mut best_d = f32::MAX;
                    for (bi, bp) in rest_pos.iter().enumerate() {
                        let d = (bp.x - pos.0).powi(2) + (bp.y - pos.1).powi(2) + (bp.z - pos.2).powi(2);
                        if d < best_d {
                            best_d = d;
                            best = bi;
                        }
                    }
                    if best == g {
                        matched += 1;
                    }
                }
            }
        }
        let n = *num_v as f32;
        let names = bones
            .iter()
            .map(|&b| parsed.bones[b].name.clone())
            .collect::<Vec<_>>();
        println!(
            "\npart {i} stride={stride} nv={num_v} vl={vl} off_v={off_v} bestPosOff={best_off} ({best_cnt}/{num_v}) bones[{:?}] {:?}",
            bones, names
        );
        println!(
            "  dominant-bone == nearest-rest-bone: {matched}/{} ({:.1}%)   weight-sums-in-0..=255: {sum_ok}/{num_v}",
            num_v,
            matched as f32 / n * 100.0
        );
        let mut keys: Vec<usize> = sums.keys().copied().collect();
        keys.sort();
        for g in keys {
            let (sx, sy, sz, c) = sums[&g];
            println!(
                "  -> bone {:2} {:<16} meanVtx=({:+.3},{:+.3},{:+.3}) restPos=({:+.3},{:+.3},{:+.3}) n={}",
                g,
                parsed.bones[g].name,
                sx / c as f32,
                sy / c as f32,
                sz / c as f32,
                rest_pos[g].x,
                rest_pos[g].y,
                rest_pos[g].z,
                c
            );
        }
    }
    Ok(())
}
