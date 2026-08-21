use rustt::ghg::parse;

fn main() {
    let path = std::env::args().nth(1).expect("usage: layerinfo <file.ghg>");
    let data = std::fs::read(&path).expect("read file");
    let parsed = parse(&data).expect("parse");
    let num20 = i32::from_le_bytes(data[0..4].try_into().unwrap());
    let header = num20 as i64;
    let mut head = header + 4 + 4 + 0xc;
    let mut p = head as usize;
    let _chunk_size = u32::from_le_bytes(data[p + 4..p + 8].try_into().unwrap()) as usize;
    let abs_gsnh = p as i64 + 16 + i32::from_le_bytes(data[p + 12..p + 16].try_into().unwrap()) as i64;
    p += 16;
    let gsnh = (abs_gsnh - 12) as usize;
    let bones_base = gsnh as i64 + 16 + 4 + 0x28 + 4 + 0x130;
    let number_layer = i32::from_le_bytes(data[(bones_base + 16 + 24) as usize..(bones_base + 20 + 24) as usize].try_into().unwrap()) as usize;
    let rel = |q: i64| -> i64 { q + i32::from_le_bytes(data[q as usize..q as usize + 4].try_into().unwrap()) as i64 };
    let abs_layer = rel(bones_base + 20 + 24);
    println!("parts={} bones={} number_layer={number_layer}", parsed.parts.len(), parsed.bones.len());
    let mut layer_pos = abs_layer;
    let mut total_items = 0usize;
    for li in 0..number_layer {
        let _text = rel(layer_pos);
        let mut q = layer_pos + 4;
        let mut slots = [0i64; 4];
        for slot in slots.iter_mut() {
            let tmp = i32::from_le_bytes(data[q as usize..q as usize + 4].try_into().unwrap());
            if tmp != 0 {
                *slot = rel(q);
            }
            q += 4;
        }
        // count render items per slot
        let mut per: Vec<usize> = Vec::new();
        let mut count_slot = |pos0: i64| -> usize {
            if pos0 == 0 { return 0; }
            let mut c = 0usize;
            let mut p = pos0;
            // per_bone variant: bone list
            let n_bones = parsed.bones.len();
            for _ in 0..n_bones {
                let tmp = i32::from_le_bytes(data[p as usize..p as usize + 4].try_into().unwrap());
                p += 4;
                if tmp != 0 {
                    let mut q = p + tmp as i64 - 4;
                    q += 8;
                    q = rel(q);
                    q += 0xb0;
                    q = rel(q);
                    let pn = i32::from_le_bytes(data[q as usize..q as usize + 4].try_into().unwrap()) as usize;
                    c += pn;
                }
            }
            c
        };
        // can't easily detect per_bone vs not; try per_bone first then fallback
        let a = count_slot(slots[0]);
        let _ = a;
        // Actually read the same way parse does but count.
        let mut c0 = 0usize;
        let mut p = slots[0];
        if p != 0 {
            let n_bones = parsed.bones.len();
            for _ in 0..n_bones {
                let tmp = i32::from_le_bytes(data[p as usize..p as usize + 4].try_into().unwrap());
                p += 4;
                if tmp != 0 {
                    let mut q = p + tmp as i64 - 4;
                    q += 8; q = rel(q); q += 0xb0; q = rel(q);
                    let pn = i32::from_le_bytes(data[q as usize..q as usize + 4].try_into().unwrap()) as usize;
                    c0 += pn;
                }
            }
        }
        let mut c1 = 0usize;
        let mut p = slots[1];
        if p != 0 {
            let mut q = p + 8; q = rel(q); q += 0xb0; q = rel(q);
            let pn = i32::from_le_bytes(data[q as usize..q as usize + 4].try_into().unwrap()) as usize;
            c1 += pn;
        }
        let mut c2 = 0usize;
        let mut p = slots[2];
        if p != 0 {
            let n_bones = parsed.bones.len();
            for _ in 0..n_bones {
                let tmp = i32::from_le_bytes(data[p as usize..p as usize + 4].try_into().unwrap());
                p += 4;
                if tmp != 0 {
                    let mut q = p + tmp as i64 - 4;
                    q += 8; q = rel(q); q += 0xb0; q = rel(q);
                    let pn = i32::from_le_bytes(data[q as usize..q as usize + 4].try_into().unwrap()) as usize;
                    c2 += pn;
                }
            }
        }
        let mut c3 = 0usize;
        let mut p = slots[3];
        if p != 0 {
            let mut q = p + 8; q = rel(q); q += 0xb0; q = rel(q);
            let pn = i32::from_le_bytes(data[q as usize..q as usize + 4].try_into().unwrap()) as usize;
            c3 += pn;
        }
        let sum = c0 + c1 + c2 + c3;
        println!("layer {li}: items={sum}  slot0(perbone)={c0}  slot1={c1}  slot2(perbone)={c2}  slot3={c3}  range=part[{total_items}..{})", total_items + sum);
        total_items += sum;
        layer_pos = q;
    }
    println!("total items across layers: {total_items}");
}
