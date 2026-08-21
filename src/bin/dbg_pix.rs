use std::collections::BTreeMap;
use std::env;

fn main() -> anyhow::Result<()> {
    let path = env::args().nth(1).unwrap_or_else(|| "shot.png".into());
    let dec = png::Decoder::new(std::fs::File::open(&path)?);
    let mut reader = dec.read_info()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;
    let w = info.width as usize;
    let h = info.height as usize;
    let bpp = info.color_type.samples() as usize;
    println!("img {}x{} bpp={}", w, h, bpp);

    let mut clusters: BTreeMap<(u8, u8, u8), u32> = BTreeMap::new();
    let mut olive = 0u32;
    let mut dark_olive = 0u32;
    for y in (0..h).step_by(3) {
        for x in (0..w).step_by(3) {
            let o = (y * w + x) * bpp;
            let (r, g, b) = (buf[o], buf[o + 1], buf[o + 2]);
            let k = ((r / 16) * 16, (g / 16) * 16, (b / 16) * 16);
            *clusters.entry(k).or_insert(0) += 1;
            if g > r && g >= b && r < 60 && b < 110 && g >= 40 && g <= 140 {
                olive += 1;
            }
            if g > r && r <= b.max(30) && g < 70 {
                dark_olive += 1;
            }
        }
    }
    let mut v: Vec<_> = clusters.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1));
    for (k, n) in v.iter().take(14) {
        println!("[{}, {}, {}] x{} ({:.1}%)", k.0, k.1, k.2, n, *n as f32 * 100.0 / (olive + 1) as f32);
    }
    println!("olive-ish px={} (sampled {:.2}% of frame)", olive, olive as f32 * 100.0 / (w as f32 / 3.0 * (h as f32 / 3.0)));
    println!("dark-olive px={}", dark_olive);
    Ok(())
}