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
    let cols = 96usize;
    let rows = 48usize;
    let cw = w / cols;
    let ch = h / rows;
    let chrs = b" .:-=+*#%@";
    for r in 0..rows {
        let mut line = String::new();
        for c in 0..cols {
            let x0 = c * cw;
            let y0 = r * ch;
            let mut sum = 0u64;
            let mut n = 0u64;
            for y in y0..(y0 + ch).min(h) {
                for x in x0..(x0 + cw).min(w) {
                    let o = (y * w + x) * bpp;
                    sum += (buf[o] as u64) + (buf[o + 1] as u64) + (buf[o + 2] as u64);
                    n += 1;
                }
            }
            let l = sum / (3 * n.max(1));
            let idx = (l as usize * (chrs.len() - 1)) / 255;
            line.push(chrs[idx] as char);
        }
        println!("{}", line);
    }
    Ok(())
}