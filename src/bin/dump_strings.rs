//! One-off diagnostic: dump ASCII strings (>= min length) from a binary file,
//! optionally filtered by a case-insensitive substring.
//! Usage: dump_strings <file> [filter]

use std::io::Read;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: dump_strings <file> [filter]");
    let filter = args.next().unwrap_or_default().to_lowercase();

    let mut buf = Vec::new();
    std::fs::File::open(&path)
        .expect("open")
        .read_to_end(&mut buf)
        .expect("read");

    let mut start = None;
    for (i, &b) in buf.iter().enumerate() {
        let printable = (0x20..0x7f).contains(&b);
        match (printable, start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                if i - s >= 5 {
                    let text = std::str::from_utf8(&buf[s..i]).unwrap();
                    if filter.is_empty() || text.to_lowercase().contains(&filter) {
                        println!("{s:08x}  {text}");
                    }
                }
                start = None;
            }
            _ => {}
        }
    }
}
