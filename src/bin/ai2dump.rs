use anyhow::{Context, Result};
use rustt::ai2;

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .context("usage: ai2dump <file.ai2>")?;
    let ai = ai2::parse_file(&path)?;

    println!("version: {}  paths: {}", ai.version, ai.paths.len());
    for (i, p) in ai.paths.iter().enumerate() {
        println!(
            "  path {i}: {:?}  {} points, {} connections",
            p.name,
            p.points.len(),
            p.connections.len()
        );
        for (j, pt) in p.points.iter().enumerate() {
            println!(
                "    point {j}: {:?} pos ({:.2}, {:.2}, {:.2}) xz {:.2}",
                pt.name, pt.pos.x, pt.pos.y, pt.pos.z, pt.xz_size
            );
        }
    }

    println!("triggers: {}", ai.triggers.len());
    for t in &ai.triggers {
        println!(
            "  {:?} pos ({:.2}, {:.2}, {:.2}) half ({:.2}, {:.2}, {:.2}) ang {:.1} @0x{:x}",
            t.name, t.pos.x, t.pos.y, t.pos.z, t.half_size.x, t.half_size.y, t.half_size.z,
            t.angle, t.offset
        );
    }

    println!("locators: {}", ai.locators.len());
    for l in &ai.locators {
        println!(
            "  {:?} pos ({:.2}, {:.2}, {:.2}) ang {:.1} @0x{:x}",
            l.name, l.pos.x, l.pos.y, l.pos.z, l.angle, l.offset
        );
    }

    println!("locator sets: {}", ai.locator_sets.len());
    for s in &ai.locator_sets {
        println!("  {:?} -> {:?}", s.name, s.locators);
    }

    println!("creatures: {}", ai.creatures.len());
    for c in &ai.creatures {
        println!(
            "  {:?} script {:?} type {:?} pos ({:.2}, {:.2}, {:.2}) ang {:.1} @0x{:x}",
            c.name, c.script, c.char_type, c.pos.x, c.pos.y, c.pos.z, c.angle, c.offset
        );
    }

    Ok(())
}
