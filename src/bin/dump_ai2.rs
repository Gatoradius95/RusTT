use rustt::ai2::parse_file;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ai = parse_file("backup/LEVELS/MAP/MAP/AI/MAP.AI2")?;
    println!("== TRIGGERS == ({})", ai.triggers.len());
    for t in &ai.triggers {
        println!(
            "{:<16} pos=({:>8.2},{:>8.2},{:>8.2}) half=({:>6.2},{:>6.2},{:>6.2}) angle={:.1}",
            t.name, t.pos.x, t.pos.y, t.pos.z, t.half_size.x, t.half_size.y, t.half_size.z, t.angle
        );
    }
    println!("\n== LOCATORS == ({})", ai.locators.len());
    for l in &ai.locators {
        println!(
            "{:<20} pos=({:>8.2},{:>8.2},{:>8.2}) angle={:.1}",
            l.name, l.pos.x, l.pos.y, l.pos.z, l.angle
        );
    }
    println!("\n== PATH SAMPLE (names/points) ==");
    for p in &ai.paths {
        let pts: Vec<String> = p.points.iter().map(|q| format!("{}@({:.1},{:.1},{:.1})", q.name, q.pos.x, q.pos.y, q.pos.z)).collect();
        println!("{}: {} points: {}", p.name, p.points.len(), pts.join(", "));
    }
    Ok(())
}
