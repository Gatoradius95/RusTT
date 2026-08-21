use anyhow::Result;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let ai_dir = args.get(1).map(|s| s.as_str()).unwrap_or("backup/LEVELS/MAP/MAP/AI");
    let map_txt = args.get(2).map(|s| s.as_str()).unwrap_or("backup/LEVELS/MAP/MAP/MAP.TXT");

    println!("=== SCP files from {ai_dir} ===");
    let mut ok = 0usize;
    let mut fail = 0usize;
    for entry in std::fs::read_dir(ai_dir)? {
        let entry = entry?;
        let path = entry.path();
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if !ext.eq_ignore_ascii_case("SCP") {
            continue;
        }
        let name = path.file_name().unwrap().to_string_lossy();
        let data = std::fs::read_to_string(&path)?;
        match rustt::scp::parse(&data) {
            Ok(script) => {
                let total_conds: usize = script.states.iter().map(|s| s.conditions.len()).sum();
                let total_actions: usize = script.states.iter().map(|s| s.actions.len()).sum();
                let total_refs: usize = script.states.iter().map(|s| s.reference_scripts.len()).sum();
                let ref_conds: usize = script.states.iter()
                    .flat_map(|s| &s.reference_scripts)
                    .map(|r| r.conditions.len())
                    .sum();
                println!(
                    "OK   {name:<20} states={:<3} conds={:<4} actions={:<4} ref_scripts={:<3} ref_conds={:<4}",
                    script.states.len(), total_conds, total_actions, total_refs, ref_conds
                );
                ok += 1;
            }
            Err(e) => {
                println!("FAIL {name:<20} {e}");
                fail += 1;
            }
        }
    }
    println!("{ok} ok, {fail} failed\n");

    println!("=== MAP.TXT from {map_txt} ===");
    let data = std::fs::read_to_string(map_txt)?;
    match rustt::map_txt::parse(&data) {
        Ok(map) => {
            println!("OK");
            println!("  settings: {:?}", map.settings);
            println!("  socks (cameras): {}", map.socks.len());
            for (i, s) in map.socks.iter().enumerate() {
                println!("    [{i}] id={:?} params={:?}", s.id, s.params);
            }
            println!("  doors: {}", map.doors.len());
            for (i, d) in map.doors.iter().enumerate() {
                println!("    [{i}] spline={:?} level={:?} one_way={} two_player={}",
                    d.spline, d.level, d.one_way, d.two_player_only);
            }
            println!("  obstacles: {}", map.obstacles.len());
            for (i, o) in map.obstacles.iter().enumerate() {
                println!("    [{i}] name={:?} objs={} range={:?} open_close={}",
                    o.name, o.obj.len(), o.range, o.play_open_close);
            }
            println!("  buildits: {}", map.buildits.len());
            for (i, b) in map.buildits.iter().enumerate() {
                println!("    [{i}] name={:?} pairs={} coin={:?}",
                    b.name, b.pairs.len(), b.coin_value);
            }
            println!("  blowups: {}", map.blowups.len());
            for (i, b) in map.blowups.iter().enumerate() {
                println!("    [{i}] force_name={:?} objs={} deb={} hp={:?}",
                    b.force_name, b.obj.len(), b.deb_names.len(), b.hit_points);
            }
            println!("  turrets: {}", map.turrets.len());
            for (i, t) in map.turrets.iter().enumerate() {
                println!("    [{i}] name={:?} obj={:?} range=({},{}) fire_int={:?} hp={}",
                    t.name, t.obj, t.view_range, t.fire_range, t.fire_interval, t.hitpoints);
            }
        }
        Err(e) => {
            println!("FAIL: {e}");
        }
    }

    Ok(())
}
