use std::collections::HashMap;

use anyhow::Result;

pub struct MapTxt {
    pub settings: HashMap<String, String>,
    pub doors: Vec<Door>,
    pub socks: Vec<Sock>,
    pub obstacles: Vec<Obstacle>,
    pub buildits: Vec<Buildit>,
    pub blowups: Vec<Blowup>,
    pub turrets: Vec<Turret>,
}

pub struct Door {
    pub spline: String,
    pub level: String,
    pub one_way: bool,
    pub two_player_only: bool,
}

pub struct Sock {
    pub id: String,
    pub params: HashMap<String, String>,
}

pub struct Obstacle {
    pub name: String,
    pub obj: Vec<ObjEntry>,
    pub range: f32,
    pub play_open_close: bool,
    pub play_once: bool,
    pub active_players_only: bool,
    pub auto_start: bool,
    pub anim_speed: Option<f32>,
    pub camera_spline: Option<String>,
    pub cam_start_time: Option<f32>,
    pub cam_end_time: Option<f32>,
    pub cam_blend_in_time: Option<f32>,
    pub cam_blend_out_time: Option<f32>,
    pub chain: Option<(u32, u32)>,
    pub chain_trigger: Option<(u32, u32)>,
    pub buildit_ref: Option<String>,
    pub sfx_open: Option<String>,
    pub sfx_close: Option<String>,
}

pub struct ObjEntry {
    pub name: String,
    pub no_collision: bool,
    pub trigger: bool,
    pub off: bool,
    pub reset_force_flags: bool,
}

pub struct Buildit {
    pub name: String,
    pub pairs: Vec<(String, String)>,
    pub coin_value: Option<u32>,
    pub push_from_pieces: bool,
    pub clunk_angle: Option<f32>,
}

pub struct Blowup {
    pub set_type: Option<String>,
    pub use_type: Option<String>,
    pub force_name: String,
    pub obj: Vec<String>,
    pub deb_names: Vec<String>,
    pub part_effect_name: Option<String>,
    pub hit_points: Option<u32>,
    pub vehicle_only: bool,
}

pub struct Turret {
    pub obj: String,
    pub name: String,
    pub x_turn_factor: f32,
    pub y_turn_factor: f32,
    pub view_range: f32,
    pub fire_range: f32,
    pub fire_interval: f32,
    pub hitpoints: u32,
    pub fire_offsets: [(f32, f32, f32); 2],
    pub range_x_rot: f32,
    pub range_y_rot: f32,
    pub follow_player: bool,
}

fn strip_comments(line: &str) -> &str {
    let s = line.trim();
    if let Some(p) = s.find("//") {
        s[..p].trim()
    } else if let Some(p) = s.find(';') {
        s[..p].trim()
    } else {
        s
    }
}

fn parse_kv(line: &str) -> (String, String) {
    let s = line.trim();
    if let Some(eq) = s.find('=') {
        let k = s[..eq].trim().to_string();
        let v = s[eq + 1..].trim().to_string();
        (k, v)
    } else if let Some(sp) = s.find(char::is_whitespace) {
        let k = s[..sp].trim().to_string();
        let v = s[sp + 1..].trim().to_string();
        (k, v)
    } else {
        (s.to_string(), String::new())
    }
}

fn unquote(s: &str) -> String {
    let t = s.trim();
    if t.starts_with('"') && t.ends_with('"') && t.len() >= 2 {
        t[1..t.len() - 1].to_string()
    } else {
        t.to_string()
    }
}

pub fn parse(data: &str) -> Result<MapTxt> {
    let mut settings = HashMap::new();
    let mut doors = Vec::new();
    let mut socks = Vec::new();
    let mut obstacles = Vec::new();
    let mut buildits = Vec::new();
    let mut blowups = Vec::new();
    let mut turrets = Vec::new();

    let mut in_settings = true;

    let lines: Vec<&str> = data.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let line = strip_comments(lines[i]);
        i += 1;
        if line.is_empty() {
            continue;
        }

        // Settings section: everything before "settings_end" is key-value pairs.
        if in_settings {
            if line == "settings_end" {
                in_settings = false;
                continue;
            }
            let (k, v) = parse_kv(line);
            if !k.is_empty() {
                settings.insert(k, v);
            }
            continue;
        }

        // Door blocks
        if line == "door_start" {
            let mut door = Door {
                spline: String::new(),
                level: String::new(),
                one_way: false,
                two_player_only: false,
            };
            while i < lines.len() {
                let l = strip_comments(lines[i]);
                i += 1;
                if l.is_empty() {
                    continue;
                }
                if l == "door_end" {
                    break;
                }
                let (k, v) = parse_kv(l);
                match k.as_str() {
                    "spline" => door.spline = unquote(&v),
                    "level" => door.level = unquote(&v),
                    "1_way" => door.one_way = true,
                    "2_player_only" => door.two_player_only = true,
                    _ => {}
                }
            }
            doors.push(door);
            continue;
        }

        // Sock blocks (camera)
        if line.starts_with("sock_start") {
            let id = line.strip_prefix("sock_start").unwrap_or("").trim().to_string();
            let mut params = HashMap::new();
            while i < lines.len() {
                let l = strip_comments(lines[i]);
                i += 1;
                if l.is_empty() {
                    continue;
                }
                if l == "sock_end" {
                    break;
                }
                let (k, v) = parse_kv(l);
                params.insert(k, v);
            }
            socks.push(Sock { id, params });
            continue;
        }

        // Obstacle blocks
        if line == "obstacle_start" {
            let mut obj_entries = Vec::new();
            let mut obs = Obstacle {
                name: String::new(),
                obj: Vec::new(),
                range: 0.0,
                play_open_close: false,
                play_once: false,
                active_players_only: false,
                auto_start: false,
                anim_speed: None,
                camera_spline: None,
                cam_start_time: None,
                cam_end_time: None,
                cam_blend_in_time: None,
                cam_blend_out_time: None,
                chain: None,
                chain_trigger: None,
                buildit_ref: None,
                sfx_open: None,
                sfx_close: None,
            };
            while i < lines.len() {
                let l = strip_comments(lines[i]);
                i += 1;
                if l.is_empty() {
                    continue;
                }
                if l == "obstacle_end" {
                    break;
                }
                let (k, v) = parse_kv(l);
                match k.as_str() {
                    "name" => obs.name = unquote(&v),
                    "range" => obs.range = v.parse().unwrap_or(0.0),
                    "play_openclose" => obs.play_open_close = true,
                    "play_once" => obs.play_once = true,
                    "active_players_only" => obs.active_players_only = true,
                    "auto_start" => obs.auto_start = true,
                    "anim_speed" => obs.anim_speed = v.parse().ok(),
                    "camera_spline" => obs.camera_spline = Some(unquote(&v)),
                    "cam_start_time" => obs.cam_start_time = v.parse().ok(),
                    "cam_end_time" => obs.cam_end_time = v.parse().ok(),
                    "cam_blend_in_time" => obs.cam_blend_in_time = v.parse().ok(),
                    "cam_blend_out_time" => obs.cam_blend_out_time = v.parse().ok(),
                    "sfx_open" => obs.sfx_open = Some(unquote(&v)),
                    "sfx_close" => obs.sfx_close = Some(unquote(&v)),
                    "chain" => {
                        // Format: "4,1" or "4,1,2"
                        let parts: Vec<&str> = v.split(',').collect();
                        if parts.len() >= 2 {
                            obs.chain = Some((
                                parts[0].parse().unwrap_or(0),
                                parts[1].parse().unwrap_or(0),
                            ));
                        }
                    }
                    "chain_trigger" => {
                        let parts: Vec<&str> = v.split(',').collect();
                        if parts.len() >= 2 {
                            obs.chain_trigger = Some((
                                parts[0].parse().unwrap_or(0),
                                parts[1].parse().unwrap_or(0),
                            ));
                        }
                    }
                    "buildit" => obs.buildit_ref = Some(unquote(&v)),
                    "obj" => {
                        // Parse "name",no_collision or "name",trigger etc
                        let raw = unquote(&v);
                        let mut parts: Vec<&str> = raw.split(',').collect();
                        let name = parts.remove(0).to_string();
                        let no_collision = parts.iter().any(|p| *p == "no_collision");
                        let trigger = parts.iter().any(|p| *p == "trigger");
                        let off = parts.iter().any(|p| *p == "off");
                        let reset_force_flags = parts.iter().any(|p| *p == "reset_force_flags");
                        obj_entries.push(ObjEntry {
                            name,
                            no_collision,
                            trigger,
                            off,
                            reset_force_flags,
                        });
                    }
                    _ => {}
                }
            }
            obs.obj = obj_entries;
            obstacles.push(obs);
            continue;
        }

        // Buildit blocks
        if line == "buildit_start" {
            let mut buildit = Buildit {
                name: String::new(),
                pairs: Vec::new(),
                coin_value: None,
                push_from_pieces: false,
                clunk_angle: None,
            };
            while i < lines.len() {
                let l = strip_comments(lines[i]);
                i += 1;
                if l.is_empty() {
                    continue;
                }
                if l == "buildit_end" {
                    break;
                }
                let (k, v) = parse_kv(l);
                match k.as_str() {
                    "name" => buildit.name = unquote(&v),
                    "pair" => {
                        // Format: "name1","name2"
                        let parts: Vec<&str> = v.split(',').collect();
                        if parts.len() >= 2 {
                            buildit
                                .pairs
                                .push((unquote(parts[0]), unquote(parts[1])));
                        }
                    }
                    "coin_value" => buildit.coin_value = v.parse().ok(),
                    "push_from_pieces" => buildit.push_from_pieces = true,
                    "clunk_angle" => buildit.clunk_angle = v.parse().ok(),
                    _ => {}
                }
            }
            buildits.push(buildit);
            continue;
        }

        // Blowup blocks
        if line == "blowup_start" {
            let mut blowup = Blowup {
                set_type: None,
                use_type: None,
                force_name: String::new(),
                obj: Vec::new(),
                deb_names: Vec::new(),
                part_effect_name: None,
                hit_points: None,
                vehicle_only: false,
            };
            while i < lines.len() {
                let l = strip_comments(lines[i]);
                i += 1;
                if l.is_empty() {
                    continue;
                }
                if l == "blowup_end" {
                    break;
                }
                let (k, v) = parse_kv(l);
                match k.as_str() {
                    "set_type" => blowup.set_type = Some(unquote(&v)),
                    "use_type" => blowup.use_type = Some(unquote(&v)),
                    "force_name" => blowup.force_name = unquote(&v),
                    "obj" => blowup.obj.push(unquote(&v)),
                    "deb_name" => blowup.deb_names.push(unquote(&v)),
                    "part_effect_name" => blowup.part_effect_name = Some(unquote(&v)),
                    "hit_points" => blowup.hit_points = v.parse().ok(),
                    "vehicle_only" => blowup.vehicle_only = true,
                    _ => {}
                }
            }
            blowups.push(blowup);
            continue;
        }

        // Turret blocks
        if line == "turret_start" {
            let mut turret = Turret {
                obj: String::new(),
                name: String::new(),
                x_turn_factor: 0.0,
                y_turn_factor: 0.0,
                view_range: 0.0,
                fire_range: 0.0,
                fire_interval: 0.0,
                hitpoints: 0,
                fire_offsets: [(0.0, 0.0, 0.0); 2],
                range_x_rot: 0.0,
                range_y_rot: 0.0,
                follow_player: false,
            };
            while i < lines.len() {
                let l = strip_comments(lines[i]);
                i += 1;
                if l.is_empty() {
                    continue;
                }
                if l == "turret_end" {
                    break;
                }
                let (k, v) = parse_kv(l);
                match k.as_str() {
                    "obj" => turret.obj = unquote(&v),
                    "name" => turret.name = unquote(&v),
                    "x_turn_factor" => turret.x_turn_factor = v.parse().unwrap_or(0.0),
                    "y_turn_factor" => turret.y_turn_factor = v.parse().unwrap_or(0.0),
                    "view_range" => turret.view_range = v.parse().unwrap_or(0.0),
                    "fire_range" => turret.fire_range = v.parse().unwrap_or(0.0),
                    "fire_interval" => turret.fire_interval = v.parse().unwrap_or(0.0),
                    "hitpoints" => turret.hitpoints = v.parse().unwrap_or(0),
                    "fire_offset_x" => turret.fire_offsets[0].0 = v.parse().unwrap_or(0.0),
                    "fire_offset_y" => turret.fire_offsets[0].1 = v.parse().unwrap_or(0.0),
                    "fire_offset_z" => turret.fire_offsets[0].2 = v.parse().unwrap_or(0.0),
                    "fire_offset2_x" => turret.fire_offsets[1].0 = v.parse().unwrap_or(0.0),
                    "fire_offset2_y" => turret.fire_offsets[1].1 = v.parse().unwrap_or(0.0),
                    "fire_offset2_z" => turret.fire_offsets[1].2 = v.parse().unwrap_or(0.0),
                    "range_x_rot" => turret.range_x_rot = v.parse().unwrap_or(0.0),
                    "range_y_rot" => turret.range_y_rot = v.parse().unwrap_or(0.0),
                    "follow_player" => turret.follow_player = true,
                    _ => {}
                }
            }
            turrets.push(turret);
            continue;
        }
    }

    Ok(MapTxt {
        settings,
        doors,
        socks,
        obstacles,
        buildits,
        blowups,
        turrets,
    })
}

pub fn parse_file(path: &str) -> Result<MapTxt> {
    let data = std::fs::read_to_string(path)?;
    parse(&data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_settings() {
        let input = r#"max_doors 80
fogr 200
fogdensity 0.110
settings_end
"#;
        let m = parse(input).unwrap();
        assert_eq!(m.settings.get("max_doors").unwrap(), "80");
        assert_eq!(m.settings.get("fogr").unwrap(), "200");
        assert_eq!(m.settings.get("fogdensity").unwrap(), "0.110");
    }

    #[test]
    fn parse_door() {
        let input = r#"settings_end
door_start
    spline "door_to_E1_01"
    level "negotiations_a"
    1_way
door_end
"#;
        let m = parse(input).unwrap();
        assert_eq!(m.doors.len(), 1);
        assert_eq!(m.doors[0].spline, "door_to_E1_01");
        assert_eq!(m.doors[0].level, "negotiations_a");
        assert!(m.doors[0].one_way);
    }

    #[test]
    fn parse_sock() {
        let input = r#"settings_end
sock_start 00
    cam_dist_to_target 2
    cam_height_above_terrain=0.8
sock_end
"#;
        let m = parse(input).unwrap();
        assert_eq!(m.socks.len(), 1);
        assert_eq!(m.socks[0].id, "00");
        assert_eq!(m.socks[0].params.get("cam_dist_to_target").unwrap(), "2");
        assert_eq!(
            m.socks[0].params.get("cam_height_above_terrain").unwrap(),
            "0.8"
        );
    }

    #[test]
    fn parse_obstacle() {
        let input = r#"settings_end
obstacle_start
    name "DE1"
    play_openclose
    obj "door_e1"
    range 0.9
    active_players_only
obstacle_end
"#;
        let m = parse(input).unwrap();
        assert_eq!(m.obstacles.len(), 1);
        let obs = &m.obstacles[0];
        assert_eq!(obs.name, "DE1");
        assert!(obs.play_open_close);
        assert_eq!(obs.range, 0.9);
        assert!(obs.active_players_only);
        assert_eq!(obs.obj.len(), 1);
        assert_eq!(obs.obj[0].name, "door_e1");
    }

    #[test]
    fn parse_buildit() {
        let input = r#"settings_end
buildit_start
    name "e4bonus"
    pair "door_1_1_1","door_1_2_1"
    pair "door_1_1_2","door_1_2_2"
    coin_value 1250
    push_from_pieces
    clunk_angle 0
buildit_end
"#;
        let m = parse(input).unwrap();
        assert_eq!(m.buildits.len(), 1);
        let b = &m.buildits[0];
        assert_eq!(b.name, "e4bonus");
        assert_eq!(b.pairs.len(), 2);
        assert_eq!(b.coin_value, Some(1250));
        assert!(b.push_from_pieces);
        assert_eq!(b.clunk_angle, Some(0.0));
    }

    #[test]
    fn parse_blowup() {
        let input = r#"settings_end
blowup_start
    set_type "chair"
    force_name "chair_01"
    obj "chair_01_shadow",off
    deb_name "chair_pop_1"
    deb_name "spark_pop_2"
    part_effect_name "gen_part_1"
blowup_end
"#;
        let m = parse(input).unwrap();
        assert_eq!(m.blowups.len(), 1);
        let b = &m.blowups[0];
        assert_eq!(b.set_type.as_deref(), Some("chair"));
        assert_eq!(b.force_name, "chair_01");
        assert_eq!(b.obj.len(), 1);
        assert_eq!(b.deb_names.len(), 2);
        assert_eq!(b.part_effect_name.as_deref(), Some("gen_part_1"));
    }

    #[test]
    fn parse_comments() {
        let input = r#"settings_end
// This is a comment
door_start
    spline "test" // inline comment
    level "map"
door_end
; another comment
"#;
        let m = parse(input).unwrap();
        assert_eq!(m.doors.len(), 1);
        assert_eq!(m.doors[0].spline, "test");
    }
}
