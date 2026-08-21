RusTT — A native reimplementation of the TT Games Nu2 engine

## 1. Overview

LEGO Star Wars, LEGO Indiana Jones, LEGO Batman, LEGO Star Wars II, and several other LEGO games from the mid-2000s were built by Traveller's Tales on an internal engine called Nu2. TT isn't Valve or Bethesda. They shipped at least 1 LEGO game a year for many years on multiple patforms, so mod support was the last thing on their minds. On Windows the games shipped as a bare executable plus a set of asset folders inside a propietary file format — no public SDK and a mostly undocumented file format family (GHG models, AN3 skeleton animation, BSA blend-shape animation, ".PAK" containers, ".SCP" scripts). This is without counting the numerous hard-coded systems like limits on the character grid, the game's overreliance on text files, and generally weird technical decisions. Modding is an uphill battle against all of these factors, and I doubt it will get much better if we have to keep looking for ways around the engine's limitations. that's why I want to give a shot at this herculean effort.

This project aims to natively reimplement that engine in Rust, so that these games can run on modern systems without the original binary, as well as having a base that modders can use to mod to their heart's content. The target and benchmark is LEGO Star Wars: The Complete Saga: the goal is to reach a state where the game is playable from start to finish — all six episodes, levels, puzzles, and bosses beatable. After that goal is reached, the next step will be rebuilding its systems on a way that will make modding easier. One of my hopes is to one day get rid of "COLLECTIONS.TXT", "CHARS.TXT" and other text files that have haunted the dreams of any experienced modder in the community. My idea is to replace them with a modular, JSON-based system. Every character would have their own JSON file with configurable basic info, using already known tags used by the game. This should remove the hassle of having to manually edit txt files when you want to install multiple character mods, and should make it easy for the game to parse both existing and modded characters. This idea is a WIP though, so I'm open to suggestions.

(The following only applies to my PC and won't be avaliable once I make the code public for obvious reasons): A real install of LSW:TCS lives in "Lego Star Wars/" and serves as both the
data source and the ground-truth reference; "backup/" holds pristine copies of the same assets for recovery. "research/" contains community format documentation (Blender add-ons, EasyAN3, BactaTank, BrickBench) used to reverse the formats (all credits to these amazing developers: AlubJ, 2mt9/pessoa, Clarence Oveur, MattonMat; that provided the code for many tools I used to better understand TT's... Quirky file formats).

The project is currently an asset-format toolkit + interactive viewer plus a gameplay test bed, and a Ghidra reverse-engineering effort against the original PC executable. The "game" bin loads the actual cantina hub ("MAP_PC.GSC") and lets you walk around with a chase camera; reverse-engineering notes live in "research/ghidra docs.md", cross-checked against the Android decompilation in "research/saga".

## 2. What works today

Library ("src/lib.rs" + modules)
- "ghg" — full GHG model parser: materials, DXT textures, parts, render items with their LOD/layer indices, the skeleton, and per-slot blend-shape (morph) delta data.
- "an3" — (WIP and Imperfect) AN3 skeletal animation parser and playback math (ANI4/ANI6/ANI8 versions), verified byte-for-byte against real files.
- "bsa" — (VERY WIP)BSA blend-shape weight animation parser and channel evaluation (hermite keyframes), verified against "backup/CHARS/".
- "dxt" — DXT texture decompression.
- "glb" — glTF/GLB export (mesh + skeleton + morph data + textures).
- "map" — GSC map parser: textures, vertex/index buffers, materials, meshes, render_parts with per-part names (from NTBL block), room-based culling for SO entities, and GIZ blowup overlay support.
- "map_txt" — VERY BASIC MAP.TXT config parser: locators, obstacles, blowups, buildits, doors, socks, settings.
- "ai2" — VERY BASIC AI2 trigger/locator parser: room AABB triggers with name/position/half_size/angle/offset, creature/locator spawns.
- "giz" — WIP GIZ file parser (reverse-engineered from "FUN_0055ad70"):
   "GizObstacle" block (110 obstacles): name, position, rotation, scale, sub-objects with flags — full version-dependent field parsing.
   "blowup" block (196 instances, 14 template types): length-prefixed template/instance names, Vec3 position, 3× i16 Euler rotation (65536-unit circle, at record offset +0x0C), plus metadata.
   "match_blowups_to_sos()" matches GIZ templates to GSC SO entities by name, with "mesh_overrides" for SOs whose mesh is embedded in room geometry (cmd_count=0).
   "apply_blowup_positions()" replaces/creates render_parts at GIZ positions with correct Y-rotation (Euler angles composed into 4×4 transform).

Viewer ("viewer" binary) — an interactive wgpu-based renderer:
- Real-time rendering of skinned meshes (GPU skinning from AN3 playback).
- GPU morph blend shapes driven live by a BSA clip (facial animation).
- Orbit camera + WASD/QE fly, scroll zoom, wireframe overlay, grid toggle, "apply bind pose" toggle.
- UI panels: Scene, Materials, Bones, Textures, Animation.
- Quality/LOD layer selection that matches the game: reads the sibling ".TXT" ("layers_special/high/medium/low/dead") and draws only the selected set, so a minifig renders as the game renders it (one body, no duplicate LOD limbs). Handles platform-suffixed names ("BOBAFETT_PC.GHG" → "BOBAFETT.TXT") and "txt_file="base"" inheritance for variants.
- Hub lighting (lightmaps): per-material "LIGHTMAP_STAGE" detection (MS00 record +0x26E, "lightmap_set_index" +0x15C + texture-flag sign bit), the LM0..2 texture set bound from the set index, lightmap UVs from the material's UV set, and the "lightmapOffset" transform ("lightmapCoord = uv  off.zw + off.xy"). Draw calls that appear under a DISP "LIGHTMAP" display command (0xb0) additionally bind the command's own page textures + offset uniform per part (type 1/2 = offset only, 3/4 = full offset/scale vector) — 239/375 materials staged on the cantina hub. "VIEWER_LM_OFF" A/B-tests the lightmap set against the baked vertex-light fallback.
- Material alpha state (MS00 record +0x40): blend mode (low nibble: 0 none, 1 src-alpha/1-src-alpha, 2 src-alpha/one, 3 reverse, 10 none-fixed-alpha) and depth mode (bits 14 -15: 0 normal, 1 no depth write, 2 always pass, 3 ignore depth). The renderer splits meshes by this state: blend 0/1 with normal depth stay in the opaque pass (masked-glass windows write depth), blend 2 (TRANSPARENT_IGNORE_DEST — the cantina strobe/mood panels and lights) gets an additive SRC_ALPHA/ONE pipeline with depth write following the depth mode, and the rest share the plain transparent pipeline. The band stage's black strobe backdrop no longer covers the wall behind it ("shots/blend_on2.png" A/B: +17% saturated pixels on the stage, the colored dots now add over the lit wall).
- AN3 playback with speed control, loop, and frame scrubbing.
- GIZ integration: auto-loads sibling ".GIZ" when opening a ".GSC" map, applies blowup instance positions + rotations (furniture, destructibles) and obstacle positions (doors, contraptions) to render_parts.

CLI tools
- "rustt" (convert): GHG → GLB.
- "dump", "layerinfo", "bsa_dump": diagnostics for models, LOD layer layout,
  and blend-shape channels.
- "pose_render": offscreen skeleton-pose renderer used to settle bone/leg
  orientation questions.

Tests (all green): AN3 smoke + loop-continuity, BSA evaluation against real files, a naga/WGSL stride regression test for the morph buffer, viewer TXT-discovery tests (suffix probing + "txt_file" inheritance), and GIZ tests (110 obstacles parsed, 196 blowup instances with rotation, GIZ→MAP.TXT→SO matching validated, blowup position+rotation application with mesh overrides).

Gameplay test bed ("game" binary, "src/bin/game/main.rs")
- Loads the Mos Eisley cantina hub plus a minifig, and runs a walkable third-person slice:
- Per-mesh bounds culling so only the current room renders (the whole-hub draw was the big framerate drain).
- Chase camera with ray-vs-mesh collision (Möller–Trumbore against the map's bounds-sorted triangles), so it stays inside the small hub rooms. This is obviously just for debug reasons and will eventually be replaced by an implementation of the game's actual camera system.
- Animation/camera/scale ground-truthed from the real assets: "PLAYER_SCALE" and walk speed/fps from "CHARS/ANAKIN/ANAKIN_JEDI.TXT", camera distance 2 / height 0.8 from "LEVELS/MAP/MAP/MAP.TXT" ("cam_dist_to_target", "cam_height_above_terrain").
- "--shot"/"--walk" headless screenshot flags for A/B verification.
- GIZ integration: auto-loads sibling ".GIZ" after map parsing, applies blowup positions + rotations and obstacle positions to render_parts.
- Buildit jibber particles: spawns a short-lived cross-flare glow sprite at each sub-object as the buildit assembles, rendered as a textured billboard from `STUFF/PARTICLE_PC.DDS`. Matches the engine's GROW particle mode — each sprite spawns ~1/30 size and ramps to full over ~0.5 s, then holds; lifetime comes from `2*rand*life_a + life_b` (≈7 s). Trigger fixed: particles fire at build/activation (one burst per sub-object), not continuously during the bob. (The buildit "building" state machine itself is still not implemented — particles currently fire during the animating/jibber phase as a visual placeholder.)

Reverse engineering (Ghidra, "research/ghidra docs.md")
- "LEGOStarWarsSaga.exe" is open in the Ghidra project "LEG"; findings are cross-referenced against the Android decompilation in "research/saga".
- Map lighting ground truth: the map is prelit/baked (no D3D lights), the "LIGHTMAP_STAGE" selectors live in MS00 material record +0x26E and the "lightmap_set_index"/UV-set fields, and the DISP "LIGHTMAP" display commands (op 0xb0) carry per-draw page textures + "lightmapOffset" — all decoded and rendering in the viewer.
- Ground truth located for the boot/level-load chain: "Game_Init", "Levels_ConfigureList" (LEVELS.TXT -> "LEVELDATA" records, "NEWGAME_LDATA" = the cantina hub), "Menu_NewGameHandler" -> "REQUESTED_LEVEL" -> "Game_LevelStateUpdate" state machine, "Player_ResetContext", "Level_ResetLoadState".
- Named globals ("LDataList", "LEVELCOUNT", "AREAS", "EPISODES", "CHARS_LIST", "NAME_HASH_TABLE", ...), built the 0x130-byte "LEVELDATA" struct, and renamed 21 functions (Nu2 parser API, hash table, char/area/episode config loaders, locator/hub helpers). Full log in "research/ghidra docs.md".
- GIZ format reverse-engineered from "FUN_0055ad70" (outer loader), with type-table lookup ("FUN_00559e20"), "GizObstacle" record reader ("FUN_00557100"), and sub-object reader ("FUN_005aef10"). The "blowup" block format was determined empirically: type definitions (variable) followed by 118-byte instance records with length-prefixed names, Vec3 position, 3× i16 Euler rotation (65536-unit circle), and template metadata.
- Buildit jibber particles reversed from "giz_buildit_spawn_particles" (0x00590690), "particle_spawn" (0x0064d270), and the per-particle update "FUN_0064eb00". Key constants read live from the PE .rdata/.data and the global "g_particle_template" (0x00810210): GROW size mode ramps `(clock/0.5)*0.1` (rate 0.5 s, target 0.1, seed ~1/30), lifetime = `2*rand*life_a + life_b` (`life_a`≈0, `life_b`=7.0) — the sprite grows then holds then dies. Particles fire at build/activation (one burst per sub-object), not during the bob. Full notes in "research/ghidra docs.md" §14.
- Particle RNG bug found via in-game behavior: "rand_f32" took the top 16 bits of a u64 LCG (`(v>>16)/0xFFFF`), yielding values in [0, 4.3e9] instead of [0,1]. This exploded particle size/velocity (screen-filling quads) and mis-positioned the sprites — it was the root cause of the "blinking void" and particles appearing away from their buildit. Fixed to take the top 16 bits correctly. (The blinking-void was also exacerbated by a phantom "has_specular" on lightmapped map materials — fixed to require a bound specular map.)

## 3. What's missing — the path to beating LSW:TCS

None of the game logic exists yet. The viewer can showcase assets, but there
is no playable loop. Ordered roughly by dependency:

Formats still undocumented
- ".PAK" containers — the game reads almost everything from them: "ALLTXT.PAK"
  (all character/world ".TXT" definitions), "SCRIPTS/AI.PAK", level PAKs in
  "LEVELS/". Nothing parses ".PAK" yet.
- Level bundles — "LEVELS/EPISODE_/<level>.PAK" hold a level's models,
  scripts, locators, and object placements. No format documentation exists.
- Save/profile format (needed to persist progress for a full playthrough).
- ".TXT" config is only partially read: we parse "layers_"; the same files
  define weapons, actions, anim bindings, AI, hp, and physics values that a
  real engine must honor.
- Cutscene data ("CUT/", BINK video) — not touched (can play externally).

Scripting / engine behavior
- The game's logic lives in "SCRIPTS/.SCP" and ".PAK" script files — the
  language/VM is not parsed or executed at all. This is the biggest single
  unknown and the critical path to gameplay.
- No object system: triggers, doors, switches, levers, destructible bricks,
  pickups, studs/coins, vehicles, NPCs, boss logic.

Gameplay systems
- Player controller (move/jump/attack/use/force) and camera (chase, triggers).
- Physics and collision.
- AI for enemies and allies.
- Animation state machine: blend_in/blend_out, fpsec, "anim_start/action"
  mapping, upper/lower-body layering, BSA sync. The viewer plays one clip +
  one BSA at a time.
- Audio (music, SFX, "AUDIO/.CFG") — none.
- Menus, character select, hub, front-end, and a save/profile system.

Rendering robustness
- Per-material alpha/transparency and blending modes (only basic opaque
  shading today).
- Culling, distance-based LOD switching, and instancing — the viewer draws the
  whole scene every frame.
- Effects (glow, force lightning, explosions) and shadows.
- Locators are parsed and used by the game bin for spawn/floor resolution;
  gameplay needs them for weapon/helmet/head attachment and throwing points.

## 4. Roadmap

Milestone 1 — Map loading (hub largely done; ".PAK" still open)
- The Mos Eisley cantina hub loads and is walkable in the "game" bin
  ("MAP_PC.GSC"); the remaining item is reversing a ".PAK" level bundle so
  arbitrary levels can be opened the way the real engine opens them.
- Reverse a level bundle from "LEVELS/EPISODE_" (start with a small, early
  one, e.g. an Episode I level or hub).
- Map the ".PAK" container layout: entry table, names, offsets, sizes,
  compression if any. This also unlocks "ALLTXT.PAK" and "SCRIPTS/AI.PAK".
- Reverse the level's internal structure: which entries are GHG models,
  textures, object/instance placements, locators, triggers, and script
  references — and how the level references its assets.
- Add a "pak"/"level" module to the "rustt" library plus a CLI dump tool for
  level bundles.
- Add a "load a level" path to the viewer so a level can be loaded and
  previewed (meshes + placements) the way characters can today.

Milestone 2 — Script engine
- Reverse the ".SCP" script language / VM and its ".PAK" packaging.
- Execute scripted object behaviors (triggers, doors, switches, pickups).

Milestone 3 — Playable vertical slice
- Player controller, gameplay camera, physics + collision, studs/pickups, and
  win/lose conditions — enough to actually play one level start to finish.
- Animation state machine: "anim_start/action" mapping, blend_in/blend_out,
  upper/lower-body layering, BSA sync.
- Locators wired up for weapon/helmet/head attachment.

Milestone 4 — Full game systems
- AI for enemies and allies, vehicles, puzzles, boss fights, destructible
  bricks, audio (music + SFX).
- Cutscene playback hook-up for "CUT/" BINK video.

Milestone 5 — Shell and polish
- Menus, character select, hub, front-end, and a save/profile system so a full
  six-episode playthrough can be completed and persisted.
- Rendering robustness: per-material transparency/blending, effects, shadows,
  culling, distance-based LOD, instancing.

## 5. Generalization — making map systems work on any level

The current GIZ/AI2/map pipeline is validated only on the cantina hub
("MAP"). To load any level, the following must be generalized:

### 5a. Mesh overrides for SOs with cmd_count=0

The cantina hardcodes mesh 982 (a room-geometry part) as the override for
"chair_01", because "chair_01"'s SO record has zero game-model commands
so the parser creates no render_part for it.  Other levels will have
their own SOs with "cmd_count=0" whose meshes live in the room geometry.

What's needed:
- Automatically detect which SO entities have "cmd_count=0" at parse time.
- For each such SO, find its mesh in the room geometry by matching the
  SO's name to a room geometry part's name (both come from the NTBL block).
  If no name match, fall back to positional proximity or a heuristic.
- Build the "mesh_overrides" map automatically instead of hardcoding
  mesh 982 / "chair_01".

### 5b. Obstacle position application

The 42/110 GIZ obstacles matched via MAP.TXT → SO name chains currently
have positions parsed but not applied to render_parts (only blowup
positions are applied).  Doors, contraptions, and other obstacles need
their positions and rotations set.

What's needed:
- Extend "apply_blowup_positions" (or write a parallel function) to handle
  "GizObstacle" records: match each obstacle's sub-object names to SO
  render_parts and set position/rotation/scale.
- Obstacles have richer data than blowups (sub-objects, multiple Vec3s,
  flags, floats) — the transform composition needs to account for
  sub-object offsets.

### 5c. Trigger-based room assignment

AI2 triggers define room AABBs.  The current system assigns rooms to
render_parts by testing their position against trigger volumes, but only
for SO entities.  Room geometry parts get their room assignment from the
GSC DISP command stream.

What's needed:
- Verify that room geometry parts always have correct room assignments
  from the DISP stream (they should, since the game sets this up).
- For SOs and blowup instances, ensure the room-assignment AABB test
  works for all room shapes (not just the cantina's simple rectangular
  rooms — some levels have L-shaped or overlapping rooms).

### 5d. Level-agnostic file discovery

The viewer and game binary currently look for a hardcoded path
("LEVELS/MAP/MAP/MAP_PC.GSC").  Other levels live under
"LEVELS/EPISODE_<n>/<level_name>/" with varying directory structures.

What's needed:
- Parse "LEVELS.TXT" to discover available levels and their paths
  (the "LEVELDATA" struct is partially reverse-engineered).
- Accept a level name or index on the command line instead of a
  hardcoded path.
- Auto-discover sibling ".GIZ", ".AI2", and ".TXT" files relative to
  the ".GSC" (the "find_sibling_giz" heuristic already does this for
  GIZ; extend to AI2 and TXT).

### 5e. Multiple GIZ block types

Only "GizObstacle" and "blowup" blocks are parsed.  The cantina GIZ has
20 block types including "GizBuildit", "GizForce", "GizmoPickup",
"Lever", "Spinner", "Tube", "ZipUp", "GizTurret", "BombGenerator",
"Panel", "HatMachine", "PushBlocks", "Torp Machine", "ShadowEditor",
"Grapple", "Plug", and "Techno".

What's needed:
- Prioritize which additional block types to reverse-engineer based on
  which levels use them (buildits for construction puzzles, levers for
  doors, pickups for studs/bricks, turrets for combat).
- At minimum, "GizBuildit" (construction puzzles) and "GizmoPickup"
  (studs/collectibles) are needed for basic gameplay.

### 5f. PAK file loading

The biggest blocker for arbitrary levels: all assets are packaged in
".PAK" files.  The cantina loads from loose files on disk (PC debug
build), but other levels only exist inside "LEVELS/EPISODE_<n>/<level>.PAK".

What's needed:
- Reverse the ".PAK" container format (entry table, names, offsets,
  sizes, compression).
- Add a virtual filesystem layer that transparently reads from either
  loose files or ".PAK" archives.
- This also unlocks "ALLTXT.PAK" (all character/world definitions) and
  "SCRIPTS/AI.PAK" (AI scripts).

