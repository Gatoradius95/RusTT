# RusTT — A native reimplementation of the TT Games Nu2 engine

## 1. Overview

LEGO Star Wars, LEGO Indiana Jones, LEGO Batman, LEGO Star Wars II, and
several other LEGO games from the mid-2000s were built by Traveller's Tales on
an internal engine called Nu2. The games shipped as a bare Windows
executable plus a set of loose asset folders — no public SDK, no source, and a
mostly undocumented file format family (GHG models, AN3 skeleton animation,
BSA blend-shape animation, DXT-compressed textures, `.TXT` configuration,
`.PAK` containers, `.SCP` scripts).

This project aims to natively reimplement that engine in Rust, so that
these games can run on modern systems without the original binary. The target
and benchmark is LEGO Star Wars: The Complete Saga: the goal is to reach a
state where the game is playable from start to finish — all six episodes,
levels, puzzles, and bosses beatable.

A real install of LSW:TCS lives in `Lego Star Wars/` and serves as both the
data source and the ground-truth reference; `backup/` holds pristine copies of
the same assets for recovery. `research/` contains community format
documentation (Blender add-ons, EasyAN3, BactaTank, BrickBench) used to reverse
the formats.

The project is currently an asset-format toolkit + interactive viewer, not
yet an engine. All the hard file-format work that makes the engine possible is
in place; the gameplay layer is not started.

## 2. What works today

Library (`src/lib.rs` + modules)
- `ghg` — full GHG model parser: materials, DXT textures, parts, render items
  with their LOD/layer indices, the skeleton, and per-slot blend-shape
  (morph) delta data.
- `an3` — AN3 skeletal animation parser and playback math (ANI4/ANI6/ANI8
  versions), verified byte-for-byte against real files.
- `bsa` — BSA blend-shape weight animation parser and channel evaluation
  (hermite keyframes), verified against `backup/CHARS/`.
- `dxt` — DXT texture decompression.
- `glb` — glTF/GLB export (mesh + skeleton + morph data + textures).

Viewer (`viewer` binary) — an interactive wgpu-based renderer:
- Real-time rendering of skinned meshes (GPU skinning from AN3 playback).
- GPU morph blend shapes driven live by a BSA clip (facial animation).
- Orbit camera + WASD/QE fly, scroll zoom, wireframe overlay, grid toggle,
  "apply bind pose" toggle.
- UI panels: Scene, Materials, Bones, Textures, Animation.
- Quality/LOD layer selection that matches the game: reads the sibling
  `.TXT` (`layers_special/high/medium/low/dead`) and draws only the selected
  set, so a minifig renders as the game renders it (one body, no duplicate
  LOD limbs). Handles platform-suffixed names (`BOBAFETT_PC.GHG` →
  `BOBAFETT.TXT`) and `txt_file="base"` inheritance for variants.
- AN3 playback with speed control, loop, and frame scrubbing.

CLI tools
- `rustt` (convert): GHG → GLB.
- `dump`, `layerinfo`, `bsa_dump`: diagnostics for models, LOD layer layout,
  and blend-shape channels.
- `pose_render`: offscreen skeleton-pose renderer used to settle bone/leg
  orientation questions.

Tests (all green): AN3 smoke + loop-continuity, BSA evaluation against
real files, a naga/WGSL stride regression test for the morph buffer, and
viewer TXT-discovery tests (suffix probing + `txt_file` inheritance).

## 3. What's missing — the path to beating LSW:TCS

None of the game logic exists yet. The viewer can showcase assets, but there
is no playable loop. Ordered roughly by dependency:

Formats still undocumented
- `.PAK` containers — the game reads almost everything from them: `ALLTXT.PAK`
  (all character/world `.TXT` definitions), `SCRIPTS/AI.PAK`, level PAKs in
  `LEVELS/`. Nothing parses `.PAK` yet.
- Level bundles — `LEVELS/EPISODE_/<level>.PAK` hold a level's models,
  scripts, locators, and object placements. No format documentation exists.
- Save/profile format (needed to persist progress for a full playthrough).
- `.TXT` config is only partially read: we parse `layers_`; the same files
  define weapons, actions, anim bindings, AI, hp, and physics values that a
  real engine must honor.
- Cutscene data (`CUT/`, BINK video) — not touched (can play externally).

Scripting / engine behavior
- The game's logic lives in `SCRIPTS/.SCP` and `.PAK` script files — the
  language/VM is not parsed or executed at all. This is the biggest single
  unknown and the critical path to gameplay.
- No object system: triggers, doors, switches, levers, destructible bricks,
  pickups, studs/coins, vehicles, NPCs, boss logic.

Gameplay systems
- Player controller (move/jump/attack/use/force) and camera (chase, triggers).
- Physics and collision.
- AI for enemies and allies.
- Animation state machine: blend_in/blend_out, fpsec, `anim_start/action`
  mapping, upper/lower-body layering, BSA sync. The viewer plays one clip +
  one BSA at a time.
- Audio (music, SFX, `AUDIO/.CFG`) — none.
- Menus, character select, hub, front-end, and a save/profile system.

Rendering robustness
- Per-material alpha/transparency and blending modes (only basic opaque
  shading today).
- Culling, distance-based LOD switching, and instancing — the viewer draws the
  whole scene every frame.
- Effects (glow, force lightning, explosions) and shadows.
- Locators are parsed but unused; gameplay needs them for weapon/helmet/head
  attachment and throwing points.

## 4. Roadmap

Milestone 1 — Map loading (imminent next step)
- Reverse a level bundle from `LEVELS/EPISODE_` (start with a small, early
  one, e.g. an Episode I level or hub).
- Map the `.PAK` container layout: entry table, names, offsets, sizes,
  compression if any. This also unlocks `ALLTXT.PAK` and `SCRIPTS/AI.PAK`.
- Reverse the level's internal structure: which entries are GHG models,
  textures, object/instance placements, locators, triggers, and script
  references — and how the level references its assets.
- Add a `pak`/`level` module to the `rustt` library plus a CLI dump tool for
  level bundles.
- Add a "load a level" path to the viewer so a level can be loaded and
  previewed (meshes + placements) the way characters can today.

Milestone 2 — Script engine
- Reverse the `.SCP` script language / VM and its `.PAK` packaging.
- Execute scripted object behaviors (triggers, doors, switches, pickups).

Milestone 3 — Playable vertical slice
- Player controller, gameplay camera, physics + collision, studs/pickups, and
  win/lose conditions — enough to actually play one level start to finish.
- Animation state machine: `anim_start/action` mapping, blend_in/blend_out,
  upper/lower-body layering, BSA sync.
- Locators wired up for weapon/helmet/head attachment.

Milestone 4 — Full game systems
- AI for enemies and allies, vehicles, puzzles, boss fights, destructible
  bricks, audio (music + SFX).
- Cutscene playback hook-up for `CUT/` BINK video.

Milestone 5 — Shell and polish
- Menus, character select, hub, front-end, and a save/profile system so a full
  six-episode playthrough can be completed and persisted.
- Rendering robustness: per-material transparency/blending, effects, shadows,
  culling, distance-based LOD, instancing.

