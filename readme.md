# RusTT

A native reimplementation of the TT Games Nu2 engine in Rust, targeting LEGO Star Wars: The Complete Saga.

The mid-2000s LEGO games (Star Wars, Indiana Jones, Batman) were built by Traveller's Tales on an internal engine called Nu2 with proprietary file formats (GHG models, AN3 animations, BSA blend-shapes, GSC maps, SCP scripts). This project aims to rebuild that engine from scratch so the games can run on modern systems and modders get a proper base to work from.

## Status

The project is currently an asset-format toolkit + interactive viewer + gameplay test bed, plus a Ghidra reverse-engineering effort against the original PC executable.

**What works today:**
- Full GHG model parsing (materials, DXT textures, parts, skeleton, blend-shape deltas)
- AN3 skeletal animation playback (WIP)
- BSA blend-shape weight animation (WIP)
- GSC map parsing with room-based culling, lightmaps, and GIZ object placement
- Interactive wgpu viewer with GPU skinning, morph blend-shapes, orbit camera, material/texture panels
- Walkable third-person cantina hub (Mos Eisley) with chase camera + collision
- glTF/GLB export
- 40+ tests covering AN3, BSA, GIZ, WGSL shader validation, and more

**What's missing:** PAK container parsing, the SCP scripting language, player controller, physics, AI, audio, menus, cutscenes. See [project.md](project.md) for the full breakdown and roadmap.

## Building

Requires Rust 1.85+ (edition 2024).

```
cargo build --release
```

## Binaries

| Binary | Description |
|--------|-------------|
| `viewer` | Interactive wgpu viewer for GHG models and GSC maps |
| `game` | Gameplay test bed (walkable cantina hub) |
| `rustt` | GHG to glTF/GLB converter |
| `bsa_dump` | BSA blend-shape diagnostic dump |
| `mapdump` | GSC map diagnostic dump |
| `ai2dump` | AI2 trigger/locator dump |

## Project structure

```
src/
  lib.rs          Core library
  ghg.rs          GHG model parser
  an3.rs          AN3 skeletal animation
  bsa.rs          BSA blend-shape animation
  dxt.rs          DXT texture decompression
  glb.rs          glTF/GLB export
  map.rs          GSC map parser
  mapmesh.rs      Map mesh builder
  map_txt.rs      MAP.TXT config parser
  ai2.rs          AI2 trigger/locator parser
  giz.rs          GIZ object placement parser
  bin/
    viewer/       Interactive wgpu viewer
    game/         Gameplay test bed
    convert.rs    GHG to GLB converter
    ...
tests/            40+ integration and diagnostic tests
```

## License

Not yet determined. The original game assets and engine are property of TT Games / Warner Bros. This project does not include any game data.
