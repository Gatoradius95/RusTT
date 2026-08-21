use std::path::Path;

use anyhow::{Context, Result};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let input = args
        .next()
        .context("usage: rustt <input.ghg> [output.glb]")?;
    let output = args
        .next()
        .map(String::from)
        .unwrap_or_else(|| {
            Path::new(&input)
                .with_extension("glb")
                .to_string_lossy()
                .into_owned()
        });

    if Path::new(&input) == Path::new(&output) {
        anyhow::bail!("output path equals input path; refusing to overwrite source");
    }

    let data = std::fs::read(&input)
        .with_context(|| format!("reading {}", input))?;
    let parsed = rustt::ghg::parse(&data).with_context(|| format!("parsing {}", input))?;

    let meshes = rustt::glb::build_meshes(&parsed);
    let glb = rustt::glb::build_glb(&parsed, &meshes)?;
    std::fs::write(&output, &glb).with_context(|| format!("writing {}", output))?;

    let tris: usize = meshes.iter().map(|m| m.idx.len() / 3).sum();
    let verts: usize = meshes.iter().map(|m| m.pos.len()).sum();
    println!(
        "{} -> {}\n  parts={} triangles={} vertices={} materials={} textures={} bones={}",
        Path::new(&input).file_name().unwrap().to_string_lossy(),
        output,
        parsed.parts.len(),
        tris,
        verts,
        parsed.materials.len(),
        parsed.textures.len(),
        parsed.bones.len()
    );
    Ok(())
}
