//! Validates the viewer's WGSL shader with naga (the same frontend wgpu uses),
//! so shader syntax/type errors surface in `cargo test` instead of at runtime.
use std::path::Path;

#[test]
fn viewer_wgsl_parses() {
    let src = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/bin/viewer/shaders.wgsl"),
    )
    .expect("read shaders.wgsl");
    naga::front::wgsl::parse_str(&src).expect("shaders.wgsl must be valid WGSL");
}
