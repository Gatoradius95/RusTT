#[test]
fn viewer_shader_validates() {
    let src =
        std::fs::read_to_string("src/bin/viewer/shaders.wgsl").expect("read shaders.wgsl");
    let module = naga::front::wgsl::parse_str(&src).expect("wgsl parse failed");
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .expect("wgsl validation failed");
}

#[test]
fn morph_delta_stride_is_sixteen() {
    // The Rust side packs shape-key deltas into a storage buffer that the
    // shader reads as `array<vec3<f32>>`. Storage arrays stride each element to
    // its alignment, so the GPU-side stride must be 16 bytes (the CPU buffer
    // pads each delta to vec4 to match). If this ever regresses, shape-key
    // morphs silently read the wrong vertices.
    let src =
        std::fs::read_to_string("src/bin/viewer/shaders.wgsl").expect("read shaders.wgsl");
    let module = naga::front::wgsl::parse_str(&src).expect("wgsl parse failed");
    let global = module
        .global_variables
        .iter()
        .find(|(_, g)| g.name.as_deref() == Some("u_morph_deltas"))
        .expect("u_morph_deltas global")
        .1;
    let ty = &module.types[global.ty];
    match &ty.inner {
        naga::TypeInner::Array { stride, .. } => {
            assert_eq!(
                *stride, 16,
                "storage array<vec3<f32>> must stride by 16 bytes"
            );
        }
        other => panic!("u_morph_deltas has unexpected type: {other:?}"),
    }
}
