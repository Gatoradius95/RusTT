struct CameraUniform {
    view_proj: mat4x4<f32>,
    // Per-draw world transform, written by the caller before each scene draw
    // (identity for the grid and static map geometry).
    model: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> u_cam: CameraUniform;

// ---------------- model ----------------

struct ModelVertexInput {
    @location(0) a_pos: vec3<f32>,
    @location(1) a_normal: vec3<f32>,
    @location(2) a_uv: vec2<f32>,
    @location(3) a_weights: vec4<f32>,
    @location(4) a_bones: vec4<u32>,
};

struct ModelVertexOutput {
    @location(0) v_uv: vec2<f32>,
    @location(1) v_normal: vec3<f32>,
    @builtin(position) v_pos: vec4<f32>,
};

@group(1) @binding(0) var u_tex: texture_2d<f32>;
@group(1) @binding(1) var u_tex_sampler: sampler;

struct MaterialUniform {
    base_color: vec4<f32>,
    has_tex: u32,
};
@group(1) @binding(2) var<uniform> u_mat: MaterialUniform;

@group(2) @binding(0) var<storage, read> u_bones: array<mat4x4<f32>>;

// ---------------- morph (blend shapes) ----------------
//
// Per-frame BSA weights drive additive per-vertex deltas on the GPU, like
// BactaTank's uDynamicBuffer. Each morphing mesh gets its own bind group:
//   binding 0: shared storage buffer of every part's deltas, laid out as
//              slot-major blocks of `num_v` vec3s (empty slots are zeros);
//   binding 1: uniform with the current BSA weights + per-part metadata;
//   binding 2: the mesh's index buffer as storage, to map the draw's
//              @builtin(vertex_index) back to the raw vertex id.

const MAX_MORPH_SLOTS: u32 = 64u;

// 64 weights packed as 16 vec4s: the uniform address space requires
// 16-byte array strides, and vec4<f32> gives that while keeping the same
// byte layout as a flat [f32; 64] on the Rust side.
struct MorphUniform {
    weights: array<vec4<f32>, 16>,
    num_v: u32,
    slot_count: u32,
    delta_base: u32,
    enabled: u32,
    _pad: vec4<u32>,
};
@group(3) @binding(0) var<storage, read> u_morph_deltas: array<vec3<f32>>;
@group(3) @binding(1) var<uniform> u_morph: MorphUniform;
@group(3) @binding(2) var<storage, read> u_morph_idx: array<u32>;

@vertex
fn vs_model(
    in: ModelVertexInput,
    @builtin(vertex_index) vi: u32,
) -> ModelVertexOutput {
    var pos = in.a_pos;
    if u_morph.enabled == 1u {
        let vid = u_morph_idx[vi];
        for (var s = 0u; s < u_morph.slot_count; s = s + 1u) {
            let w = u_morph.weights[s / 4u][s % 4u];
            if w != 0.0 {
                pos = pos + u_morph_deltas[u_morph.delta_base + s * u_morph.num_v + vid] * w;
            }
        }
    }
    var skin = u_bones[in.a_bones[0]] * in.a_weights[0];
    skin = skin + u_bones[in.a_bones[1]] * in.a_weights[1];
    skin = skin + u_bones[in.a_bones[2]] * in.a_weights[2];
    skin = skin + u_bones[in.a_bones[3]] * in.a_weights[3];
    let total = in.a_weights[0] + in.a_weights[1] + in.a_weights[2] + in.a_weights[3];
    if total > 0.0 {
        skin = skin * (1.0 / total);
    }
    var out: ModelVertexOutput;
    out.v_pos = u_cam.view_proj * (u_cam.model * (skin * vec4<f32>(pos, 1.0)));
    out.v_uv = in.a_uv;
    let n_mat = mat3x3<f32>(skin[0].xyz, skin[1].xyz, skin[2].xyz);
    out.v_normal = n_mat * in.a_normal;
    return out;
}

@fragment
fn fs_model(in: ModelVertexOutput) -> @location(0) vec4<f32> {
    var color = u_mat.base_color;
    if u_mat.has_tex == 1u {
        color = color * textureSample(u_tex, u_tex_sampler, in.v_uv);
    }
    let n = normalize(in.v_normal);
    let light_dir = normalize(vec3<f32>(0.4, 1.0, 0.3));
    let ndl = max(dot(n, light_dir), 0.0);
    let ambient = 0.42;
    let lit = color.rgb * (ambient + (1.0 - ambient) * ndl);
    return vec4<f32>(lit, color.a);
}

// ---------------- lines ----------------

struct LineVertexInput {
    @location(0) a_pos: vec3<f32>,
    @location(1) a_color: vec4<f32>,
};

struct LineVertexOutput {
    @location(0) v_color: vec4<f32>,
    @builtin(position) v_pos: vec4<f32>,
};

@vertex
fn vs_lines(in: LineVertexInput) -> LineVertexOutput {
    var out: LineVertexOutput;
    out.v_pos = u_cam.view_proj * vec4<f32>(in.a_pos, 1.0);
    out.v_color = in.a_color;
    return out;
}

@fragment
fn fs_lines(in: LineVertexOutput) -> @location(0) vec4<f32> {
    return in.v_color;
}
