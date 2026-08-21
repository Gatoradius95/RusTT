struct CameraUniform {
    view_proj: mat4x4<f32>,
    // Per-draw world transform, written by the caller before each scene draw
    // (identity for the grid and static map geometry).
    model: mat4x4<f32>,
    // World-space camera position plus fog state. Lighting is per-mesh:
    // each scene binds its own LightSet as group 3 binding 3 and the uber
    // shader's lighting block reads from there, not from the camera.
    cam_pos: vec4<f32>,
    // Fog state (FOG_STAGE): x=start, y=end, z=exponential density,
    // w=fog mode (0 off, 1 linear, 2 exp, 3 exp2).
    fog_color: vec4<f32>,
    fog_params: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u_cam: CameraUniform;

// Backbuffer texture for refraction (opaque pass renders here first).
@group(0) @binding(1) var u_backbuffer: texture_2d<f32>;
@group(0) @binding(2) var u_backbuf_sampler: sampler;

// ---------------- model ----------------

struct ModelVertexInput {
    @location(0) a_pos: vec3<f32>,
    @location(1) a_normal: vec3<f32>,
    @location(2) a_uv: vec2<f32>,
    @location(3) a_weights: vec4<f32>,
    @location(4) a_bones: vec4<u32>,
    // Baked per-vertex light (RGBA8). Map meshes carry it; characters upload
    // opaque white so they are unaffected.
    @location(5) a_color: vec4<f32>,
    // Lightmap UVs (raw file values: u in [0..1], v in [-1..0]; u <= 0
    // selects the vertex-lit fallback). Zero for characters and for map
    // materials without a lightmap stage.
    @location(6) a_lmuv: vec2<f32>,
    @location(7) a_tangent: vec4<f32>,
};

struct ModelVertexOutput {
    @location(0) v_uv: vec2<f32>,
    @location(1) v_normal: vec3<f32>,
    @location(2) v_world_pos: vec3<f32>,
    @location(3) v_color: vec4<f32>,
    @location(4) v_lmuv: vec2<f32>,
    @location(5) @interpolate(flat) v_mesh_type: u32,
    @location(6) v_tangent: vec4<f32>,
    @builtin(position) v_pos: vec4<f32>,
};

@group(1) @binding(0) var u_tex: texture_2d<f32>;
@group(1) @binding(1) var u_tex_sampler: sampler;

// Model-level highlight/glint texture (the per-character lens-flare sprite),
// sampled with UV0 to modulate the Phong specular term.
@group(1) @binding(3) var u_highlight: texture_2d<f32>;

// Lightmap textures (LM0/LM1/LM2), the three textures immediately after the
// material's `lightmap_set_index`. All bound to the same filtered sampler;
// they sample with the LIGHTMAP_UVSET. White when the material has no
// lightmap stage (the game's lightmaps are exact-doubles of atlas regions).
@group(1) @binding(4) var u_lm0: texture_2d<f32>;
@group(1) @binding(5) var u_lm1: texture_2d<f32>;
@group(1) @binding(6) var u_lm2: texture_2d<f32>;
// Normal map (binding 7), specular map (binding 8), and cubemap (binding 9).
@group(1) @binding(7) var u_norm: texture_2d<f32>;
@group(1) @binding(8) var u_spec: texture_2d<f32>;
@group(1) @binding(9) var u_cube: texture_2d<f32>;
@group(1) @binding(10) var u_cube_sampler: sampler;

// Refraction types (packed into blend_mode upper 16 bits).
const REFRACTION_NONE: u32 = 0u;
const REFRACTION_GLASS: u32 = 3u;

struct MaterialUniform {
    base_color: vec4<f32>,
    has_tex: u32,
    // Lighting stage derived from the MS00 `shaderDefines` bits (record
    // +0x26C): 0 = DISABLE, 1 = LAMBERT, 6 = PHONG, ...
    lighting_stage: u32,
    // 1 when the material's defines pick PRELIGHT_FX (`defines & 0x1000`) —
    // the baked per-vertex light multiply. DISABLE materials (PRELIGHT_FX
    // without the 0x80000000 live bit) still consume the baked vertex light.
    prelit: u32,
    // 1 when u_highlight should modulate the Phong specular.
    has_highlight: u32,
    // Uber Shader 2.0 per-material params (MS00 record +0x12C/0x130/0x144/0x148).
    // x = kCosPower, y = kSpecular, z = kFresnel, w = kFresnelPower.
    specular_params: vec4<f32>,
    ambient_color: vec4<f32>,
    incandescent_glow: vec4<f32>,
    // Lightmap stage (0 = DISABLE, 1 = LIGHTMAP_SMOOTH, 2 =
    // LIGHTMAP_DIRECTIONAL) — the game's LIGHTMAP_STAGE define.
    lm_stage: u32,
    // 1 when a lightmap texture set (LM0..2) is bound; 0 keeps the baked
    // vertex light as the only prelit diffuse.
    has_lm: u32,
    // Material blend mode (low nibble of alpha_type MS00 +0x40):
    // 0 = NONE/opaque, 1 = TRANSPARENT (srcA/1-srcA), 2 = ADDITIVE.
    // Opaque materials must output alpha=1.0 so the pipeline's alpha blend
    // is a no-op (otherwise vcol alpha ~0.996 bleeds 0.4% of background
    // at part-boundary seams).
    blend_mode: u32,
    // Alpha-test threshold: when > 0, fragments with alpha <= this value are
    // discarded (D3D9 ALPHATESTENABLE + ALPHAREF). The original game uses
    // ALPHAREF = 0x10 (≈0.0627) for cutout materials (face decals, foliage).
    // 0.0 disables the test.
    alpha_cutoff: f32,
    // `lm_bits` = the game's `lightmapOffset` as float bits:
    // `lightmapCoord = rawUV * lm_bits.zw + lm_bits.xy`.
    lm_bits: vec4<u32>,
    // 1 when a normal map is bound at binding 7.
    has_normal: u32,
    // 1 when a specular map is bound at binding 8.
    has_specular: u32,
    // 1 when a cubemap is bound at binding 9 (envmap type == Cube).
    has_cubemap: u32,
    // Cubemap reflection strength (material +0x12C, "reflectionPower").
    reflection_power: f32,
};

// LIGHTING_STAGE values from the uber shader source (uberShader2.glsl).
const LIGHTING_DISABLE: u32 = 0u;
const LIGHTING_LAMBERT: u32 = 1u;
const LIGHTING_PHONG: u32 = 6u;
@group(1) @binding(2) var<uniform> u_mat: MaterialUniform;

@group(2) @binding(0) var<storage, read> u_bones: array<mat4x4<f32>>;

// Per-mesh transform buffer (group 4).  Identity for static geometry; buildit
// sub-objects get per-frame animated transforms written by the game loop.
// Indexed by gl_InstanceIndex (the mesh ID passed via draw_indexed).
@group(4) @binding(0) var<storage, read> u_mesh_xforms: array<mat4x4<f32>>;

// Per-mesh SO/room type flag (0 = room geometry, 1 = SO entity).
// Used by the 'O' debug coloring mode.
@group(4) @binding(1) var<storage, read> u_mesh_type: array<u32>;

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
// byte layout as a flat [f32; 64] on the Rust side. The remaining metadata
// is two vec4<u32> (also 16-aligned): meta0 = [num_v, slot_count,
// delta_base, enabled], meta1 = [channel_base, 0, 0, 0].
struct MorphUniform {
    weights: array<vec4<f32>, 16>,
    meta0: vec4<u32>,
    meta1: vec4<u32>,
};
@group(3) @binding(0) var<storage, read> u_morph_deltas: array<vec3<f32>>;
@group(3) @binding(1) var<uniform> u_morph: MorphUniform;
@group(3) @binding(2) var<storage, read> u_morph_idx: array<u32>;

// Per-mesh lighting, the Uber Shader 2.0 lighting block (7 vec4s). Map
// meshes get their own lights baked from the RTL list at their position;
// characters share the default rig. light_pos[i].w is the intensity factor
// (the original's lightPositionX.w).
struct LightSet {
    scene_ambient: vec4<f32>,
    light_color: array<vec4<f32>, 3>,
    light_pos: array<vec4<f32>, 3>,
};
@group(3) @binding(3) var<uniform> u_lights: LightSet;

// ---------------- cubemap helper ----------------

// Maps a world-space direction to UVs on a 4x3 horizontal-cross cubemap
// layout (matching BactaTank Classic shdStandard.fsh cubeToCrossUV).
//   row 0:            [+Y]
//   row 1: [-X][+Z][+X][-Z]
//   row 2:            [-Y]
fn cubeToCrossUV(dir: vec3<f32>) -> vec2<f32> {
    let adir = abs(dir);
    var sc: f32;
    var tc: f32;
    var face: i32;
    var ma: f32;

    if (adir.x >= adir.y && adir.x >= adir.z) {
        ma = adir.x;
        if (dir.x > 0.0) { sc = -dir.z; tc = -dir.y; face = 0; } // +X
        else              { sc =  dir.z; tc = -dir.y; face = 1; } // -X
    } else if (adir.y >= adir.x && adir.y >= adir.z) {
        ma = adir.y;
        if (dir.y > 0.0) { sc =  dir.x; tc =  dir.z; face = 2; } // +Y
        else              { sc =  dir.x; tc = -dir.z; face = 3; } // -Y
    } else {
        ma = adir.z;
        if (dir.z > 0.0) { sc =  dir.x; tc = -dir.y; face = 4; } // +Z
        else              { sc = -dir.x; tc = -dir.y; face = 5; } // -Z
    }

    var uv = vec2<f32>(0.5 * (sc / ma + 1.0), 0.5 * (tc / ma + 1.0));

    // Seam fix: push UVs inward by half a texel to avoid face-edge bleeding.
    let fw = 1.0 / 4.0;
    let fh = 1.0 / 3.0;
    let border = 4.0 / 128.0;
    uv = clamp(uv, vec2<f32>(border), vec2<f32>(1.0 - border));

    var offset: vec2<f32>;
    if (face == 0)      { offset = vec2<f32>(2.0 * fw, 1.0 * fh); }
    else if (face == 1) { offset = vec2<f32>(0.0 * fw, 1.0 * fh); }
    else if (face == 2) { offset = vec2<f32>(1.0 * fw, 0.0 * fh); }
    else if (face == 3) { offset = vec2<f32>(1.0 * fw, 2.0 * fh); }
    else if (face == 4) { offset = vec2<f32>(1.0 * fw, 1.0 * fh); }
    else                { offset = vec2<f32>(3.0 * fw, 1.0 * fh); }

    return offset + uv * vec2<f32>(fw, fh);
}

@vertex
fn vs_model(
    in: ModelVertexInput,
    @builtin(vertex_index) vi: u32,
    @builtin(instance_index) ii: u32,
) -> ModelVertexOutput {
    var pos = in.a_pos;
    if u_morph.meta0[3] == 1u {
        let vid = u_morph_idx[vi];
        let cbase = min(u_morph.meta1[0], 64u);
        for (var s = 0u; s < u_morph.meta0[1]; s = s + 1u) {
            // Shape-key channels number every morphing part's slots in part
            // order, so this part's slot `s` is global channel
            // `channel_base + s`. Clamp past the end (a BSA with fewer
            // channels than the model has slots) to weight 0.
            let c = cbase + s;
            if c < 64u {
                let w = u_morph.weights[c / 4u][c % 4u];
                if w != 0.0 {
                    pos = pos + u_morph_deltas[u_morph.meta0[2] + s * u_morph.meta0[0] + vid] * w;
                }
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
    let mx = u_mesh_xforms[ii];
    let world_pos = u_cam.model * mx * (skin * vec4<f32>(pos, 1.0));
    out.v_pos = u_cam.view_proj * world_pos;
    out.v_uv = in.a_uv;
    // Skin the normal in model space, then rotate into world space for the
    // per-fragment lighting (lights are world-space directions).
    let n_mat = mat3x3<f32>(skin[0].xyz, skin[1].xyz, skin[2].xyz);
    let n_mat2 = mat3x3<f32>(mx[0].xyz, mx[1].xyz, mx[2].xyz);
    let world_n = mat3x3<f32>(u_cam.model[0].xyz, u_cam.model[1].xyz, u_cam.model[2].xyz);
    out.v_normal = world_n * n_mat2 * (n_mat * in.a_normal);
    out.v_world_pos = world_pos.xyz;
    out.v_color = in.a_color;
    out.v_lmuv = in.a_lmuv;
    out.v_mesh_type = u_mesh_type[ii];
    // Skin and rotate tangent into world space (same transform as the normal).
    out.v_tangent = vec4<f32>(world_n * n_mat2 * (n_mat * in.a_tangent.xyz), in.a_tangent.w);
    return out;
}

@fragment
fn fs_model(in: ModelVertexOutput) -> @location(0) vec4<f32> {
    var color = u_mat.base_color;
    // Engine getColor(): every PRELIGHT_FX surface takes its opacity from the
    // baked vertex-color alpha (baseColor = vec4(1,1,1, fs_layer0_color.a) for
    // LIGHTMAP_STAGE==0). FUN_00749620 initially sets COLOR_FACTOR to "1.0",
    // but FUN_0074a970 overrides it to "2.0" at runtime (confirmed via
    // vtable dispatch trace). The alpha doubling comes from kTint (×2).
    if u_mat.prelit == 1u {
        // Alpha is boosted ×2 (kTint.a): vcol byte 127 => opaque; the tank
        // glass (0x4c) => ~60%.
        color.a = min(in.v_color.a * 2.0, 1.0);
    }
    if u_mat.has_tex == 1u {
        color = color * textureSample(u_tex, u_tex_sampler, in.v_uv);
    }
    // PS 2.0+: func_fs_computeLayer0_color() = varying_colorSet0 * COLOR_FACTOR.
    // FUN_0074a970 overrides COLOR_FACTOR to "2.0" at runtime (confirmed via
    // vtable dispatch trace). The uber shader multiplies vertexColor ×
    // COLOR_FACTOR into the surface color (multiLayerBlendingStage line 578)
    // for EVERY surface that carries a colorSet0 attribute — prelit or not.
    // Map meshes carry baked per-vertex light (0..127 scale), so the ×2.0
    // maps bake-fill 127 → ~0.996 (no-op) while genuinely dark vertex colors
    // (0, 49, 82, 95) darken to match the game. Character meshes have no file
    // color and are filled with the 127 bake-fill (glb.rs), keeping them at
    // ~1.0 — the game's `fs_layer0_color = 1.0` for its colorless LAMBERT
    // characters. Without this multiply, dark-vertex surfaces render at full
    // brightness (flat / washed out) instead of the game's shaded look.
    let vcol = in.v_color.rgb;
    var baked = vcol;
    color = vec4<f32>(color.rgb * vcol * 2.0, color.a);
    let n = normalize(in.v_normal);
    // Normal mapping: compute TBN from screen-space derivatives when a normal
    // map is bound.  The normal map tangent-space Z is remapped from [0,1] to
    // [-1,1] and the result replaces the interpolated vertex normal.
    var shading_n = n;
    let dbg_flags = u32(u_cam.cam_pos.w + 0.5);
    if u_mat.has_normal == 1u && (dbg_flags & 16u) == 0u {
        // TBN from per-vertex tangent (BactaTank shdStandard.vsh lines 85-89):
        // T = tangent.xyz, B = cross(N, T) * T.w, N = vertex normal.
        let T = normalize(in.v_tangent.xyz);
        let B = normalize(cross(n, T) * in.v_tangent.w);
        let TBN = mat3x3<f32>(T, B, n);
        let n_sample = textureSample(u_norm, u_tex_sampler, in.v_uv).xyz;
        shading_n = normalize(TBN * (n_sample * 2.0 - 1.0));
    }
    let view_dir = normalize(u_cam.cam_pos.xyz - in.v_world_pos);

    // LIGHTMAP_STAGE (uberShader2.glsl:816-824): the lightmap is sampled
    // for ALL LIGHTMAP_SMOOTH surfaces, not just prelit — the sample happens
    // unconditionally before any PRELIGHT_FX check. For prelit surfaces the
    // lightmap IS the diffuse lighting; for non-prelit surfaces it forms
    // the base and live lights accumulate on top (no ambient — the lightmap
    // replaces it).
    //
    // The lightmaps live in the atlas's LIGHTMAP_UVSET; the file's top-down
    // v stays in [-1..0], flipped to bottom-up with fract(). v_lmuv.x <= 0
    // selects the vertex-lit fallback (hlsl:825): baked vertex light only.
    var lm_diffuse = vec3<f32>(1.0);
    var has_lm_sample = false;
    if u_mat.lm_stage != 0u && u_mat.has_lm == 1u && in.v_lmuv.x > 0.0 {
        // The lightmapOffset transform the shipped vertex shader applies:
        // `lightmapCoord = vs_uvN * lightmapOffset.zw + lightmapOffset.xy`
        // (wiiobject.vert). raw v is in [-1..0]; fract() folds it into the
        // texture's [0..1] range like the D3D9 wrap sampler did.
        let lm_off = bitcast<vec4<f32>>(u_mat.lm_bits);
        let mapped = in.v_lmuv * lm_off.zw + lm_off.xy;
        let luv = vec2<f32>(mapped.x, fract(mapped.y));
        // The shipped engine fragment (wiiobject.frag getLightColor) treats
        // LIGHTMAP_STAGE 1 and 2 identically: diffuseLight = lightmap1 sample
        // only. The 3-way directional blend exists in the source but is dead
        // (the block is compiled out), and the lightmap bindings expose four
        // textures where lightmap2..4 feed that unused path.
        lm_diffuse = textureSample(u_lm0, u_tex_sampler, luv).rgb;
        has_lm_sample = true;
    }

    // Uber Shader 2.0 lighting (uberShader2.glsl lightingStage + Phong
    // specular):
    //   Prelit: diffuseLight = lightmap (or 1.0 without). No live lights.
    //   Non-prelit + lightmap: diffuseLight = lightmap + live_lights.
    //   Non-prelit, no lightmap: diffuseLight = live_lights + ambient.
    //   Non-prelit, no lightmap, LIGHTING_DISABLE: diffuseLight = 1.0.
    var diffuse = vec3<f32>(0.0);
    if u_mat.prelit == 1u {
        // Prelit shading: lightmap is the sole diffuse source (or 1.0 —
        // the baked vertex light already folded into `color`).
        diffuse = lm_diffuse;
    } else if has_lm_sample {
        // Non-prelit with lightmap: lightmap base + live lights on top,
        // no ambient (uberShader2.glsl:816-896).
        diffuse = lm_diffuse;
        if u_mat.lighting_stage != LIGHTING_DISABLE {
            for (var i = 0u; i < 3u; i++) {
                let l = normalize(u_lights.light_pos[i].xyz);
                let ndl = max(dot(shading_n, l), 0.0);
                diffuse += ndl * u_lights.light_pos[i].w * u_lights.light_color[i].rgb;
            }
            diffuse = mix(diffuse, vec3<f32>(1.0), u_mat.incandescent_glow.rgb);
        }
    } else if u_mat.lighting_stage == LIGHTING_DISABLE {
        // Non-prelit, no lightmap, no lighting: unlit.
        diffuse = vec3<f32>(1.0);
    } else {
        // Non-prelit, no lightmap: live lights + ambient.
        for (var i = 0u; i < 3u; i++) {
            let l = normalize(u_lights.light_pos[i].xyz);
            let ndl = max(dot(shading_n, l), 0.0);
            diffuse += ndl * u_lights.light_pos[i].w * u_lights.light_color[i].rgb;
        }
        diffuse += u_mat.ambient_color.rgb + u_lights.scene_ambient.rgb;
        diffuse = mix(diffuse, vec3<f32>(1.0), u_mat.incandescent_glow.rgb);
    }
    var specular = vec3<f32>(0.0);
    if u_mat.lighting_stage == LIGHTING_PHONG {
        for (var i = 0u; i < 3u; i++) {
            let l = normalize(u_lights.light_pos[i].xyz);
            // Phong: R = reflect(-L, N), view vector dotted against it.
            let r = reflect(-l, shading_n);
            specular += u_lights.light_color[i].rgb
                * pow(max(dot(view_dir, r), 0.0), u_mat.specular_params.x)
                * u_lights.light_color[i].w;
        }
        specular *= u_mat.specular_params.y;
        // Specular map: modulate specular intensity per-texel.
        if u_mat.has_specular == 1u {
            let spec_sample = textureSample(u_spec, u_tex_sampler, in.v_uv).r;
            specular *= spec_sample;
        }
        if u_mat.prelit == 1u {
            // PRELIGHT_FX_LIVE_SPECULAR: specular darkened by the active
            // diffuse source — fs_prelitSpecularStage returns
            // specularPhong * fs_layer0_color. For lightmapped surfaces
            // fs_layer0_color is the lightmap sample; for non-lightmapped
            // it is the vertex color (baked).
            specular *= max(baked, lm_diffuse);
        }
        // Model-level highlight/glint texture (a lens-flare sprite, mostly
        // empty with one bright ball). Modulating the specular by it raw
        // both killed the sheen (the flare's background is near-black) and
        // stamped the bright ball over parts whose UVs land on it, so use
        // it as a gentle 0..0.3 boost on top of the Phong specular instead.
        if u_mat.has_highlight == 1u {
            let glint = textureSample(u_highlight, u_tex_sampler, in.v_uv).r;
            specular *= 0.7 + 0.3 * glint;
        }
    }

    // Cubemap reflection: sample the 4x3 cross-layout cubemap texture using
    // the reflected view vector. Matches BactaTank Classic shdStandard.fsh
    // lines 556-562. Uses nearest-neighbor sampler to avoid face-edge bleeding.
    // Debug toggle (key '0'): cam_pos.w bit 3 forces cubemap off.
    var cubemap = vec3<f32>(0.0);
    if u_mat.has_cubemap == 1u && (dbg_flags & 8u) == 0u {
        let I = normalize(u_cam.cam_pos.xyz - in.v_world_pos);
        let refl_dir = reflect(-I, shading_n);
        cubemap = textureSample(u_cube, u_cube_sampler, cubeToCrossUV(refl_dir)).rgb
                  * u_mat.reflection_power;
    }

    var lit = vec4<f32>(color.rgb * diffuse + specular + cubemap, color.a);

    // Alpha test (D3D9 ALPHATESTENABLE + ALPHAREF): discard fragments with
    // alpha at or below the cutoff. The original game's baseline is
    // ALPHAREF=0x10 (GREATER func), so pixels with alpha <= 0.0627 die.
    // This handles cutout materials (face decals, foliage) where the texture
    // has a hard alpha edge — the transparent background must be killed, not
    // blended. The test runs before the blend_mode alpha override so it sees
    // the raw texture/vertex alpha.
    if u_mat.alpha_cutoff > 0.0 && lit.a <= u_mat.alpha_cutoff {
        discard;
    }

    // Opaque materials (blend_mode == 0): force alpha to 1.0 so the
    // pipeline's ALPHA_BLENDING is a no-op.  Without this, vcol alpha
    // ~0.996 bleeds ~0.4% of the framebuffer background at part-boundary
    // edges, producing visible seam lines.
    if u_mat.blend_mode == 0u {
        lit.a = 1.0;
    }

    // Decode refraction type from blend_mode upper 16 bits.
    let refraction_type = (u_mat.blend_mode >> 16u) & 0xFFFFu;
    if refraction_type == REFRACTION_GLASS {
        // Glass refraction: sample backbuffer, lerp with surface color.
        // This gives transparent tinted glass that refracts the background.
        let bb_uv = in.v_pos.xy / vec2<f32>(textureDimensions(u_backbuffer));
        let bb_color = textureSample(u_backbuffer, u_backbuf_sampler, bb_uv);
        // Tint the background with the glass surface color.
        // Output alpha is the glass opacity so ALPHA_BLENDING composites it
        // over the opaque scene already loaded in the swapchain.
        let glass_alpha = clamp(lit.a * 3.0, 0.3, 0.85);
        lit = vec4<f32>(
            mix(bb_color.rgb, bb_color.rgb * lit.rgb, glass_alpha),
            glass_alpha
        );
    }

    // FORCE_OPAQUE mode (cam_pos.w bit0): show blend_mode debug colors
    // to verify the uniform reaches the shader correctly.
    let _mode = u32(u_cam.cam_pos.w + 0.5);
    let _force_opaque = (_mode & 1u) != 0u;
    let _color_correct = (_mode & 2u) != 0u;
    if _force_opaque {
        lit.a = 1.0;
        if u_mat.blend_mode == 0u {
            lit = vec4(0.0, 1.0, 0.0, 1.0); // green = opaque path
        } else {
            lit = vec4(1.0, 0.0, 0.0, 1.0); // red = blended path
        }
    }

    // Fog (FOG_STAGE), disabled when fog_params.w == 0.
    let fog_mode = u_cam.fog_params.w;
    if fog_mode == 1.0 {
        let dist = distance(u_cam.cam_pos.xyz, in.v_world_pos);
        let f = clamp(
            (dist - u_cam.fog_params.x) / max(u_cam.fog_params.y - u_cam.fog_params.x, 0.0001),
            0.0, 1.0);
        lit = vec4<f32>(mix(lit.rgb, u_cam.fog_color.rgb, f), lit.a);
    }

    // Post-process color correction (P key): approximates the original D3D9
    // game's sRGB-space lighting by lifting shadows, nudging gamma, and
    // boosting saturation.  The curve parameters are tuned to get close to
    // the "wrong but traditional" look without leaving linear space.
    if _color_correct {
        let lum = dot(lit.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
        let c = lit.rgb * 0.92 + 0.04;                          // scale + lift
        let g = pow(max(c, vec3<f32>(0.0)), vec3<f32>(0.96));   // gamma
        lit = vec4<f32>(mix(vec3<f32>(lum), g, 1.10), lit.a);   // saturation
    }

    // SO coloring mode (O key): green = room geometry, yellow = SO entity.
    let _so_color = (_mode & 4u) != 0u;
    if _so_color {
        if in.v_mesh_type == 1u {
            lit = vec4<f32>(1.0, 1.0, 0.0, 1.0);  // yellow = SO
        } else {
            lit = vec4<f32>(0.0, 1.0, 0.0, 1.0);  // green = room geometry
        }
    }

    return lit;
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
