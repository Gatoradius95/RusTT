//! Offscreen skeleton-pose renderer: draws the animated skeleton as lines for
//! a set of animations/frames under two rotation conventions (current vs
//! Z-mirror negXYZ), side by side, to diagnose leg orientation questions.
use std::path::Path;

use anyhow::{Context, Result};
use glam::{Mat4, Vec3};
use rustt::an3::An3;
use rustt::ghg;

fn joint(w: &Mat4) -> Vec3 {
    w.transform_point3(Vec3::ZERO)
}

/// parent-relative rotation honoring the 0x20 logic, with optional euler
/// mirroring (convention "negXYZ" negates all three euler angles).
fn bone_rot(an3: &An3, bone: usize, frame: f32, rest_local: Option<&Mat4>, neg: bool) -> Mat4 {
    let sx = an3.channel_value(bone, 3, frame);
    let sy = an3.channel_value(bone, 4, frame);
    let sz = an3.channel_value(bone, 5, frame);
    let (x, y, z) = if neg { (-sx, -sy, -sz) } else { (sx, sy, sz) };
    let r_anim = Mat4::from_rotation_z(z) * Mat4::from_rotation_y(y) * Mat4::from_rotation_x(x);
    if an3.uses_x20(bone) {
        match rest_local {
            Some(rl) => {
                let rl = Mat4::from_mat3(glam::Mat3::from_mat4(*rl));
                if an3.footer.get(bone).map_or(false, |f| f & 0x01 != 0) {
                    rl * r_anim
                } else {
                    rl
                }
            }
            None => r_anim,
        }
    } else {
        r_anim
    }
}

fn worlds_at(an3: &An3, parents: &[i32], rest_locals: &[Mat4], frame: f32, neg: bool) -> Vec<Mat4> {
    let mut worlds = Vec::with_capacity(an3.num_bones);
    for b in 0..an3.num_bones {
        let t = Vec3::new(
            an3.channel_value(b, 0, frame),
            an3.channel_value(b, 1, frame),
            -an3.channel_value(b, 2, frame),
        );
        let r = bone_rot(an3, b, frame, rest_locals.get(b), neg);
        let local = Mat4::from_translation(t) * r;
        let w = if parents[b] < 0 {
            local
        } else {
            worlds[parents[b] as usize] * local
        };
        worlds.push(w);
    }
    worlds
}

const W: usize = 640;
const H: usize = 860;

struct Canvas {
    buf: Vec<u8>,
}

impl Canvas {
    fn new() -> Self {
        Self { buf: vec![0u8; W * H * 4] }
    }
    fn px(&mut self, x: usize, y: usize, c: [u8; 3]) {
        if x >= W || y >= H {
            return;
        }
        let i = (y * W + x) * 4;
        self.buf[i..i + 3].copy_from_slice(&c);
        self.buf[i + 3] = 255;
    }
    fn line(&mut self, a: [f32; 2], b: [f32; 2], c: [u8; 3], th: i32) {
        let mut x0 = a[0] as i32;
        let mut y0 = a[1] as i32;
        let x1 = b[0] as i32;
        let y1 = b[1] as i32;
        let dx = (x1 - x0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let dy = -(y1 - y0).abs();
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        loop {
            for ty in -th..=th {
                for tx in -th..=th {
                    self.px((x0 + tx) as usize, (y0 + ty) as usize, c);
                }
            }
            if x0 == x1 && y0 == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x0 += sx;
            }
            if e2 <= dx {
                err += dx;
                y0 += sy;
            }
        }
    }
}

struct Mapper {
    s: f32,
    cx: f32,
    cy: f32,
    midz: f32,
    miny: f32,
}

impl Mapper {
    fn new(joints: &[Vec3]) -> Self {
        let mut minz = f32::INFINITY;
        let mut maxz = f32::NEG_INFINITY;
        let mut miny = f32::INFINITY;
        let mut maxy = f32::NEG_INFINITY;
        for j in joints {
            minz = minz.min(j.z);
            maxz = maxz.max(j.z);
            miny = miny.min(j.y);
            maxy = maxy.max(j.y);
        }
        let span = (maxz - minz).max(maxy - miny).max(1e-3);
        let s = (W as f32 * 0.72) / span;
        let cx = W as f32 / 2.0;
        let cy = H as f32 * 0.62;
        Self {
            s,
            cx,
            cy,
            midz: (minz + maxz) / 2.0,
            miny,
        }
    }
    fn map(&self, j: &Vec3) -> [f32; 2] {
        [
            self.cx + (j.z - self.midz) * self.s,
            self.cy - (j.y - self.miny) * self.s * 0.72,
        ]
    }
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .context("usage: pose_render <model.ghg> <out.png> [anim.an3 [frame] ...]")?;
    let out_path = args.next().context("missing out.png")?;

    let data = std::fs::read(&model_path)?;
    let p = ghg::parse(&data)?;
    let parents: Vec<i32> = p.bones.iter().map(|b| b.parent).collect();
    let rest_locals: Vec<Mat4> = p.bones.iter().map(|b| b.local).collect();

    let mut items: Vec<(String, An3, f32)> = Vec::new();
    while let Some(a) = args.next() {
        let frame: f32 = args.next().and_then(|f| f.parse().ok()).unwrap_or(0.0);
        let name = Path::new(&a)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| a.clone());
        let anim = An3::parse(&std::fs::read(&a)?)?;
        items.push((name, anim, frame));
    }
    if items.is_empty() {
        anyhow::bail!("no animations given");
    }

    let rows = items.len();
    let img_w = W * 2;
    let img_h = rows * H;
    let mut img = vec![0u8; img_w * img_h * 4];
    for px in img.chunks_exact_mut(4) {
        px.copy_from_slice(&[18, 20, 26, 255]);
    }

    for (ri, (name, an3, frame)) in items.iter().enumerate() {
        let oy = ri * H;
        println!("row {ri}: {name}  frame {frame:.1}");
        for (ci, neg) in [false, true].into_iter().enumerate() {
            let worlds = worlds_at(an3, &parents, &rest_locals, *frame, neg);
            let joints: Vec<Vec3> = worlds.iter().map(joint).collect();
            let m = Mapper::new(&joints);
            let mut canvas = Canvas::new();
            for b in 0..an3.num_bones {
                if parents[b] >= 0 && (parents[b] as usize) < joints.len() {
                    let c = if (23..=30).contains(&b) {
                        [240, 90, 90]
                    } else {
                        [200, 200, 210]
                    };
                    canvas.line(
                        m.map(&joints[b]),
                        m.map(&joints[parents[b] as usize]),
                        c,
                        2,
                    );
                }
            }
            for b in 23..=30 {
                let p = m.map(&joints[b]);
                for ty in -3i32..=3 {
                    for tx in -3i32..=3 {
                        canvas.px(
                            (p[0] as i32 + tx) as usize,
                            (p[1] as i32 + ty) as usize,
                            [250, 220, 60],
                        );
                    }
                }
            }
            let ox = ci * W;
            for y in 0..H {
                for x in 0..W {
                    let si = (y * W + x) * 4;
                    let di = ((oy + y) * img_w + (ox + x)) * 4;
                    img[di..di + 4].copy_from_slice(&canvas.buf[si..si + 4]);
                }
            }
        }
    }

    let mut enc = png::Encoder::new(
        std::fs::File::create(&out_path)?,
        img_w as u32,
        img_h as u32,
    );
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut wr = enc.write_header()?;
    wr.write_image_data(&img)?;
    println!("wrote {out_path} ({}x{})", img_w, img_h);
    Ok(())
}
