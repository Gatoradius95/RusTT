use glam::{Mat4, Vec3};

const UP: Vec3 = Vec3::Y;

pub struct OrbitCamera {
    pub target: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            yaw: 0.7,
            pitch: 0.35,
            distance: 6.0,
            fov_y: 45f32.to_radians(),
            near: 0.1,
            far: 10000.0,
        }
    }
}

impl OrbitCamera {
    pub fn position(&self) -> Vec3 {
        let (sp, cp) = self.pitch.sin_cos();
        let (sy, cy) = self.yaw.sin_cos();
        self.target + Vec3::new(cp * cy, sp, cp * sy) * self.distance
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        glam::camera::lh::proj::directx::perspective(self.fov_y, aspect, self.near, self.far)
            * glam::camera::lh::view::look_at_mat4(self.position(), self.target, UP)
    }

    pub fn frame(&mut self, bounds: &crate::scene::Bounds) {
        let d = bounds.radius.max(0.001);
        self.distance = d * 2.4;
        self.target = bounds.center;
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Horizontal (XZ-plane) forward vector pointing from the eye toward the target.
    fn forward_h(&self) -> Vec3 {
        let v = self.target - self.position();
        let h = Vec3::new(v.x, 0.0, v.z);
        if h.length_squared() < 1e-8 {
            Vec3::NEG_Z
        } else {
            h.normalize()
        }
    }

    fn right(&self) -> Vec3 {
        // Left-handed view (glam `lh`): screen-right = `up × forward`, the
        // opposite of the right-handed `forward × up`.
        UP.cross(self.forward_h()).normalize()
    }

    /// Orbit around the target with a cursor delta in pixels.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.008;
        self.pitch = (self.pitch + dy * 0.008).clamp(-1.55, 1.55);
    }

    /// Pan the target in the camera plane; `dx`/`dy` in pixels.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        let s = self.distance * 0.0012;
        self.target -= self.right() * dx * s;
        self.target += UP * dy * s;
    }

    /// Zoom by a mouse-wheel delta (positive = wheel up = zoom in).
    pub fn zoom(&mut self, wheel: f32) {
        self.distance = (self.distance * (1.0 - wheel * 0.12)).clamp(0.1, 50000.0);
    }

    /// Move the target with a per-second speed vector from WASDQE.
    pub fn move_target(&mut self, fwd: f32, strafe: f32, up: f32, dt: f32) {
        let s = self.distance * 0.9 * dt;
        self.target += self.forward_h() * fwd * s;
        self.target += self.right() * strafe * s;
        self.target.y += up * s;
    }
}
