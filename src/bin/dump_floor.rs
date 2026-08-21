use rustt::map::parse;
use rustt::mapmesh::expand_all;

fn floor_at(meshes: &Vec<rustt::glb::MeshData>, sx: f32, sz: f32) -> f32 {
    let mut lowest = f32::MAX;
    for md in meshes {
        for tri in md.idx.chunks_exact(3) {
            let a = md.pos[tri[0] as usize];
            let b = md.pos[tri[1] as usize];
            let c = md.pos[tri[2] as usize];
            let minx = a[0].min(b[0]).min(c[0]);
            let maxx = a[0].max(b[0]).max(c[0]);
            let minz = a[2].min(b[2]).min(c[2]);
            let maxz = a[2].max(b[2]).max(c[2]);
            if sx < minx || sx > maxx || sz < minz || sz > maxz { continue; }
            let e0 = [b[0]-a[0], b[1]-a[1], b[2]-a[2]];
            let e1 = [c[0]-a[0], c[1]-a[1], c[2]-a[2]];
            let n = [e0[1]*e1[2]-e0[2]*e1[1], e0[2]*e1[0]-e0[0]*e1[2], e0[0]*e1[1]-e0[1]*e1[0]];
            if n[1].abs() < 1e-8 { continue; }
            let t = (n[0]*(sx-a[0]) + n[1]*(500.0-a[1]) + n[2]*(sz-a[2])) / n[1];
            if !(t >= 0.0) || t > 500.0 { continue; }
            let y = 500.0 - t;
            let v0 = [b[0]-a[0], b[1]-a[1], b[2]-a[2]];
            let v1 = [c[0]-a[0], c[1]-a[1], c[2]-a[2]];
            let v2 = [sx-a[0], y-a[1], sz-a[2]];
            let d00 = v0[0]*v0[0]+v0[1]*v0[1]+v0[2]*v0[2];
            let d01 = v0[0]*v1[0]+v0[1]*v1[1]+v0[2]*v1[2];
            let d11 = v1[0]*v1[0]+v1[1]*v1[1]+v1[2]*v1[2];
            let d20 = v2[0]*v0[0]+v2[1]*v0[1]+v2[2]*v0[2];
            let d21 = v2[0]*v1[0]+v2[1]*v1[1]+v2[2]*v1[2];
            let den = d00*d11-d01*d01;
            if den.abs() < 1e-12 { continue; }
            let vv = (d11*d20-d01*d21)/den;
            let ww = (d00*d21-d01*d20)/den;
            if vv>=0.0 && ww>=0.0 && vv+ww<=1.0 { lowest = lowest.min(y); }
        }
    }
    lowest
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data: &'static [u8] = Box::leak(std::fs::read("backup/LEVELS/MAP/MAP/MAP_PC.GSC")?.into_boxed_slice());
    let map = parse(data)?;
    let meshes = expand_all(&map);
    let pts = [
        ("spawn(MAINROOM trigger)", -27.49f32, -52.72f32),
        ("BARMAN_1 (bar)", -25.17, -48.22),
        ("JABBA_1 (jabba alcove)", -28.92, -73.07),
        ("MAINROOMIDLE_3", -28.14, -54.85),
        ("MAINROOMIDLE_1", -31.06, -51.82),
        ("SESSION trigger", -10.73, -54.70),
        ("BAND_1 (stage)", -30.13, -47.06),
    ];
    for (name, x, z) in pts {
        println!("{:<28} floor y={:.3}", name, floor_at(&meshes, x, z));
    }
    Ok(())
}
