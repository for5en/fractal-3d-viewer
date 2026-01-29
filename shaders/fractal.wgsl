//     -----     SHADER     -----     //

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
};

struct Settings {
    pos: vec3<f32>,
    bailout: f32,

    forward: vec3<f32>,
    max_steps: f32,

    right: vec3<f32>,
    fov: f32,

    up: vec3<f32>,
    iterations: i32,

    constant: vec4<f32>,

    threshold: f32,
    power: f32,
    coloring: i32,
    fractal: i32,
}

struct Locals {
    time: f32,
}

fn calc_normal(p: vec3<f32>) -> vec3<f32> 
{
    let eps = 0.001;
    let dx = vec3<f32>(eps, 0.0, 0.0);
    let dy = vec3<f32>(0.0, eps, 0.0);
    let dz = vec3<f32>(0.0, 0.0, eps);

    let nx = mainDE(p + dx) - mainDE(p - dx);
    let ny = mainDE(p + dy) - mainDE(p - dy);
    let nz = mainDE(p + dz) - mainDE(p - dz);

    return normalize(vec3<f32>(nx, ny, nz));
}


//     -----     COLORINGS     -----     //

fn color_polish(hit: bool, z: vec3<f32>, cam_pos: vec3<f32>) -> vec4<f32>
{
    var color = vec3<f32>(0.0, 0.0, 0.0);

    if (hit) {
        let normal = calc_normal(z);
        let light_dir = normalize(vec3<f32>(-0.5, 1.0, -0.5));
        let view_dir = normalize(cam_pos - z);

        let ambient = 0.2;
        let diffuse = clamp(dot(normal, light_dir), 0.0, 1.0);
        let reflect_dir = normalize(2.0 * dot(normal, light_dir) * normal - light_dir);
        let spec = pow(clamp(dot(view_dir, reflect_dir), 0.0, 1.0), 32.0);

        let base_color = 0.5 + 0.5 * normal;

        color = base_color * (ambient + 0.7*diffuse) + vec3<f32>(spec);
    }

    return vec4<f32>(color, 1.0);
}

fn rainbow_gloss(hit: bool, z: vec3<f32>, cam_pos: vec3<f32>) -> vec4<f32> {
    var color = vec3<f32>(0.0, 0.0, 0.0);

    if (hit) {
        let normal = calc_normal(z);
        let light_dir = normalize(vec3<f32>(-0.5, 1.0, -0.5));
        let view_dir = normalize(cam_pos - z);

        // Fresnel-like effect
        let fresnel = pow(1.0 - dot(normal, view_dir), 3.0);

        // Color gradient depending on normal direction
        let base_color = 0.5 + 0.5 * normal;
        
        // Add some "rainbow-ish" shift based on position
        let rainbow = vec3<f32>(
            0.5 + 0.5 * sin(z.x * 3.0 + r_locals.time),
            0.5 + 0.5 * sin(z.y * 5.0 + r_locals.time),
            0.5 + 0.5 * sin(z.z * 7.0 + r_locals.time)
        );

        // Diffuse + ambient
        let ambient = 0.2;
        let diffuse = clamp(dot(normal, light_dir), 0.0, 1.0);

        // Combine everything
        color = mix(base_color * (ambient + 0.7*diffuse), rainbow, fresnel);
    }
    
    return vec4<f32>(color, 1.0);
}

fn rainbow_glitter(hit: bool, z: vec3<f32>, cam_pos: vec3<f32>) -> vec4<f32> {
    var color = vec3<f32>(0.0, 0.0, 0.0);

    if (hit) {
        let normal = calc_normal(z);
        let light_dir = normalize(vec3<f32>(-0.6, 1.0, -0.4));
        let view_dir = normalize(cam_pos - z);

        // --- Fresnel effect ---
        let fresnel = pow(1.0 - dot(normal, view_dir), 4.0);

        // --- Base color from normal (dynamic gradient) ---
        let base_color = 0.4 + 0.6 * normal;

        // --- Time-varying "rainbow wave" ---
        let rainbow = vec3<f32>(
            0.5 + 0.5 * sin(z.x * 5.0 + r_locals.time * 1.5),
            0.5 + 0.5 * sin(z.y * 4.0 - r_locals.time * 1.2),
            0.5 + 0.5 * sin(z.z * 7.0 + r_locals.time * 2.0)
        );

        // --- Soft dynamic noise based on position ---
        let noise = 0.1 * vec3<f32>(
            sin(z.x*12.0 + r_locals.time*3.0),
            cos(z.y*9.0 + r_locals.time*2.0),
            sin(z.z*15.0 - r_locals.time*2.5)
        );

        // --- Diffuse + ambient lighting ---
        let ambient = 0.25;
        let diffuse = clamp(dot(normal, light_dir), 0.0, 1.0);

        // --- Combine everything with Fresnel controlling "glow edges" ---
        color = mix(base_color * (ambient + 0.7*diffuse), rainbow + noise, fresnel);

        // Optional: slight exponential contrast for punchiness
        color = pow(color, vec3<f32>(1.2, 1.2, 1.2));
    }

    return vec4<f32>(color, 1.0);
}

fn rainbow(hit: bool, z: vec3<f32>, cam_pos: vec3<f32>) -> vec4<f32> {
    var color = vec3<f32>(0.0, 0.0, 0.0);

    if (hit) {
        let normal = calc_normal(z);
        let view_dir = normalize(cam_pos - z);
        let dist = length(cam_pos - z);

        // --- angle-based glow ---
        let angle_factor = pow(1.0 - dot(normal, view_dir), 5.0);

        // --- swirling colors based on position and time ---
        let swirl = vec3<f32>(
            sin(z.x*4.0 - r_locals.time*2.0) + cos(z.y*3.0 + r_locals.time),
            sin(z.y*5.0 + r_locals.time*1.5) - cos(z.z*2.0 - r_locals.time*1.2),
            sin(z.z*3.0 + r_locals.time*2.5) + cos(z.x*4.0 - r_locals.time*0.7)
        );

        // --- distance-based shading (soft fog) ---
        let fog = exp(-dist * 0.2);

        // --- normal-based tint ---
        let normal_tint = 0.5 + 0.5 * normal;

        // --- combine everything ---
        color = swirl * 0.6 + normal_tint * 0.8;
        color = mix(color, vec3<f32>(1.0,1.0,1.0), angle_factor);
        color = color * fog;

        // --- optional: enhance contrast for punchiness ---
        color = pow(color, vec3<f32>(1.3, 1.3, 1.3));
    }

    return vec4<f32>(color, 1.0);
}

fn sandstorm(hit: bool, z: vec3<f32>, cam_pos: vec3<f32>) -> vec4<f32> {
    var color = vec3<f32>(0.0);

    if (hit) {
        let normal = calc_normal(z);
        let view_dir = normalize(cam_pos - z);

        // intensywność zależna od wysokości i czasu
        let flames = 0.5 + 0.5 * sin(z.y * 10.0 + r_locals.time * 5.0);

        // gradient ognia
        let base = vec3<f32>(1.0, 0.5, 0.0) * flames; // pomarańcz
        let hot = vec3<f32>(1.0, 1.0, 0.3) * flames; // żółć
        color = mix(base, hot, clamp(dot(normal, view_dir), 0.0, 1.0));

        // lekka mgła
        color *= exp(-length(z - cam_pos) * 0.2);
        color = pow(color, vec3<f32>(1.1));
    }

    return vec4<f32>(color, 1.0);
}

fn white_polish(hit: bool, z: vec3<f32>, cam_pos: vec3<f32>) -> vec4<f32> {
    var color = vec3<f32>(0.0);

    if (hit) {
        let normal = calc_normal(z);
        let light_dir = normalize(vec3<f32>(-0.7, 1.0, -0.5));
        let view_dir = normalize(cam_pos - z);

        // ambient i diffuse
        let ambient = 0.2;
        let diffuse = clamp(dot(normal, light_dir), 0.0, 1.0);

        // specular
        let reflect_dir = normalize(2.0 * dot(normal, light_dir) * normal - light_dir);
        let spec = pow(clamp(dot(view_dir, reflect_dir), 0.0, 1.0), 32.0);

        // głębia koloru w zależności od odległości od kamery
        let depth = clamp(length(cam_pos - z) / 10.0, 0.0, 1.0);

        // kolory bazowe gradientowe, żeby każdy detal był czytelny
        let base_color = mix(
            vec3<f32>(0.2, 0.3, 0.6),  // ciemniejszy niebieski
            vec3<f32>(0.8, 0.5, 0.2),  // cieplejszy brąz/pomarańcz
            normal.y * 0.5 + 0.5       // gradient wg normalnej
        );

        // łączymy wszystko
        color = base_color * (ambient + 0.8 * diffuse) + vec3<f32>(spec);

        // dodajemy lekki wpływ głębi
        color = mix(color, color * vec3<f32>(1.0 - depth), 0.3);

        // delikatne wyrównanie kolorów
        color = pow(color, vec3<f32>(0.9));
    }

    return vec4<f32>(color, 1.0);
}

fn rainbow_mat(hit: bool, z: vec3<f32>, cam_pos: vec3<f32>) -> vec4<f32> {
    var color = vec3<f32>(0.0);
    let time = r_locals.time;

    if (hit) {
        let n = calc_normal(z);
        let light_dir = normalize(vec3<f32>(-0.7, 1.0, -0.5));
        let view_dir = normalize(cam_pos - z);

        // --- ambient + diffuse ---
        let ambient = 0.25;
        let diffuse = clamp(dot(n, light_dir), 0.0, 1.0);

        // --- specular ---
        let reflect_dir = normalize(2.0 * dot(n, light_dir) * n - light_dir);
        let spec = pow(clamp(dot(view_dir, reflect_dir), 0.0, 1.0), 64.0);

        // --- dynamic RGB pattern ---
        let r = 0.5 + 0.5 * sin(5.0 * z.x + time);
        let g = 0.5 + 0.5 * sin(5.0 * z.y + time * 1.3);
        let b = 0.5 + 0.5 * sin(5.0 * z.z + time * 1.7);

        var base_color = vec3<f32>(r, g, b);

        // --- gradient wg normalnych ---
        base_color = mix(base_color, 0.5 + 0.5 * n, 0.6);

        // --- delikatny wzór zależny od pozycji (detale) ---
        let pattern = sin(z.x*10.0) * cos(z.y*10.0) * sin(z.z*10.0);
        base_color += vec3<f32>(0.1, 0.08, 0.06) * pattern;

        // --- fog / głębia ---
        let dist = length(cam_pos - z);
        let fog = clamp(dist / 10.0, 0.0, 1.0);
        base_color = mix(base_color, base_color * 0.3, fog);

        // --- final gamma correction i clamp ---
        color = clamp(pow(base_color * (ambient + 0.9 * diffuse) + vec3<f32>(spec), vec3<f32>(0.9)), vec3<f32>(0.0), vec3<f32>(1.0));
    }

    return vec4<f32>(color, 1.0);
}

fn epilepsy(hit: bool, z: vec3<f32>, cam_pos: vec3<f32>) -> vec4<f32> {
    if (!hit) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    let time = r_locals.time;

    let n = calc_normal(z);
    let v = normalize(cam_pos - z);

    // --------- HARD LIGHT ---------
    let l1 = normalize(vec3<f32>( 1.0,  1.0, -1.0));
    let l2 = normalize(vec3<f32>(-1.0,  0.5,  1.0));
    let d1 = dot(n, l1);
    let d2 = dot(n, l2);

    // --------- SPACE DISTORTION ---------
    let p = z * 15.0;

    // --------- CHAOS PHASES ---------
    let t1 = time * 12.0;
    let t2 = time * 25.0;
    let t3 = time * 40.0;

    // --------- FRACTAL INTERFERENCE ---------
    let f1 = sin(p.x * 8.0 + t1) * sin(p.y * 9.0 - t2);
    let f2 = cos(p.y * 11.0 + t2) * sin(p.z * 7.0 + t3);
    let f3 = sin(p.z * 13.0 - t3) * cos(p.x * 10.0 + t1);

    let chaos = f1 + f2 + f3;

    // --------- MOIRÉ / ALIAS PATTERNS ---------
    let grid1 = step(0.0, sin(p.x * 20.0 + time * 30.0));
    let grid2 = step(0.0, sin(p.y * 21.0 - time * 35.0));
    let grid3 = step(0.0, sin(p.z * 22.0 + time * 40.0));
    let grid = grid1 * grid2 + grid3;

    // --------- SPIRAL PHASE ---------
    let r = length(z.xy);
    let spiral = sin(r * 40.0 - time * 50.0);

    // --------- RGB PHASE SHIFT ---------
    let r_col = sin(chaos * 6.0 + spiral * 8.0 + t1);
    let g_col = sin(chaos * 7.0 + spiral * 9.0 + t2 + 2.1);
    let b_col = sin(chaos * 8.0 + spiral * 10.0 + t3 + 4.2);

    var color = vec3<f32>(r_col, g_col, b_col);

    // --------- NORMAL FEEDBACK (DETAIL BOOST) ---------
    color += 1.5 * vec3<f32>(
        sin(n.x * 30.0 + time * 20.0),
        sin(n.y * 30.0 - time * 25.0),
        sin(n.z * 30.0 + time * 30.0)
    );

    // --------- VIEW DEPENDENT FLASH ---------
    let flash = pow(1.0 - dot(n, v), 8.0);
    color += flash * vec3<f32>(5.0, 3.0, 6.0);

    // --------- STROBE ---------
    let strobe = step(0.0, sin(time * 60.0));
    color *= mix(0.2, 3.0, strobe);

    // --------- CONTRAST OVERDRIVE ---------
    color = sign(color) * pow(abs(color), vec3<f32>(0.3));

    // --------- CLAMP IS FOR WEAK ---------
    return vec4<f32>(color, 1.0);
}






fn main_coloring(hit: bool, z: vec3<f32>, cam_pos: vec3<f32>) -> vec4<f32> {
    switch(u32(settings.coloring)) {
        case 0u: { return white_polish(hit, z, cam_pos); }
        case 1u: { return color_polish(hit, z, cam_pos); }
        case 2u: { return rainbow_mat(hit, z, cam_pos); }
        case 3u: { return rainbow_gloss(hit, z, cam_pos); }
        case 4u: { return rainbow_glitter(hit, z, cam_pos); }
        case 5u: { return rainbow(hit, z, cam_pos); }
        case 6u: { return sandstorm(hit, z, cam_pos); }
        case 7u: { return epilepsy(hit, z, cam_pos); }
        default: { return vec4<f32>(1.0, 0.0, 1.0, 1.0); }
    }
}

//     -----     DISTANCE ESTIMATORS     -----     //

fn mandelbulbDE(p: vec3<f32>) -> f32 
{
    var z = p;
    var dr: f32 = 1.0;
    var r: f32 = 0.0;

    for (var i: i32 = 0; i < settings.iterations; i = i + 1) 
    {
        r = length(z);
        if (r > 4.0) { break; }

        let theta = acos(z.z / r);
        let phi = atan2(z.y, z.x);

        dr = pow(r, settings.power - 1.0) * settings.power * dr + 1.0;

        let zr = pow(r, settings.power);
        let sin_theta = sin(theta * settings.power);
        let cos_phi = cos(phi * settings.power);
        let sin_phi = sin(phi * settings.power);

        z = zr * vec3<f32>(
            sin_theta * cos_phi,
            sin_theta * sin_phi,
            cos(theta * settings.power)
        ) + p;
    }

    return 0.5 * log(r) * r / dr;
}

fn juliabulbDE(p: vec3<f32>, _c: vec4<f32>) -> f32 {
    let c = vec3<f32>(_c.x, _c.y, _c.z);
    var z = p;
    var dr: f32 = 1.0;
    var r: f32 = 0.0;

    for (var i: i32 = 0; i < settings.iterations; i = i + 1) {
        r = length(z);
        if (r > 4.0) { break; }

        let theta = acos(z.z / r);
        let phi = atan2(z.y, z.x);

        dr = pow(r, settings.power - 1.0) * settings.power * dr + 1.0;

        let zr = pow(r, settings.power);
        let sin_theta = sin(theta * settings.power);
        let cos_phi = cos(phi * settings.power);
        let sin_phi = sin(phi * settings.power);

        z = zr * vec3<f32>(
            sin_theta * cos_phi,
            sin_theta * sin_phi,
            cos(theta * settings.power)
        ) + c;
    }

    return 0.5 * log(r) * r / dr;
}

fn mengerDE(p: vec3<f32>) -> f32 {
    var scale: f32 = 1.0;
    var z = p;

    for (var i: i32 = 0; i < settings.iterations; i = i + 1) {
        z = abs(z);
        if (z.x < z.y) { z = vec3<f32>(z.y, z.x, z.z); }
        if (z.x < z.z) { z = vec3<f32>(z.z, z.y, z.x); }

        z = z * 3.0 - vec3<f32>(2.0, 2.0, 2.0);
        scale *= 3.0;
    }

    let d = length(max(z - vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.0)));
    return d / scale; // tylko tu dzielimy
}


fn sierpinskiDE(p: vec3<f32>) -> f32 {
    var scale: f32 = 1.0;
    var z = p;

    for (var i: i32 = 0; i < settings.iterations; i = i + 1) {
        // Odbicia względem płaszczyzn
        if (z.x + z.y < 0.0) { z = vec3<f32>(-z.x, -z.y, z.z); }
        if (z.x + z.z < 0.0) { z = vec3<f32>(-z.x, z.y, -z.z); }
        if (z.y + z.z < 0.0) { z = vec3<f32>(z.x, -z.y, -z.z); }

        // Skalowanie i translacja
        z = z * 2.0 - vec3<f32>(1.0, 1.0, 1.0);
        scale *= 2.0;
    }

    return (length(z) - 0.5) / scale;
}

fn torusDE(p: vec3<f32>, _c: vec4<f32>) -> f32 {
    let t = vec2<f32>(_c.x, _c.y);
    var q = vec2<f32>(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

/*fn mandelboxDE(p: vec3<f32>) -> f32 {
    var z = p;
    var scale: f32 = 1.0;           // typowa skala Mandelboxa
    var offset: f32 = 1.0;
    var min_radius: f32 = 0.5;
    var fixed_radius: f32 = 1.0;
    var dr: f32 = 1.0;

    for (var i: i32 = 0; i < settings.iterations; i = i + 1) {
        // Box fold
        z.x = clamp(z.x, -1.0, 1.0) * 2.0 - z.x;
        z.y = clamp(z.y, -1.0, 1.0) * 2.0 - z.y;
        z.z = clamp(z.z, -1.0, 1.0) * 2.0 - z.z;



        // Sphere fold
        var r = length(z);
        if (r < min_radius) {
            z = z * (fixed_radius / min_radius);
            dr = dr * (fixed_radius / min_radius);
        } else if (r < fixed_radius) {
            z = z * (fixed_radius / r);
            dr = dr * (fixed_radius / r);
        }

        z = z * scale + p;
        dr = dr * abs(scale) + 1.0;
    }

    return length(z) / abs(dr);
}

fn multibrotDE(p: vec3<f32>) -> f32 {
    var z = p;
    var dr: f32 = 1.0;
    var r: f32 = 0.0;

    for (var i: i32 = 0; i < settings.iterations; i = i + 1) {
        r = length(z);
        if (r > 4.0) { break; }

        let theta = acos(z.z / r);
        let phi = atan2(z.y, z.x);

        dr = pow(r, settings.power - 1.0) * settings.power * dr + 1.0;

        let zr = pow(r, settings.power);
        let sin_theta = sin(theta * settings.power);
        let cos_phi = cos(phi * settings.power);
        let sin_phi = sin(phi * settings.power);

        z = zr * vec3<f32>(
            sin_theta * cos_phi,
            sin_theta * sin_phi,
            cos(theta * settings.power)
        ) + p;
    }

    return 0.5 * log(r) * r / dr;
}*/

fn mandelbulbQuatDE(p: vec3<f32>, c: vec4<f32>) -> f32 {
    var z = vec4<f32>(p.x, p.y, p.z, c.x);
    var dr: f32 = 1.0;
    var r: f32 = 0.0;

    for (var i: i32 = 0; i < settings.iterations; i = i + 1) {
        r = length(z.xyzw);
        if (r > 4.0) { break; }

        let theta = acos(z.z / r);
        let phi   = atan2(z.y, z.x);
        let w     = z.w;

        dr = pow(r, settings.power - 1.0) * settings.power * dr + 1.0;
        let zr = pow(r, settings.power);

        let new_xyz = zr * vec3<f32>(
            sin(theta * settings.power) * cos(phi * settings.power),
            sin(theta * settings.power) * sin(phi * settings.power),
            cos(theta * settings.power)
        ) + p;

        z = vec4<f32>(new_xyz.x, new_xyz.y, new_xyz.z, z.w);


        z.w = pow(w, settings.power) + 0.0;
    }

    return 0.5 * log(r) * r / dr;
}

fn juliabulbQuatDE(p: vec3<f32>, c: vec4<f32>) -> f32 {
    var z = vec4<f32>(p.x, p.y, p.z, c.x);
    var dr: f32 = 1.0;
    var r: f32 = 0.0;

    for (var i: i32 = 0; i < settings.iterations; i = i + 1) {
        r = length(z.xyzw);
        if (r > 4.0) { break; }

        let theta = acos(z.z / r);
        let phi   = atan2(z.y, z.x);
        let w     = z.w;

        // DE
        dr = pow(r, settings.power - 1.0) * settings.power * dr + 1.0;
        let zr = pow(r, settings.power);

        // rotacja w 3D
        let new_xyz = zr * vec3<f32>(
            sin(theta * settings.power) * cos(phi * settings.power),
            sin(theta * settings.power) * sin(phi * settings.power),
            cos(theta * settings.power)
        ) + c.xyz;

        z = vec4<f32>(new_xyz.x, new_xyz.y, new_xyz.z, z.w);

        z.w = pow(w, settings.power) + c.w;
    }

    return 0.5 * log(r) * r / dr;
}


fn mandeljulia_minDE(p: vec3<f32>, c: vec4<f32>) -> f32 {
    let mb = mandelbulbDE(p);
    let julia = juliabulbDE(p, c);
    return min(mb, julia);
}

fn mandeljulia_maxDE(p: vec3<f32>, c: vec4<f32>) -> f32 {
    let mb = mandelbulbDE(p);
    let julia = juliabulbDE(p, c);
    return max(mb, julia);
}



fn mainDE(p: vec3<f32>) -> f32 
{
    switch(u32(settings.fractal)) {
        case 0u: { return mandelbulbDE(p); }
        case 1u: { return juliabulbDE(p, settings.constant); }
        case 2u: { return mengerDE(p); }
        case 3u: { return sierpinskiDE(p); }
        case 4u: { return torusDE(p, settings.constant); }
        case 5u: { return mandelbulbQuatDE(p, settings.constant); }
        case 6u: { return juliabulbQuatDE(p, settings.constant); }
        case 7u: { return mandeljulia_minDE(p, settings.constant); }
        case 8u: { return mandeljulia_maxDE(p, settings.constant); }
        default: { return 0.0; }
    }
}

//     -----     SHADER BODY     -----     //

@group(0) @binding(2) 
var<uniform> r_locals: Locals;

@group(0) @binding(3)
var<uniform> settings: Settings;

@vertex
fn vs_main(@location(0) position: vec2<f32>) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(position, 0.0, 1.0);
    out.tex_coord = position * 0.5 + vec2<f32>(0.5, 0.5); // [-1,1] -> [0,1]
    return out;
}

@fragment
fn fs_main(@location(0) tex_coord: vec2<f32>) -> @location(0) vec4<f32> {
    let cam_pos = settings.pos;
    let forward = settings.forward;
    let right = settings.right;
    let up = settings.up;
    let fov = settings.fov;

    let sx = (tex_coord.x * 2.0 - 1.0) * 16/9;
    let sy = tex_coord.y * 2.0 - 1.0;
    var ray_dir = normalize(forward + sx*fov*right + sy*fov*up);

    var z = cam_pos;
    var total_dist: f32 = 0.0;
    var hit = false;

    for (var i = 0.0; i < settings.max_steps; i = i + 1.0) 
    {
        let dist = mainDE(z);
        z = z + ray_dir * dist;
        total_dist = total_dist + dist;

        if (dist < settings.threshold) 
        {
            hit = true;
            break;
        }
    }

    return main_coloring(hit, z, cam_pos);
}

