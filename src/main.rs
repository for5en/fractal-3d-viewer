#![deny(clippy::all)]
#![forbid(unsafe_code)]

use std::process::Command;
use std::time::Instant;

use crate::renderers::FractalRenderer;

use error_iter::ErrorIter as _;
use log::error;
use pixels::{Error, Pixels, SurfaceTexture};
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::KeyCode;
use winit::window::WindowBuilder;
use winit_input_helper::WinitInputHelper;
use winit::window::CursorGrabMode;

mod renderers;
use renderers::FractalSettings;

// MANUAL SETTINGS
const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
//const FRACTALS: [&str; 11] = ["Mandelbulb", "Juliabulb", "Menger", "Sierpinski", "Torus", "Mandelbox", "Multibrot", "MandelbulbQuat", "JuliabulbQuat", "Mandeljulia Min", "Mandeljulia Max"];
const FRACTALS: [&str; 9] = ["Mandelbulb", "Juliabulb", "Menger", "Sierpinski", "Torus", "MandelbulbQuat", "JuliabulbQuat", "Mandeljulia Min", "Mandeljulia Max"];
const COLORINGS: [&str; 8] = ["White Polish", "Color Polish", "Rainbow Mat", "Rainbow Gloss", "Rainbow Glitter", "Rainbow","Sandstorm", "Epilepsy"];

struct FractalAnimationParams
{
    pub speed: f32,
    pub constant_ranges: [f32; 4],
    pub power_range: f32
}

pub struct CameraAnimationParams {
    pub radius: f32,         
    pub azimuth: f32,        
    pub height_amplitude: f32,  
    pub speed: f32,          
    pub phase: f32,
}


struct Settings {
    pub movement_speed: f32,
    pub mouse_sensitivity: f32,
    pub sprint_enabled: bool,
    pub camera_animation_enabled: bool,
    pub camera_animation_params: CameraAnimationParams,
    pub fractal_animation_enabled: bool,
    pub fractal_animation_params: FractalAnimationParams,
    pub fractal_settings: FractalSettings,
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

fn hash(x: f32) -> f32 {
    let mut h = x.sin() * 43758.5453;
    h = h - h.floor();
    h
}


pub fn animate_fractal(
    last_t: &mut f32,
    delta_t: f32,
    period: f32,
    power: f32,
    power_range: f32,
    constant: [f32; 4],
    constant_ranges: [f32; 4],
) -> (f32, [f32; 4]) {
    *last_t += delta_t / period;
    let t = *last_t;

    let id0 = t.floor();
    let id1 = id0 + 1.0;
    let local_t = smoothstep((t - id0).fract());

    let target_p0 = power + (hash(id0) * 2.0 - 1.0) * power_range;
    let target_p1 = power + (hash(id1) * 2.0 - 1.0) * power_range;
    let power_new = lerp(target_p0, target_p1, local_t);

    let mut c_new = [0.0; 4];
    for i in 0..4 {
        let target0 = constant[i] + (hash(id0 + i as f32 * 10.0) * 2.0 - 1.0) * constant_ranges[i];
        let target1 = constant[i] + (hash(id1 + i as f32 * 10.0) * 2.0 - 1.0) * constant_ranges[i];
        c_new[i] = lerp(target0, target1, local_t);
    }

    (power_new, c_new)
}

fn start_camera_animation(cam_pos: [f32; 3], cam: &mut CameraAnimationParams) {
    let x = cam_pos[0];
    let y = cam_pos[1];
    let z = cam_pos[2];

    cam.radius = (x*x + z*z).sqrt();

    cam.azimuth = z.atan2(x);

    cam.height_amplitude = y.abs().max(0.0001);

    cam.phase = (y / cam.height_amplitude).asin() - cam.azimuth;
}

fn animate_camera(cam: &mut CameraAnimationParams, delta_time: f32) -> [f32; 3] {
    cam.azimuth += delta_time * cam.speed;

    let r = cam.radius;
    let az = cam.azimuth;
    let h = cam.height_amplitude;
    let phase = cam.phase;

    let x = r * az.cos();
    let z = r * az.sin();
    let y = h * (az + phase).sin();

    [x, y, z]
}






fn write_log(fractal_settings: FractalSettings, movement_speed: f32, mouse_sensitivity: f32, time: f32)
{
    println!("
    LOG (TIME={})
        User's settings:
            camera position: (x: {}, y: {}, z: {})
            fov: {}
            movement_speed: {}
            mouse_sensitivity: {}

        Fractal's settings:
            fractal: {}
            constant: ({}, {}, {}, {})
            coloring: {}
            maximum iterations: {}
            power: {}
            bailout: {}

        Raymarching:
            maximum steps: {}
            hit threshold: {}
    ",
    time,
    fractal_settings.pos[0],
    fractal_settings.pos[1],
    fractal_settings.pos[2],
    fractal_settings.fov,
    movement_speed,
    mouse_sensitivity,

    FRACTALS[fractal_settings.fractal as usize],
    fractal_settings.constant[0],
    fractal_settings.constant[1],
    fractal_settings.constant[2],
    fractal_settings.constant[3],
    COLORINGS[fractal_settings.coloring as usize],
    fractal_settings.iterations,
    fractal_settings.power,
    fractal_settings.bailout,

    fractal_settings.max_steps,
    fractal_settings.threshold
    )
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
    [v[0]/len, v[1]/len, v[2]/len]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ]
}

fn main() -> Result<(), Error> {
    let status = Command::new("xinput")
        .arg("set-prop")
        .arg("12")
        .arg("libinput Disable While Typing Enabled")
        .arg("0")
        .status()
        .expect("Nie udało się uruchomić xinput");

    println!("Status zmiany ustawien touchpada: {}", status);

    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    let mut input = WinitInputHelper::new();
    let window = {
        let size = LogicalSize::new(WIDTH as f64, HEIGHT as f64);
        WindowBuilder::new()
            .with_title(&format!("Fractal 3D: building shader, keep waiting..."))
            .with_inner_size(size)
            .with_min_inner_size(size)
            .build(&event_loop)
            .unwrap()
    };
    window.set_cursor_grab(CursorGrabMode::Confined).ok();
    window.set_cursor_visible(false);

    let window_size = window.inner_size();
    let mut pixels = {
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
        Pixels::new(WIDTH, HEIGHT, surface_texture)?
    };

    // GLOBAL SETTINGS
    let fractals_total = FRACTALS.len() as i32;
    let colorings_total = COLORINGS.len() as i32;
    let mut last_frame = Instant::now();
    let mut time = 0.0;
    let mut animation_time = 0.0;
    let mut delta_time= 0.0;
    let mut window_locked = true;
    let mut option = 'p';
    let mut constant_holder = [0.0, 0.0, 0.0, 0.0];
    let mut power_holder = 0.0;

    // FRACTALS SETTINGS
    let mut fractal = 0;

    let mut settings: Vec<Settings> = Vec::new();
    for i in 0..(fractals_total as usize)
    {
        settings.push(Settings
        {
            movement_speed: 0.1,
            mouse_sensitivity: 0.002,
            sprint_enabled: false,
            camera_animation_enabled: false,
            camera_animation_params: CameraAnimationParams { radius: 0.0, height_amplitude: 0.0, speed: 0.0, azimuth: 0.0, phase: 0.0 },
            fractal_animation_enabled: false,
            fractal_animation_params: FractalAnimationParams { speed: 1.0, constant_ranges: [1.0, 1.0, 1.0, 1.0], power_range: 1.0 },
            fractal_settings: FractalSettings {
                                            pos: [-2.0, 0.0, 0.0],
                                            forward: [1.0, 0.0, 0.0],
                                            right: [0.0, 1.0, 0.0],
                                            up: [0.0, 0.0, 1.0],
                                            constant: [0.2, 0.2, 0.2, 0.2],

                                            bailout: 5.0,
                                            max_steps: 150.0,
                                            fov: 1.5,
                                            iterations: 8,
                                            threshold: 0.001,
                                            power: 8.0,
                                            coloring: 0,
                                            fractal: i as i32,
                                        },
        });
    }

    let mut fractal_renderer = FractalRenderer::new(&pixels, window_size.width, window_size.height)?;

    let res = event_loop.run(|event, elwt| {
        // Draw the current frame
        if let Event::WindowEvent {
            event: WindowEvent::RedrawRequested,
            ..
        } = event
        {
            let render_result = pixels.render_with(|encoder, render_target, context| {
                let fractal_texture = fractal_renderer.texture_view();
                context.scaling_renderer.render(encoder, fractal_texture);

                fractal_renderer.update(&context.queue, time);

                delta_time = Instant::now().duration_since(last_frame).as_secs_f32();
                last_frame = Instant::now();
                time += delta_time;

                fractal_renderer.render(encoder, render_target, context.scaling_renderer.clip_rect());

                Ok(())
            });

            if let Err(err) = render_result {
                log_error("pixels.render_with", err);
                elwt.exit();
                return;
            }
        }

        if settings[fractal].camera_animation_enabled
        {
            settings[fractal].camera_animation_params.speed = settings[fractal].movement_speed;
            settings[fractal].fractal_settings.pos = animate_camera(&mut settings[fractal].camera_animation_params, delta_time);
        }

        if settings[fractal].fractal_animation_enabled
        {
            (settings[fractal].fractal_settings.power, settings[fractal].fractal_settings.constant) = (power_holder, constant_holder);
        }

        match option
        {
            'p' =>  if settings[fractal].fractal_animation_enabled {
                window.set_title(&format!("Fractal 3D: {}, Coloring: {}, power range: [{ :.2e}, { :.2e}]", FRACTALS[fractal], COLORINGS[settings[fractal].fractal_settings.coloring as usize], 
                    settings[fractal].fractal_settings.power - settings[fractal].fractal_animation_params.power_range,
                    settings[fractal].fractal_settings.power + settings[fractal].fractal_animation_params.power_range
                ));
            } else {
                window.set_title(&format!("Fractal 3D: {}, Coloring: {}, power: { :.2e}", FRACTALS[fractal], COLORINGS[settings[fractal].fractal_settings.coloring as usize], settings[fractal].fractal_settings.power));
            },
            '1' | '2' | '3' | '4' => if settings[fractal].fractal_animation_enabled {
                window.set_title(&format!("Fractal 3D: {}, Coloring: {}, constant ranges: ([{ :.2e}, { :.2e}]; [{ :.2e}, { :.2e}]; [{ :.2e}, { :.2e}]; [{ :.2e}, { :.2e}])", FRACTALS[fractal], COLORINGS[settings[fractal].fractal_settings.coloring as usize], 
                    settings[fractal].fractal_settings.constant[0] - settings[fractal].fractal_animation_params.constant_ranges[0],
                    settings[fractal].fractal_settings.constant[0] + settings[fractal].fractal_animation_params.constant_ranges[0],
                    settings[fractal].fractal_settings.constant[1] - settings[fractal].fractal_animation_params.constant_ranges[1],
                    settings[fractal].fractal_settings.constant[1] + settings[fractal].fractal_animation_params.constant_ranges[1],
                    settings[fractal].fractal_settings.constant[2] - settings[fractal].fractal_animation_params.constant_ranges[2],
                    settings[fractal].fractal_settings.constant[2] + settings[fractal].fractal_animation_params.constant_ranges[2],
                    settings[fractal].fractal_settings.constant[3] - settings[fractal].fractal_animation_params.constant_ranges[3],
                    settings[fractal].fractal_settings.constant[3] + settings[fractal].fractal_animation_params.constant_ranges[3],
                ));
            } else {
                window.set_title(&format!("Fractal 3D: {}, Coloring: {}, constant: ({ :.2e}, { :.2e}, { :.2e}, { :.2e})", FRACTALS[fractal], COLORINGS[settings[fractal].fractal_settings.coloring as usize], 
                    settings[fractal].fractal_settings.constant[0],
                    settings[fractal].fractal_settings.constant[1],
                    settings[fractal].fractal_settings.constant[2],
                    settings[fractal].fractal_settings.constant[3]
                ));
            },
            'k' => window.set_title(&format!("Fractal 3D: {}, Coloring: {}, movement_speed: {:.2e}", FRACTALS[fractal], COLORINGS[settings[fractal].fractal_settings.coloring as usize], settings[fractal].movement_speed)),
            'j' => window.set_title(&format!("Fractal 3D: {}, Coloring: {}, mouse_sensitivity: {:.2e}", FRACTALS[fractal], COLORINGS[settings[fractal].fractal_settings.coloring as usize], settings[fractal].mouse_sensitivity)),
            'f' => window.set_title(&format!("Fractal 3D: {}, Coloring: {}, fov: {:.2e}", FRACTALS[fractal], COLORINGS[settings[fractal].fractal_settings.coloring as usize], settings[fractal].fractal_settings.fov)),
            't' => window.set_title(&format!("Fractal 3D: {}, Coloring: {}, threshold: {:.2e}", FRACTALS[fractal], COLORINGS[settings[fractal].fractal_settings.coloring as usize], settings[fractal].fractal_settings.threshold)),
            'u' => window.set_title(&format!("Fractal 3D: {}, Coloring: {}, bailout: {:.2e}", FRACTALS[fractal], COLORINGS[settings[fractal].fractal_settings.coloring as usize], settings[fractal].fractal_settings.bailout)),
            'y' => window.set_title(&format!("Fractal 3D: {}, Coloring: {}, max_steps: {:.2e}", FRACTALS[fractal], COLORINGS[settings[fractal].fractal_settings.coloring as usize], settings[fractal].fractal_settings.max_steps)),
            'i' => window.set_title(&format!("Fractal 3D: {}, Coloring: {}, iterations: {}", FRACTALS[fractal], COLORINGS[settings[fractal].fractal_settings.coloring as usize], settings[fractal].fractal_settings.iterations)),
            _ => ()
        }

        if input.update(&event) {

            // Closing app
            if input.key_pressed(KeyCode::Escape) || input.close_requested() {
                elwt.exit();
                return;
            }

            // Movement and Commands
            if window_locked
            {
                // Switch mouse mode
                if input.key_pressed(KeyCode::Tab) 
                {
                    window.set_cursor_grab(CursorGrabMode::None).unwrap();
                    window.set_cursor_visible(true);
                    window_locked = false;
                }


                // Commands - settings
                if input.key_pressed(KeyCode::KeyK)     { option = 'k'; } // movement_speed
                if input.key_pressed(KeyCode::KeyJ)     { option = 'j'; } // mouse_sensitivity
                if input.key_pressed(KeyCode::KeyF)     { option = 'f'; } // fov
                if input.key_pressed(KeyCode::KeyY)     { option = 'y'; } // max_steps
                if input.key_pressed(KeyCode::KeyU)     { option = 'u'; } // bailout
                if input.key_pressed(KeyCode::KeyI)     { option = 'i'; } // iterations
                if input.key_pressed(KeyCode::KeyT)     { option = 't'; } // threshold
                if input.key_pressed(KeyCode::KeyP)     { option = 'p'; } // power
                if input.key_pressed(KeyCode::Digit1)   { option = '1'; } // constant change x
                if input.key_pressed(KeyCode::Digit2)   { option = '2'; } // constant change y
                if input.key_pressed(KeyCode::Digit3)   { option = '3'; } // constant change z
                if input.key_pressed(KeyCode::Digit4)   { option = '4'; } // constant change w

                // Special commands
                if input.key_pressed(KeyCode::KeyC) // coloring
                {
                    settings[fractal].fractal_settings.coloring = (settings[fractal].fractal_settings.coloring + 1) % colorings_total;

                }
                if input.key_pressed(KeyCode::KeyV) // fractal
                {
                    //settings[fractal].fractal_settings.fractal = (settings[fractal].fractal_settings.fractal + 1) % fractals_total;
                    fractal = (fractal + 1) % fractals_total as usize;
                }
                if input.key_pressed(KeyCode::KeyB) // logger
                {
                    write_log(settings[fractal].fractal_settings, settings[fractal].movement_speed, settings[fractal].mouse_sensitivity, time);
                }
                if input.key_pressed(KeyCode::Backspace) // reset
                {
                    settings[fractal] = Settings
                                        {
                                            movement_speed: 0.1,
                                            mouse_sensitivity: 0.002,
                                            sprint_enabled: false,
                                            camera_animation_enabled: false,
                                            camera_animation_params: CameraAnimationParams { radius: 0.0, height_amplitude: 0.0, speed: 0.0, azimuth: 0.0, phase: 0.0 },
                                            fractal_animation_enabled: false,
                                            fractal_animation_params: FractalAnimationParams { speed: 1.0, constant_ranges: [1.0, 1.0, 1.0, 1.0], power_range: 1.0 },
                                            fractal_settings: FractalSettings {
                                                                            pos: [-2.0, 0.0, 0.0],
                                                                            forward: [1.0, 0.0, 0.0],
                                                                            right: [0.0, 1.0, 0.0],
                                                                            up: [0.0, 0.0, 1.0],
                                                                            constant: [0.2, 0.2, 0.2, 0.2],

                                                                            bailout: 5.0,
                                                                            max_steps: 150.0,
                                                                            fov: 1.5,
                                                                            iterations: 8,
                                                                            threshold: 0.001,
                                                                            power: 8.0,
                                                                            coloring: 0,
                                                                            fractal: fractal as i32,
                                                                        },
                                        };
                }

                if input.key_pressed(KeyCode::KeyZ)
                {
                    if !settings[fractal].fractal_animation_enabled  
                    {
                        settings[fractal].fractal_animation_enabled = true;
                    }
                    else 
                    {
                        settings[fractal].fractal_animation_enabled = false;
                    }
                }

                if input.key_held(KeyCode::NumpadAdd)
                {
                    match option
                    {
                        'p' => settings[fractal].fractal_settings.power       += delta_time,
                        '1' => settings[fractal].fractal_settings.constant[0] += delta_time*0.1,
                        '2' => settings[fractal].fractal_settings.constant[1] += delta_time*0.1,
                        '3' => settings[fractal].fractal_settings.constant[2] += delta_time*0.1,
                        '4' => settings[fractal].fractal_settings.constant[3] += delta_time*0.1,
                        'k' => settings[fractal].movement_speed               *= 1.0 + 0.3*delta_time,
                        'j' => settings[fractal].mouse_sensitivity            *= 1.0 + 0.3*delta_time,
                        'f' => settings[fractal].fractal_settings.fov         *= 1.0 + 0.3*delta_time,
                        't' => settings[fractal].fractal_settings.threshold   *= 1.0 + 0.3*delta_time,
                        'u' => settings[fractal].fractal_settings.bailout     *= 1.0 + delta_time,
                        'y' => settings[fractal].fractal_settings.max_steps   *= 1.0 + delta_time,
                        _ => ()
                    }
                }

                if input.key_held(KeyCode::NumpadSubtract)
                {
                    match option
                    {
                        'p' => settings[fractal].fractal_settings.power       -= delta_time,
                        '1' => settings[fractal].fractal_settings.constant[0] -= delta_time*0.1,
                        '2' => settings[fractal].fractal_settings.constant[1] -= delta_time*0.1,
                        '3' => settings[fractal].fractal_settings.constant[2] -= delta_time*0.1,
                        '4' => settings[fractal].fractal_settings.constant[3] -= delta_time*0.1,
                        'k' => settings[fractal].movement_speed               /= 1.0 + 0.3*delta_time,
                        'j' => settings[fractal].mouse_sensitivity            /= 1.0 + 0.3*delta_time,
                        'f' => settings[fractal].fractal_settings.fov         /= 1.0 + 0.3*delta_time,
                        't' => settings[fractal].fractal_settings.threshold   /= 1.0 + 0.3*delta_time,
                        'u' => settings[fractal].fractal_settings.bailout     /= 1.0 + delta_time,
                        'y' => settings[fractal].fractal_settings.max_steps   /= 1.0 + delta_time,
                        _ => ()
                    }
                }

                if input.key_pressed(KeyCode::NumpadAdd) && option == 'i'
                {
                    settings[fractal].fractal_settings.iterations  += 1;
                }  

                if input.key_pressed(KeyCode::NumpadSubtract) && option == 'i'
                {
                    settings[fractal].fractal_settings.iterations  -= 1;
                }
                
                // Fractal animation commands
                if settings[fractal].fractal_animation_enabled
                {
                    if input.key_held(KeyCode::Comma)
                    {
                        settings[fractal].fractal_animation_params.speed *= 1.0 + delta_time*0.25;
                    }
                    if input.key_held(KeyCode::Period)
                    {
                        settings[fractal].fractal_animation_params.speed /= 1.0 + delta_time*0.25;
                    }
                    if input.key_held(KeyCode::BracketLeft)
                    {
                        match option
                        {
                            'p' => settings[fractal].fractal_animation_params.power_range = (settings[fractal].fractal_animation_params.power_range - delta_time*0.15).max(0.0),
                            '1' => settings[fractal].fractal_animation_params.constant_ranges[0] = (settings[fractal].fractal_animation_params.constant_ranges[0] - delta_time*0.15).max(0.0),
                            '2' => settings[fractal].fractal_animation_params.constant_ranges[1] = (settings[fractal].fractal_animation_params.constant_ranges[1] - delta_time*0.15).max(0.0),
                            '3' => settings[fractal].fractal_animation_params.constant_ranges[2] = (settings[fractal].fractal_animation_params.constant_ranges[2] - delta_time*0.15).max(0.0),
                            '4' => settings[fractal].fractal_animation_params.constant_ranges[3] = (settings[fractal].fractal_animation_params.constant_ranges[3] - delta_time*0.15).max(0.0),
                            _ => ()
                        }
                    }
                    if input.key_held(KeyCode::BracketRight)
                    {
                        match option
                        {
                            'p' => settings[fractal].fractal_animation_params.power_range += delta_time*0.15,
                            '1' => settings[fractal].fractal_animation_params.constant_ranges[0] += delta_time*0.15,
                            '2' => settings[fractal].fractal_animation_params.constant_ranges[1] += delta_time*0.15,
                            '3' => settings[fractal].fractal_animation_params.constant_ranges[2] += delta_time*0.15,
                            '4' => settings[fractal].fractal_animation_params.constant_ranges[3] += delta_time*0.15,
                            _ => ()
                        }
                    }
                }

                // Camera Movement
                if settings[fractal].camera_animation_enabled
                {
                    let pos = settings[fractal].fractal_settings.pos;

                    let forward = normalize([
                        -pos[0],
                        -pos[1],
                        -pos[2],
                    ]);

                    let mut world_up = [0.0, 1.0, 0.0];
                    if forward[0].abs() < 1e-6 && forward[1].abs() < 1e-6 {
                        world_up = [0.0, 0.0, 1.0];
                    }

                    let right = normalize(cross(forward, world_up));

                    let up = normalize(cross(right, forward));

                    settings[fractal].fractal_settings.forward = forward;
                    settings[fractal].fractal_settings.right   = right;
                    settings[fractal].fractal_settings.up      = up;


                    if input.key_held(KeyCode::KeyW)
                    {
                        settings[fractal].camera_animation_params.radius -= settings[fractal].movement_speed * delta_time;
                    }
                    if input.key_held(KeyCode::KeyS)
                    {
                        settings[fractal].camera_animation_params.radius += settings[fractal].movement_speed * delta_time;
                    }
                    if input.key_held(KeyCode::KeyA)
                    {
                        settings[fractal].camera_animation_params.azimuth -= settings[fractal].movement_speed * delta_time;
                    }
                    if input.key_held(KeyCode::KeyD)
                    {
                        settings[fractal].camera_animation_params.azimuth += settings[fractal].movement_speed * delta_time;
                    }
                    if input.key_held(KeyCode::Space)
                    {
                        settings[fractal].camera_animation_params.height_amplitude += settings[fractal].movement_speed * delta_time;
                    }
                    if input.key_held(KeyCode::ControlLeft)
                    {
                        settings[fractal].camera_animation_params.height_amplitude -= settings[fractal].movement_speed * delta_time;
                    }

                }
                else 
                {
                    let (dy, dx) = input.mouse_diff();

                    let yaw   = -dx as f32 * settings[fractal].mouse_sensitivity;
                    let pitch = dy as f32 * settings[fractal].mouse_sensitivity;

                    let world_up = [0.0, 1.0, 0.0];

                    let mut forward = settings[fractal].fractal_settings.forward;

                    let cos_y = yaw.cos();
                    let sin_y = yaw.sin();

                    forward = normalize([
                        forward[0] * cos_y + world_up[0] * sin_y,
                        forward[1] * cos_y + world_up[1] * sin_y,
                        forward[2] * cos_y + world_up[2] * sin_y,
                    ]);

                    let right = normalize(cross(forward, world_up));

                    let cos_p = pitch.cos();
                    let sin_p = pitch.sin();

                    forward = normalize([
                        forward[0] * cos_p + right[0] * sin_p,
                        forward[1] * cos_p + right[1] * sin_p,
                        forward[2] * cos_p + right[2] * sin_p,
                    ]);

                    let up = normalize(cross(right, forward));

                    settings[fractal].fractal_settings.forward = forward;
                    settings[fractal].fractal_settings.right   = right;
                    settings[fractal].fractal_settings.up      = up;


                    if input.key_held(KeyCode::KeyW)
                    {
                        for i in 0..3
                        {
                            settings[fractal].fractal_settings.pos[i] += delta_time * settings[fractal].movement_speed * settings[fractal].fractal_settings.forward[i];
                        }
                    }
                    if input.key_held(KeyCode::KeyS)
                    {
                        for i in 0..3
                        {
                            settings[fractal].fractal_settings.pos[i] -= delta_time * settings[fractal].movement_speed * settings[fractal].fractal_settings.forward[i];
                        }
                    }
                    if input.key_held(KeyCode::KeyD)
                    {
                        for i in 0..3
                        {
                            settings[fractal].fractal_settings.pos[i] += delta_time * settings[fractal].movement_speed * settings[fractal].fractal_settings.right[i];
                        }
                    }
                    if input.key_held(KeyCode::KeyA)
                    {
                        for i in 0..3
                        {
                            settings[fractal].fractal_settings.pos[i] -= delta_time * settings[fractal].movement_speed * settings[fractal].fractal_settings.right[i];
                        }
                    }
                    if input.key_held(KeyCode::Space)
                    {
                        for i in 0..3
                        {
                            settings[fractal].fractal_settings.pos[i] += delta_time * settings[fractal].movement_speed * world_up[i];
                        }
                    }
                    if input.key_held(KeyCode::ControlLeft)
                    {
                        for i in 0..3
                        {
                            settings[fractal].fractal_settings.pos[i] -= delta_time * settings[fractal].movement_speed * world_up[i];
                        }
                    }
                }

                // Sprint
                if input.key_pressed(KeyCode::ShiftLeft)
                {
                    if settings[fractal].sprint_enabled
                    {
                        settings[fractal].sprint_enabled = false;
                        settings[fractal].movement_speed /= 3.0;
                    }
                    else 
                    {
                        settings[fractal].sprint_enabled = true;
                        settings[fractal].movement_speed *= 3.0;
                    }
                }

            }
            else if input.key_pressed(KeyCode::Tab) // Switch mouse mode
            {
                window.set_cursor_grab(CursorGrabMode::Confined).unwrap();
                window.set_cursor_visible(false);
                window_locked = true;
            }

            // Resize the window
            if let Some(size) = input.window_resized() {
                if let Err(err) = pixels.resize_surface(size.width, size.height) {
                    log_error("pixels.resize_surface", err);
                    elwt.exit();
                    return;
                }
                if let Err(err) = fractal_renderer.resize(&pixels, size.width, size.height) {
                    log_error("fractal_renderer.resize", err);
                    elwt.exit();
                    return;
                }
            }

            if settings[fractal].fractal_animation_enabled
            {
                (power_holder, constant_holder) = (settings[fractal].fractal_settings.power, settings[fractal].fractal_settings.constant);
                (settings[fractal].fractal_settings.power, settings[fractal].fractal_settings.constant) = 
                animate_fractal(
                    &mut animation_time,
                    delta_time,
                    settings[fractal].fractal_animation_params.speed * 10.0, 
                    settings[fractal].fractal_settings.power, 
                    settings[fractal].fractal_animation_params.power_range, 
                    settings[fractal].fractal_settings.constant, 
                    settings[fractal].fractal_animation_params.constant_ranges
                )
            }

            // Special commands - animations
            if input.key_pressed(KeyCode::KeyX)
            {
                if settings[fractal].camera_animation_enabled
                {
                    settings[fractal].camera_animation_enabled = false;
                }
                else 
                {
                    settings[fractal].camera_animation_enabled = true;
                    start_camera_animation(settings[fractal].fractal_settings.pos, &mut settings[fractal].camera_animation_params);
                }
            }

            // Update internal state and request a redraw
            fractal_renderer.update_settings(&pixels.queue(), &settings[fractal].fractal_settings);
            window.request_redraw();
        }
    });
    res.map_err(|e| Error::UserDefined(Box::new(e)))
}

fn log_error<E: std::error::Error + 'static>(method_name: &str, err: E) {
    error!("{method_name}() failed: {err}");
    for source in err.sources().skip(1) {
        error!("  Caused by: {source}");
    }
}

// Markdown preview
// ctrl + shift + v
