# Fractal 3D Viewer

Interactive 3D fractal viewer with full control over camera movement, fractal parameters, and animations.

![Fractal Animation Gif](assets/juliabulb.gif)

## Features
- Real-time 3D fractal rendering
- Camera rotation and zoom
- Adjustable fractal parameters
- Keyboard and mouse controls

## How to build & run
1. Make sure you have Rust installed: https://www.rust-lang.org/tools/install
2. Clone the repository:
   ```bash
   git clone https://github.com/for5en/fractal_3d_viewer.git
   cd fractal_3d_viewer
   ```
3. Build the project:
   ```bash
   cargo build --release
   ```
4. Run the viewer:
   ```bash
   cargo run --release
   ```

## Movement

| Key | Action |
|-----|--------|
| **W** | Move forward |
| **A** | Move left |
| **S** | Move backward |
| **D** | Move right |
| **Space** | Move up |
| **Left Ctrl** | Move down |
| **Shift** | Sprint *(toggle)* |


## Special Commands

| Key | Action |
|-----|--------|
| **C** | Change coloring |
| **V** | Change fractal |
| **B** | Write log (description of all parameters) |
| **X** | Animate camera |
| **Z** | Animate power and constant |
| **Tab** | Change mouse mode (free / locked) |
| **Backspace** | Reset |

## Option Commands

### Value Control

| Key | Action |
|-----|--------|
| **Numpad +** | Increase value |
| **Numpad −** | Decrease value |


### Camera Settings

| Key | Parameter |
|-----|-----------|
| **K** | Movement speed |
| **J** | Mouse sensitivity |
| **F** | Field of view (FOV) |


### Fractal Settings

| Key | Parameter |
|-----|-----------|
| **I** | Fractal iterations (single click) |
| **P** | Fractal equation power |
| **Y** | Raymarching maximum steps |
| **T** | Raymarching hit threshold |
| **U** | Raymarching bailout |
| **1** | Constant X value |
| **2** | Constant Y value |
| **3** | Constant Z value |
| **4** | Constant W value |


## Camera Animation

| Key | Action |
|-----|--------|
| **W / A / S / D / Space / Ctrl** | Change trajectory |


## Fractal Animation

### Parameter Selection

| Key | Action |
|-----|--------|
| **P** | Power settings |
| **1 / 2 / 3 / 4** | Constant settings |

### Animation Control

| Key | Action |
|-----|--------|
| **Numpad +** | Increase pivot |
| **Numpad −** | Decrease pivot |
| **[** | Decrease range size |
| **]** | Increase range size |
| **,** | Decrease speed |
| **.** | Increase speed |

