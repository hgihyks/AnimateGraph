import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter, PillowWriter
from sympy import symbols, lambdify
import sympy as sp
import shutil
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from dataclasses import dataclass


@dataclass
class GraphConfig:
    equation: str | tuple[str, str, str]  # Single equation for 2D or tuple of (x,y,z) for 3D
    mode: str = '2d'  # '2d' or '3d'
    x_range: tuple[float, float] = (-10, 10)
    y_range: tuple[float, float] | None = None
    z_range: tuple[float, float] | None = None

    # Animation settings
    frames: int = 100
    interval: int = 50
    speed_zones: list[tuple[float, float, float]] = None  # [(x_start, x_end, speed), ...]

    # Camera settings (for 3D)
    camera_rotation: bool = False  # Whether to rotate camera
    elevation_angle: float = 0  # Vertical viewing angle in degrees
    initial_azimuth: float = 45  # Initial horizontal viewing angle in degrees
    rotation_speed: float = 1  # Full rotations per animation

    # Visual settings
    background_color: str = 'black'
    graph_color: str = 'white'
    show_grid: bool = True

    # Output settings
    save_path: str | None = None

    def __post_init__(self):
        if self.speed_zones is None:
            # Default speed zone covering the entire x_range with speed 1
            self.speed_zones = [(self.x_range[0], self.x_range[1], 1)]


def map_frame_to_index(frame, total_frames, x_vals, speed_zones):
    """
    Maps frame number to index in x_vals based on multiple speed zones.
    speed_zones: list of tuples [(x_start, x_end, speed), ...]
    Higher speed value means faster animation (fewer frames per unit).
    """
    x_start, x_end = x_vals[0], x_vals[-1]

    # Fill in any gaps in speed_zones
    full_zones = []
    current = x_start
    speed_zones = sorted(speed_zones, key=lambda z: z[0])
    for start, end, speed in speed_zones:
        if start > current:
            full_zones.append((current, start, 1))  # default speed
        full_zones.append((start, end, speed))
        current = end
    if current < x_end:
        full_zones.append((current, x_end, 1))

    # Compute total weighted length
    total_weight = sum((end - start) / speed for start, end, speed in full_zones)

    # Progress in total weight units
    progress = frame / total_frames * total_weight

    # Find which zone the current progress falls in
    dist = 0
    for start, end, speed in full_zones:
        segment_weight = (end - start) / speed
        if dist + segment_weight >= progress:
            dist_in_segment = progress - dist
            x_current = start + dist_in_segment * speed
            break
        dist += segment_weight
    else:
        x_current = x_end

    idx = np.searchsorted(x_vals, x_current)
    return idx

def animate_equation(config: GraphConfig):
    """
    Animate the graph of the given equation with variable speed based on config.
    For 3D, provide a parametric equation in terms of t, e.g., (x(t), y(t), z(t))
    """
    if config.mode == '3d':
        t = symbols('t')
        if not isinstance(config.equation, (tuple, list)) or len(config.equation) != 3:
            raise ValueError("For 3D mode, equation must be a tuple or list of 3 expressions (x(t), y(t), z(t))")

        x_expr = sp.sympify(config.equation[0])
        y_expr = sp.sympify(config.equation[1])
        z_expr = sp.sympify(config.equation[2])

        fx = lambdify(t, x_expr, modules=['numpy'])
        fy = lambdify(t, y_expr, modules=['numpy'])
        fz = lambdify(t, z_expr, modules=['numpy'])

        t_vals = np.linspace(config.x_range[0], config.x_range[1], 1000)
        x_vals = fx(t_vals)
        y_vals = fy(t_vals)
        z_vals = fz(t_vals)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor(config.background_color)
        fig.patch.set_facecolor(config.background_color)
        
        if config.show_grid:
            ax.grid(True)
        else:
            ax.grid(False)
            ax.set_axis_off()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            
        # Set initial camera angle
        ax.view_init(elev=config.elevation_angle, azim=config.initial_azimuth)
        
        ax.set_xlim(np.min(x_vals), np.max(x_vals))
        ax.set_ylim(np.min(y_vals), np.max(y_vals))
        ax.set_zlim(np.min(z_vals), np.max(z_vals))
        line, = ax.plot([], [], [], lw=2, color=config.graph_color)

        def update(frame):
            end = map_frame_to_index(frame, config.frames, t_vals, config.speed_zones)
            line.set_data(x_vals[:end], y_vals[:end])
            line.set_3d_properties(z_vals[:end])
            
            if config.camera_rotation:
                # Calculate rotation angle based on frame
                angle = config.initial_azimuth + (360 * config.rotation_speed * frame / config.frames)
                ax.view_init(elev=config.elevation_angle, azim=angle)
                
            return line,

    else:
        x = symbols('x')
        expr = sp.sympify(config.equation)
        func = lambdify(x, expr, modules=['numpy'])

        x_vals = np.linspace(config.x_range[0], config.x_range[1], 1000)
        y_vals = func(x_vals)

        fig, ax = plt.subplots()
        ax.set_facecolor(config.background_color)
        fig.patch.set_facecolor(config.background_color)
        if config.show_grid:
            ax.grid(True)
        else:
            ax.grid(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set_xlim(config.x_range)
        buffer = (max(y_vals) - min(y_vals) * 0.1) if max(y_vals) != min(y_vals) else 1
        ax.set_ylim(min(y_vals) - buffer, max(y_vals) + buffer)
        line, = ax.plot([], [], lw=2, color=config.graph_color)

        def update(frame):
            end = map_frame_to_index(frame, config.frames, x_vals, config.speed_zones)
            line.set_data(x_vals[:end], y_vals[:end])
            return line,

    ani = animation.FuncAnimation(fig, update, frames=config.frames, interval=config.interval, blit=True)

    if config.save_path:
        if config.save_path.endswith(".mp4") and shutil.which("ffmpeg"):
            writer = FFMpegWriter(fps=20, metadata=dict(artist='Graph Animator'), bitrate=1800)
        else:
            print("⚠️ FFmpeg not found or file extension unsupported. Falling back to GIF using Pillow.")
            writer = PillowWriter(fps=20)
        ani.save(config.save_path, writer=writer)
        print(f"Animation saved to {config.save_path}")
    else:
        plt.show()
        return ani

# Example usage:
config_2d = GraphConfig(
    equation="sin(x)",
    mode='2d',
    speed_zones=[(-5, -1, 5), (0, 2, 1)]
)
#animate_equation(config_2d)

# Example usage with camera rotation
config_3d = GraphConfig(
    equation=("sin(t)", "cos(t)", "tan(t)"),
    mode='3d',
    x_range=(0, 12*np.pi),
    speed_zones=[(0, 2, 1), (2, 4, 5)],
    show_grid=False,
    camera_rotation=True,
    elevation_angle=20,
    initial_azimuth=45,
    rotation_speed=2  # Make two full rotations during the animation
)
animate_equation(config_3d)
