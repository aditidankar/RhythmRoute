import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
import argparse
import os


def resample_3d_line(points_3d, num_samples):
    """Resamples a 3D path to have a specific number of points."""
    if len(points_3d) < 2:
        return np.array([points_3d[0]] * num_samples) if len(points_3d) > 0 else np.empty((0, 3))
    
    distances = np.sqrt(np.sum(np.diff(points_3d, axis=0)**2, axis=1))
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

    if cumulative_distances[-1] == 0:
        return np.array([points_3d[0]] * num_samples)

    sample_distances = np.linspace(0, cumulative_distances[-1], num_samples)
    
    sample_points = np.empty((num_samples, 3))
    sample_points[:, 0] = np.interp(sample_distances, cumulative_distances, points_3d[:, 0])
    sample_points[:, 1] = np.interp(sample_distances, cumulative_distances, points_3d[:, 1])
    sample_points[:, 2] = np.interp(sample_distances, cumulative_distances, points_3d[:, 2])

    return sample_points


def draw_semicircle(start_coord, radius, steps=150):
    x0, y0, z0 = start_coord
    center_x, center_y = x0, y0
    start = 0.2
    end = 0.8
    angles = np.linspace(start * np.pi, end * np.pi, steps)
    x = (center_x + radius) - radius * np.cos(angles)
    y = center_y - radius * np.sin(angles)
    z = np.full_like(x, z0)
    return x, y, z


def draw_T_shape(start_coord, bar_length, bar_height, steps=150):
    x0, y0, z0 = start_coord
    vertex1 = np.array([x0, y0, z0])
    vertex2 = np.array([x0, y0 + bar_length, z0])
    vertex3 = np.array([x0 + bar_height, y0 + bar_length / 2, z0])
    
    steps_per_section = steps // 3
    
    x_values_horizontal = np.linspace(vertex1[0], vertex2[0], steps_per_section)
    y_values_horizontal = np.linspace(vertex1[1], vertex2[1], steps_per_section)
    
    x_values_vertical = np.linspace(vertex2[0], vertex3[0], steps - steps_per_section)
    y_values_vertical = np.full_like(x_values_vertical, y0 + bar_length / 2)
    
    x_values = np.concatenate([x_values_horizontal, x_values_vertical])
    y_values = np.concatenate([y_values_horizontal, y_values_vertical])
    z_values = np.full_like(y_values, z0)
    
    return x_values, y_values, z_values


def draw_equilateral_triangle(start_coord, side_length, steps=150):
    x0, y0, z0 = start_coord
    vertex1 = np.array([x0, y0, z0])
    vertex2 = np.array([x0 + side_length, y0, z0])
    height = side_length * np.sqrt(3) / 2
    vertex3 = np.array([x0 + side_length / 2, y0 + height, z0])
    
    steps_per_side = steps // 3
    
    x_values = np.concatenate([
        np.linspace(vertex1[0], vertex2[0], steps_per_side),
        np.linspace(vertex2[0], vertex3[0], steps_per_side),
        np.linspace(vertex3[0], vertex1[0], steps - 2 * steps_per_side)
    ])
    y_values = np.concatenate([
        np.linspace(vertex1[1], vertex2[1], steps_per_side),
        np.linspace(vertex2[1], vertex3[1], steps_per_side),
        np.linspace(vertex3[1], vertex1[1], steps - 2 * steps_per_side)
    ])
    z_values = np.full_like(x_values, z0)
    
    return x_values, y_values, z_values


def draw_square(start_coord, side_length, steps=152):
    x0, y0, z0 = start_coord
    vertex1 = np.array([x0, y0, z0])
    vertex2 = np.array([x0 + side_length, y0, z0])
    vertex3 = np.array([x0 + side_length, y0 + side_length, z0])
    vertex4 = np.array([x0, y0 + side_length, z0])
    
    steps_per_side = steps // 4
    
    x_values = np.concatenate([
        np.linspace(vertex1[0], vertex2[0], steps_per_side),
        np.linspace(vertex2[0], vertex3[0], steps_per_side),
        np.linspace(vertex3[0], vertex4[0], steps_per_side),
        np.linspace(vertex4[0], vertex1[0], steps - 3 * steps_per_side)
    ])
    y_values = np.concatenate([
        np.linspace(vertex1[1], vertex2[1], steps_per_side),
        np.linspace(vertex2[1], vertex3[1], steps_per_side),
        np.linspace(vertex3[1], vertex4[1], steps_per_side),
        np.linspace(vertex4[1], vertex1[1], steps - 3 * steps_per_side)
    ])
    z_values = np.full_like(x_values, z0)
    
    return x_values, y_values, z_values


def rotate_numpy(points, angle_degrees=10):
    angle_radians = np.radians(angle_degrees)
    # Z-up rotation matrix (rotates around Z-axis)
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians), 0],
        [np.sin(angle_radians), np.cos(angle_radians), 0],
        [0, 0, 1]
    ])
    return points @ rotation_matrix.T


def draw_up_down_line(start_z=.0, radius=0.1, theta_step=20, steps=150):
    theta = np.linspace(0, 2 * np.pi * steps / theta_step, steps)
    z_values = start_z + radius * np.sin(theta)
    x_values = np.zeros_like(z_values)
    y_values = np.zeros_like(z_values)
    return x_values, y_values, z_values


def draw_straight_line(start_coord, step_length, steps=150):
    x0, y0, z0 = start_coord
    x_values = np.linspace(x0, x0 + step_length * (steps-1), steps)
    y_values = np.full_like(x_values, y0)
    z_values = np.full_like(x_values, z0)
    return x_values, y_values, z_values


def draw_curve_line(start_coord, radius, step_length, theta_step=50, steps=150):
    x0, y0, z0 = start_coord
    x_values = np.linspace(x0, x0 + step_length * (steps-1), steps)
    theta = np.linspace(0, 2 * np.pi * steps / theta_step, steps)
    y_values = y0 + radius * np.sin(theta)
    z_values = np.full_like(x_values, z0)
    return x_values, y_values, z_values


def draw_curve_line2(start_coord, radius, step_length, theta_step=50, steps=150):
    x0, y0, z0 = start_coord
    x_values = np.linspace(x0, x0 + step_length * (steps-1), steps)
    theta = np.linspace(0, 2 * np.pi * steps / theta_step, steps)
    y_values = y0 + radius * np.cos(theta)
    z_values = np.full_like(x_values, z0)
    return x_values, y_values, z_values


def draw_circle(start_coord, radius, steps=150):
    x0, y0, z0 = start_coord
    center_x, center_y = x0, y0
    angles = np.linspace(0, 2 * np.pi, steps)
    x = (center_x - radius) + radius * np.cos(angles)
    y = center_y + radius * np.sin(angles)
    z = np.full_like(x, z0)
    return x, y, z


def draw_ellipse(start_coord, a, b, steps=150):
    x0, y0, z0 = start_coord
    center_x = x0 - a
    center_y = y0 + b
    angles = np.linspace(0, 2 * np.pi, steps)
    x = center_x + a * np.cos(angles)
    y = center_y - b + b * np.sin(angles)
    z = np.full_like(x, z0)
    return x, y, z


def draw_spiral(start_coord, radius, step_length, steps=150):
    x0, y0, z0 = start_coord
    z_values = np.linspace(z0, z0 + step_length * (steps-1), steps) # Height increases with steps
    theta = np.linspace(0, 2 * np.pi * steps / 20, steps)
    x_values = x0 + radius * np.cos(theta)
    y_values = y0 + radius * np.sin(theta)
    return x_values, y_values, z_values


def save_and_plot_trajectory(points_3d, out_filename, do_plot=False):
    """Saves the trajectory and optionally plots it."""
    output_dir = os.path.dirname(out_filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    np.save(out_filename, points_3d)
    print(f"Saved trajectory with {len(points_3d)} points to {out_filename}")

    if do_plot:
        plot_trajectory(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])


def plot_trajectory(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (Height)')
    # Make aspect ratio equal
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate and save a 3D trajectory from a predefined shape.")
    parser.add_argument("--shape", type=str, required=True, 
                        choices=['semicircle', 't_shape', 'triangle', 'square', 'line', 'curve', 'curve2', 'circle', 'ellipse', 'spiral', 'up_down_line'],
                        help="The shape of the trajectory to generate.")
    parser.add_argument("--samples", type=int, default=150, help="Number of points to sample for the final trajectory.")
    parser.add_argument("--out", type=str, default="user_trajectory.npy", help="Output file name (e.g., data/trajectories/my_trajectory.npy).")
    parser.add_argument("--plot", action='store_true', help="Display a plot of the generated trajectory.")

    # Shape-specific arguments
    parser.add_argument("--start_coord", type=float, nargs=3, default=[0., 0., 0.], help="Start coordinate (x, y, z).")
    parser.add_argument("--radius", type=float, default=0.5, help="Radius for shapes like circle, semicircle, spiral.")
    parser.add_argument("--side_length", type=float, default=1.0, help="Side length for square or triangle.")
    parser.add_argument("--bar_length", type=float, default=1.0, help="Length of the horizontal bar in T-shape.")
    parser.add_argument("--bar_height", type=float, default=1.0, help="Height of the vertical bar in T-shape.")
    parser.add_argument("--a", type=float, default=1.0, help="Major axis 'a' for ellipse.")
    parser.add_argument("--b", type=float, default=0.5, help="Minor axis 'b' for ellipse.")
    parser.add_argument("--step_length", type=float, default=0.01, help="Step length for line, spiral, or curves.")

    args = parser.parse_args()
    
    # Generate the initial points
    if args.shape == 'circle':
        x, y, z = draw_circle(args.start_coord, args.radius, args.samples)
    elif args.shape == 'semicircle':
        x, y, z = draw_semicircle(args.start_coord, args.radius, args.samples)
    elif args.shape == 'square':
        x, y, z = draw_square(args.start_coord, args.side_length, args.samples)
    elif args.shape == 'triangle':
        x, y, z = draw_equilateral_triangle(args.start_coord, args.side_length, args.samples)
    elif args.shape == 't_shape':
        x, y, z = draw_T_shape(args.start_coord, args.bar_length, args.bar_height, args.samples)
    elif args.shape == 'ellipse':
        x, y, z = draw_ellipse(args.start_coord, args.a, args.b, args.samples)
    elif args.shape == 'spiral':
        x, y, z = draw_spiral(args.start_coord, args.radius, args.step_length, args.samples)
    elif args.shape == 'line':
        x, y, z = draw_straight_line(args.start_coord, args.step_length, args.samples)
    elif args.shape == 'curve':
        x, y, z = draw_curve_line(args.start_coord, args.radius, args.step_length, steps=args.samples)
    elif args.shape == 'curve2':
        x, y, z = draw_curve_line2(args.start_coord, args.radius, args.step_length, steps=args.samples)
    elif args.shape == 'up_down_line':
        x, y, z = draw_up_down_line(start_z=args.start_coord[2], radius=args.radius, steps=args.samples)


    # Combine into a single array and resample to ensure correct length
    points_3d = np.stack([x, y, z], axis=1)
    resampled_points = resample_3d_line(points_3d, args.samples)

    # Normalize to be in a [-1, 1] cube for consistency with the model's potential training data
    max_range = (resampled_points.max(axis=0) - resampled_points.min(axis=0)).max()
    if max_range > 1e-6: # Avoid division by zero for single points
        resampled_points = (resampled_points - resampled_points.mean(axis=0)) / max_range

    save_and_plot_trajectory(resampled_points, args.out, args.plot)