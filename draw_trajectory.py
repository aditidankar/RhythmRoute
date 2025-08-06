import numpy as np
import tkinter as tk
from scipy.interpolate import interp1d
import argparse
import os

def interpolate_line(points, num_samples):
    if len(points) < 2:
        return np.array(points)
    
    # Calculate the distance between consecutive points
    distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

    if cumulative_distances[-1] == 0:
        return np.array([points[0]] * num_samples)

    sample_distances = np.linspace(0, cumulative_distances[-1], num_samples)
    
    sample_points = np.empty((num_samples, 2))
    sample_points[:, 0] = np.interp(sample_distances, cumulative_distances, [p[0] for p in points])
    sample_points[:, 1] = np.interp(sample_distances, cumulative_distances, [p[1] for p in points])
    
    return sample_points

class LineSamplerApp:
    def __init__(self, root, num_samples=150, out_filename="user_trajectory.npy"):
        self.root = root
        self.out_filename = out_filename
        self.canvas = tk.Canvas(root, width=400, height=400, bg="white")
        self.canvas.pack()
        self.drawing_points = []
        self.lines = []
        self.current_line = []
        self.height_control_points = [(50, 350), (175, 350), (225, 350), (350, 350)]
        self.dragging_point_index = None
        self.drawing = True
        self.num_samples = num_samples

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        adjust_height_button = tk.Button(root, text="Adjust Height", command=self.enter_height_adjust_mode)
        adjust_height_button.pack()
        
        save_button = tk.Button(root, text="Save 3D Points", command=self.save_points)
        save_button.pack()

        self.draw_height_control_points()

    def on_click(self, event):
        if self.drawing:
            self.current_line.append((event.x, event.y))
        else:
            self.on_control_point_click(event)

    def on_drag(self, event):
        if self.drawing and self.current_line:
            self.current_line.append((event.x, event.y))
            self.canvas.create_line(self.current_line[-2], self.current_line[-1], fill="black")
        elif not self.drawing and self.dragging_point_index is not None:
            self.on_control_point_drag(event)

    def on_release(self, event):
        if self.drawing and self.current_line:
            self.lines.append(self.current_line)
            self.current_line = []
        elif not self.drawing:
            self.on_control_point_release(event)

    def enter_height_adjust_mode(self):
        self.drawing = False
        self.canvas.bind("<Button-1>", self.on_control_point_click)
        self.canvas.bind("<B1-Motion>", self.on_control_point_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_control_point_release)

    def on_control_point_click(self, event):
        for i, point in enumerate(self.height_control_points):
            if (point[0] - 5 <= event.x <= point[0] + 5) and (point[1] - 5 <= event.y <= point[1] + 5):
                self.dragging_point_index = i
                return

    def on_control_point_drag(self, event):
        if self.dragging_point_index is not None:
            self.height_control_points[self.dragging_point_index] = (event.x, event.y)
            self.draw_height_control_points()

    def on_control_point_release(self, event):
        self.dragging_point_index = None

    def draw_height_control_points(self):
        self.canvas.delete("height_control")
        for point in self.height_control_points:
            self.canvas.create_oval(point[0]-5, point[1]-5, point[0]+5, point[1]+5, fill="red", tags="height_control")
        self.redraw_height_curve()

    def redraw_height_curve(self):
        self.canvas.delete("height_curve")
        if len(self.height_control_points) >= 2:
            x, y = zip(*sorted(self.height_control_points))
            if len(self.height_control_points) < 4:
                kind = 'linear'
            else:
                kind = 'cubic'
            curve = interp1d(x, y, kind=kind, fill_value="extrapolate")
            xs = np.linspace(min(x), max(x), 100)
            ys = curve(xs)
            for i in range(len(xs) - 1):
                self.canvas.create_line(xs[i], ys[i], xs[i+1], ys[i+1], fill="red", tags="height_curve")

    def save_points(self):
        if not self.lines:
            print("No lines drawn to save.")
            return

        all_points_2d = []
        total_length = sum(np.sum(np.sqrt(np.sum(np.diff(line, axis=0)**2, axis=1))) for line in self.lines)
        
        for line in self.lines:
            line = np.array(line)
            line_length = np.sum(np.sqrt(np.sum(np.diff(line, axis=0)**2, axis=1)))
            num_line_samples = int(np.round((line_length / total_length) * self.num_samples))
            if num_line_samples > 0:
                all_points_2d.extend(interpolate_line(line, num_line_samples))

        if not all_points_2d:
            print("Could not generate any points.")
            return

        all_points_2d = np.array(all_points_2d)
        
        # Ensure we have the exact number of samples
        if len(all_points_2d) != self.num_samples:
             all_points_2d = interpolate_line(all_points_2d, self.num_samples)


        xs_2d = all_points_2d[:, 0]
        ys_2d = all_points_2d[:, 1]
        
        if len(self.height_control_points) >= 2:
            x_h, y_h = zip(*sorted(self.height_control_points))
            if len(self.height_control_points) < 4:
                kind = 'linear'
            else:
                kind = 'cubic'
            curve = interp1d(x_h, y_h, kind=kind, fill_value="extrapolate")
            zs = curve(xs_2d)
        else:
            zs = np.zeros(len(xs_2d))

        # The model expects a Z-up coordinate system.
        # drawn x -> model x
        # drawn y -> model y
        # drawn z (height) -> model z
        # Also normalize it to be roughly in -1 to 1 range, assuming canvas is 400x400
        samples_3d = np.stack([
            (xs_2d - 200) / 200,
            (ys_2d - 200) / 200,
            (zs - 200) / 200
        ], axis=1).astype(np.float32)
        
        output_dir = os.path.dirname(self.out_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        np.save(self.out_filename, samples_3d)
        print(f"Saved {len(samples_3d)} 3D points to {self.out_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw a 3D trajectory and save it.")
    parser.add_argument("--samples", type=int, default=150, help="Number of points to sample along the trajectory.")
    parser.add_argument("--out", type=str, default="user_trajectory.npy", help="Output file name (e.g., data/trajectories/my_trajectory.npy).")
    args = parser.parse_args()

    root = tk.Tk()
    root.title("Trajectory Drawer")
    app = LineSamplerApp(root, num_samples=args.samples, out_filename=args.out)
    root.mainloop()