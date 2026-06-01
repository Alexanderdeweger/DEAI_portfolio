import tkinter as tk
import time
import random
import csv
import os
import argparse

try:
    import numpy as np
except ImportError:
    np = None

WIDTH = 600
HEIGHT = 800
PLAYER_W = 60
PLAYER_H = 20
PLAYER_Y = HEIGHT - 40
PLAYER_SPEED = 300.0

OBSTACLE_SIZE = 30
BASE_SPAWN_INTERVAL = 1.0
MIN_SPAWN_INTERVAL = 0.2
SPAWN_ACCEL = 0.02

BASE_OBSTACLE_SPEED = 120.0
OBSTACLE_SPEED_INCREASE = 4.0

LOG_INTERVAL = 1.0 / 30.0
MAX_OBSTACLES_LOGGED = 3
DATA_FILE = "training_data.csv"


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


class DataLogger:
    def __init__(self, path):
        header = ["t", "player_x"]
        for i in range(MAX_OBSTACLES_LOGGED):
            header += [f"obs{i + 1}_x", f"obs{i + 1}_y"]

        mode = "a"
        new_file = False
        if os.path.exists(path):
            with open(path, "r", newline="") as existing:
                reader = csv.reader(existing)
                existing_header = next(reader, [])
            if existing_header != header:
                mode = "w"
                new_file = True
        else:
            mode = "w"
            new_file = True

        self.file = open(path, mode, newline="")
        self.writer = csv.writer(self.file)
        if new_file:
            self.writer.writerow(header)
            self.file.flush()
        self.last_flush = time.perf_counter()

    def log(self, t, player_x, obstacles):
        row = [f"{t:.3f}", f"{player_x:.4f}"]
        for i in range(MAX_OBSTACLES_LOGGED):
            if i < len(obstacles):
                row += [f"{obstacles[i][0]:.4f}", f"{obstacles[i][1]:.4f}"]
            else:
                row += ["-1", "-1"]
        self.writer.writerow(row)
        now = time.perf_counter()
        if now - self.last_flush >= 1.0:
            self.file.flush()
            self.last_flush = now

    def close(self):
        if self.file:
            self.file.flush()
            self.file.close()
            self.file = None


class NeuralPolicy:
    def __init__(self, w1, b1, w2, b2):
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2

    @classmethod
    def from_npz(cls, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        data = np.load(path, allow_pickle=True)
        w1 = data["W1"]
        b1 = data["b1"]
        w2 = data["W2"]
        b2 = data["b2"]
        return cls(w1, b1, w2, b2)

    def predict(self, features):
        x = np.array(features, dtype=np.float32)[None, :]
        z1 = np.tanh(x @ self.w1 + self.b1)
        output = z1 @ self.w2 + self.b2
        next_player_x = float(output[0, 0])
        next_player_x = np.clip(next_player_x, 0.0, 1.0)
        current_player_x = features[0]
        delta = next_player_x - current_player_x
        if abs(delta) < 0.01:
            return 0
        elif delta > 0:
            return 1
        else:
            return -1


class Obstacle:
    def __init__(self, canvas, x, y, size, speed):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.size = size
        self.speed = speed
        half = size / 2
        self.item = canvas.create_rectangle(
            x - half, y - half, x + half, y + half, fill="#ff4444", outline=""
        )

    def update(self, dt):
        self.y += self.speed * dt
        half = self.size / 2
        self.canvas.coords(
            self.item, self.x - half, self.y - half, self.x + half, self.y + half
        )

    def off_screen(self, height):
        return self.y - (self.size / 2) > height

    def collides_with_player(self, px, py, pw, ph):
        half = self.size / 2
        left = self.x - half
        right = self.x + half
        top = self.y - half
        bottom = self.y + half

        pleft = px - (pw / 2)
        pright = px + (pw / 2)
        ptop = py - (ph / 2)
        pbottom = py + (ph / 2)

        return not (right < pleft or left > pright or bottom < ptop or top > pbottom)


class Game:
    def __init__(self, ai_model=None):
        self.root = tk.Tk()
        self.root.title("PyFlow Training Game")
        self.canvas = tk.Canvas(self.root, width=WIDTH, height=HEIGHT, bg="#111111")
        self.canvas.pack()

        self.player_x = WIDTH / 2
        self.player_item = self.canvas.create_rectangle(0, 0, 0, 0, fill="#44aaff", outline="")
        self._update_player()

        self.lives = 3
        self.info_text = self.canvas.create_text(
            10, 10, anchor="nw", fill="white", font=("Arial", 12), text=self._info_text(0.0)
        )

        self.obstacles = []
        self.keys = {"Left": False, "Right": False}
        self.ai_model = ai_model

        self.root.bind("<KeyPress-Left>", lambda e: self._set_key("Left", True))
        self.root.bind("<KeyRelease-Left>", lambda e: self._set_key("Left", False))
        self.root.bind("<KeyPress-Right>", lambda e: self._set_key("Right", True))
        self.root.bind("<KeyRelease-Right>", lambda e: self._set_key("Right", False))
        self.root.bind("<Escape>", lambda e: self.on_close())

        self.start_time = time.perf_counter()
        self.last_time = self.start_time
        self.last_spawn = self.start_time
        self.last_log = self.start_time
        self.game_over = False

        self.logger = DataLogger(DATA_FILE)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _set_key(self, key, is_down):
        self.keys[key] = is_down

    def _current_action(self):
        if self.ai_model:
            return self.ai_model.predict(self._current_features())
        left = self.keys["Left"]
        right = self.keys["Right"]
        if left and not right:
            return -1
        if right and not left:
            return 1
        return 0

    def _update_player(self):
        half_w = PLAYER_W / 2
        half_h = PLAYER_H / 2
        self.canvas.coords(
            self.player_item,
            self.player_x - half_w,
            PLAYER_Y - half_h,
            self.player_x + half_w,
            PLAYER_Y + half_h,
        )

    def _info_text(self, elapsed):
        return f"Lives: {self.lives}  Time: {elapsed:.1f}s"

    def _spawn_interval(self, elapsed):
        return max(MIN_SPAWN_INTERVAL, BASE_SPAWN_INTERVAL - (elapsed * SPAWN_ACCEL))

    def _spawn_obstacle(self, elapsed):
        x = random.uniform(OBSTACLE_SIZE / 2, WIDTH - OBSTACLE_SIZE / 2)
        speed = BASE_OBSTACLE_SPEED + (elapsed * OBSTACLE_SPEED_INCREASE)
        self.obstacles.append(Obstacle(self.canvas, x, -OBSTACLE_SIZE / 2, OBSTACLE_SIZE, speed))

    def _current_features(self):
        obstacles_sorted = sorted(self.obstacles, key=lambda o: o.y, reverse=True)
        features = [self.player_x / WIDTH]
        for i in range(MAX_OBSTACLES_LOGGED):
            if i < len(obstacles_sorted):
                features.append(obstacles_sorted[i].x / WIDTH)
                features.append(obstacles_sorted[i].y / HEIGHT)
            else:
                features.extend([-1.0, -1.0])
        return features

    def _log_frame(self, elapsed):
        obstacles_sorted = sorted(self.obstacles, key=lambda o: o.y, reverse=True)
        obs_points = []
        for obs in obstacles_sorted[:MAX_OBSTACLES_LOGGED]:
            obs_points.append((obs.x / WIDTH, obs.y / HEIGHT))
        self.logger.log(elapsed, self.player_x / WIDTH, obs_points)

    def _end_game(self):
        self.game_over = True
        self.canvas.create_text(
            WIDTH / 2, HEIGHT / 2, text="Game Over", fill="white", font=("Arial", 32)
        )
        self.logger.close()

    def tick(self):
        if self.game_over:
            return

        now = time.perf_counter()
        dt = now - self.last_time
        self.last_time = now
        elapsed = now - self.start_time

        action = self._current_action()
        if action != 0:
            self.player_x += action * PLAYER_SPEED * dt
            self.player_x = clamp(self.player_x, PLAYER_W / 2, WIDTH - PLAYER_W / 2)
            self._update_player()

        interval = self._spawn_interval(elapsed)
        if now - self.last_spawn >= interval:
            self._spawn_obstacle(elapsed)
            self.last_spawn = now

        to_remove = []
        for obs in self.obstacles:
            obs.update(dt)
            if obs.off_screen(HEIGHT):
                to_remove.append(obs)
            elif obs.collides_with_player(self.player_x, PLAYER_Y, PLAYER_W, PLAYER_H):
                to_remove.append(obs)
                self.lives -= 1
                if self.lives <= 0:
                    self._end_game()
                    return

        for obs in to_remove:
            self.canvas.delete(obs.item)
            self.obstacles.remove(obs)

        self.canvas.itemconfig(self.info_text, text=self._info_text(elapsed))

        if now - self.last_log >= LOG_INTERVAL:
            self._log_frame(elapsed)
            self.last_log = now

        self.root.after(16, self.tick)

    def on_close(self):
        if not self.game_over:
            self.game_over = True
            self.logger.close()
        self.root.destroy()

    def run(self):
        self.tick()
        self.root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyFlow training game.")
    parser.add_argument(
        "--ai",
        nargs="?",
        const="model.npz",
        default=None,
        help="Enable AI with optional model path (default: model.npz).",
    )
    args = parser.parse_args()

    ai_model = None
    if args.ai:
        if np is None:
            raise SystemExit("NumPy is required for AI mode. Install with: pip install numpy")
        ai_model = NeuralPolicy.from_npz(args.ai)

    Game(ai_model=ai_model).run()

#python pyflow_game.py --ai model.npz