"""
train_agent.py  –  Train a small MLP on recorded gameplay logs.

Usage:
    python train_agent.py --log game_log.csv --out model.pt

The CSV columns are:
    time, player_x, points, lives, stones
    stones format: "x1:y1;x2:y2;..."

State vector fed to the network (fixed size):
    - player_x  (normalised 0-1)
    - For up to MAX_STONES stones: (stone_x_norm, stone_y_norm, active)
      Inactive slots are zero-padded.
    Total: 1 + MAX_STONES * 3 = 43 features

Label (action) is derived by comparing player_x in consecutive rows:
    moved left  -> 0
    stayed      -> 1
    moved right -> 2
"""

import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── constants ────────────────────────────────────────────────────────────────
WIDTH       = 440
HEIGHT      = 440
MAX_STONES  = 14
STATE_DIM   = 1 + MAX_STONES * 3   # 43
NUM_ACTIONS = 3                     # left, stay, right

# ── model ────────────────────────────────────────────────────────────────────

class AgentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_ACTIONS),
        )

    def forward(self, x):
        return self.net(x)

# ── helpers ──────────────────────────────────────────────────────────────────

def parse_stones(stones_str):
    """Return list of (x, y) floats from the CSV stones field."""
    if not stones_str or stones_str.strip() == '':
        return []
    pairs = []
    for token in stones_str.split(';'):
        token = token.strip()
        if ':' in token:
            sx, sy = token.split(':', 1)
            try:
                pairs.append((float(sx), float(sy)))
            except ValueError:
                pass
    return pairs


def build_state(player_x, stones):
    """Build a fixed-length float32 numpy array representing the game state."""
    vec = np.zeros(STATE_DIM, dtype=np.float32)
    vec[0] = player_x / WIDTH
    for i, (sx, sy) in enumerate(stones[:MAX_STONES]):
        base = 1 + i * 3
        vec[base]     = sx / WIDTH
        vec[base + 1] = sy / HEIGHT
        vec[base + 2] = 1.0   # slot active
    return vec


def load_dataset(log_path):
    """
    Read the CSV, build (state, action) pairs.
    Action is inferred from the change in player_x between successive rows.
    Rows where lives change (collision / reset) are skipped as labels.
    """
    rows = []
    with open(log_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if len(rows) < 2:
        raise ValueError("Log file has fewer than 2 rows – play more first!")

    states, labels = [], []

    for i in range(len(rows) - 1):
        r0, r1 = rows[i], rows[i + 1]

        # skip across life/reset boundaries
        if r0['lives'] != r1['lives']:
            continue

        px0 = float(r0['player_x'])
        px1 = float(r1['player_x'])
        stones = parse_stones(r0['stones'])
        state  = build_state(px0, stones)

        dx = px1 - px0
        if dx < -1:
            action = 0   # left
        elif dx > 1:
            action = 2   # right
        else:
            action = 1   # stay

        states.append(state)
        labels.append(action)

    if len(states) == 0:
        raise ValueError("No usable training pairs found in the log.")

    X = torch.tensor(np.array(states), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    return X, y

# ── training loop ─────────────────────────────────────────────────────────────

def train(log_path, out_path, epochs=50, batch_size=64, lr=1e-3):
    print(f"Loading data from {log_path} …")
    X, y = load_dataset(log_path)
    print(f"  {len(X)} samples | class counts: { {i: (y==i).sum().item() for i in range(3)} }")

    dataset = TensorDataset(X, y)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model     = AgentNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(yb)
            correct    += (logits.argmax(1) == yb).sum().item()
            total      += len(yb)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={total_loss/total:.4f}  acc={correct/total:.3f}")

    torch.save(model.state_dict(), out_path)
    print(f"\nModel saved to {out_path}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log',    default='game_log.csv', help='Path to game_log.csv')
    parser.add_argument('--out',    default='model.pt',     help='Where to save the trained model')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch',  type=int, default=64)
    parser.add_argument('--lr',     type=float, default=1e-3)
    args = parser.parse_args()

    if not os.path.exists(args.log):
        print(f"ERROR: log file '{args.log}' not found.")
        print("Play the game first to generate gameplay data:")
        print("  python game.py")
        raise SystemExit(1)

    train(args.log, args.out, epochs=args.epochs, batch_size=args.batch, lr=args.lr)